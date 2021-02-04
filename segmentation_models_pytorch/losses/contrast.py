import torch
from torch import Tensor as T
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from typing import Optional, Tuple
from einops import rearrange, reduce


class PixelContrastLoss(_Loss):
    def __init__(self, n_classes: int, memory_size: int, embedding_dim: int,
                 pixels_per_image: int = 10, k_pos: int = 1024, k_neg: int = 2048, k_anchors: int = 1024,
                 temperature: float = 0.1,
                 sampling: str = "harder", segmentation_aware: bool = True, allocate_samples: bool = False,
                 reduction: str = "mean"):
        super(PixelContrastLoss, self).__init__(reduction=reduction)

        assert sampling in ("harder", "hardest", "random")

        self.n_classes = n_classes
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.pixels_per_image = pixels_per_image
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.k_anchors = k_anchors
        self.temperature = temperature
        self.sampling = sampling
        self.segmentation_aware = segmentation_aware
        self.allocate_samples = allocate_samples

        # Memory bank
        self.register_buffer(
            "memory_bank",
            torch.zeros((n_classes,
                         memory_size,
                         pixels_per_image + 1,
                         embedding_dim),
                        dtype=torch.float32),
        )

        # Flags to maintain memory integrity
        self.memory_full = False
        self.counter = 0

    def forward(self, emb: T, y_true: T, y_pred: T) -> T:
        batch_size, *_ = emb.size()
        # Populate region and pixel memory
        for batch_idx, (emb_i, y) in enumerate(zip(emb, y_true)):
            # get index in memory bank
            memory_idx = self.counter % self.memory_size
            # get embeddings from classes separately
            for class_idx in range(self.n_classes):
                mask_idx = torch.where(y[0] == class_idx)
                class_emb = emb_i[:, mask_idx[0], mask_idx[1]].T  # pixels_in_class x emb_size
                # skip the class if there are just a few pixels
                if class_emb.size(0) > self.pixels_per_image:
                    rand_idx = torch.randperm(class_emb.size(0))[:self.pixels_per_image]
                    self.memory_bank[class_idx, memory_idx, 0] = class_emb.mean(dim=0).detach()
                    self.memory_bank[class_idx, memory_idx, 1:] = class_emb[rand_idx].detach()

            # increment counter and check if we did a full round on memory bank
            self.counter += 1
            if not self.memory_full and self.counter > self.memory_size:
                self.memory_full = True

        anchors, classes = self._sample_anchors(emb=emb, k=batch_size * self.k_anchors, y_true=y_true, y_pred=y_pred)

        loss = []
        # allocating samples can consume a lot of memory but allows to calculate loss at once
        if self.allocate_samples:
            positive_samples = torch.zeros(batch_size * self.k_anchors, self.k_pos, self.embedding_dim).to(emb.device)
            negative_samples = torch.zeros(batch_size * self.k_anchors, self.k_neg, self.embedding_dim).to(emb.device)
        else:
            positive_samples = negative_samples = None

        for i, (anchor, class_idx) in enumerate(zip(anchors, classes.long())):
            try:
                pos, neg = self._sample_pos_neg(emb=anchor, class_idx=class_idx)
            except RuntimeError:
                print(classes[classes != 0])
                raise RuntimeError

            # Calculate loss right away or populate sample tensor
            if self.allocate_samples:
                positive_samples[i], negative_samples[i] = pos, neg
            else:
                loss.append(self._info_nce(emb=anchor[None, ...],
                                           pos=pos[None, ...],
                                           neg=neg[None, ...]).mean())

        # Calculate loss over collected samples or average results of online calculations
        if self.allocate_samples:
            loss = self._info_nce(emb=anchors, pos=positive_samples, neg=negative_samples).mean()
        else:
            loss = torch.stack(loss).mean()

        return loss

    def _info_nce(self, emb: T, pos: T, neg: T):
        # b - batch size, d - embedding dim, k - examples
        pos_dot = torch.einsum("bd,bkd->bk", emb, pos) / self.temperature
        neg_dot = torch.einsum("bd,bkd->bk", emb, neg) / self.temperature
        max_val = torch.max(torch.cat([pos_dot, neg_dot], dim=-1))

        numerator = torch.exp(pos_dot - max_val)
        denominator = reduce(torch.exp(neg_dot - max_val), "b k -> b ()", "sum") + numerator
        loss = -torch.log((numerator / denominator) + 1e-8)
        loss = reduce(loss, "b k -> b", "mean")
        if self.reduction is not None:
            loss = reduce(loss, "b -> ()", self.reduction)
        return loss

    def _sample_top_k(self, embedding: T, class_idx: int, top_k: int, positive: bool):
        """
        Sample from the memory top-K closest negative or farthest positives to the given embedding

        Args:
            embedding: Tensor, embedding vector (1, embedding_dim)
            class_idx: int, embedding label
            top_k: int, to sample from memory
            positive: bool, if sample positive (same class) or negative (other classes)

        Returns: Tensor (k, embedding_dim)
        """
        memory = self._get_memory(class_idx=class_idx, positive=positive)
        # measure distance to embedding from memory
        dst = torch.norm(memory - embedding, dim=1, p=None)
        # sample nearest negatives or farthest positives
        try:
            # sample either top k or full available memory
            knn = dst.topk(min(top_k, memory.size(0)), largest=positive)
        except RuntimeError:
            print(f"\nFailed to sample top {top_k} from dst={dst.size()}, "
                  f"memory={memory.size()}, class_idx={class_idx}, positive={positive}")
            raise RuntimeError
        return memory[knn.indices]

    def _sample_pos_neg(self, emb: T, class_idx: int, k_pos: Optional[int] = None, k_neg: Optional[int] = None, sampling: str = None) -> Tuple[T, T]:

        # we might want to use custom values
        if k_pos is None:
            k_pos = self.k_pos
        if k_neg is None:
            k_neg = self.k_neg
        if sampling is None:
            sampling = self.sampling

        if sampling == "random":
            return self._sample_random(class_idx=class_idx, k_pos=k_pos, k_neg=k_neg)
        elif sampling == "harder":
            return self._sample_harder(emb=emb, class_idx=class_idx, k_pos=k_pos, k_neg=k_neg)
        elif sampling == "hardest":
            return self._sample_hardest(emb=emb, class_idx=class_idx, k_pos=k_pos, k_neg=k_neg)
        else:
            raise ValueError("Unknown sampling type")

    def _get_memory(self, class_idx: int, positive: bool):
        # class class indices to sample
        cls = torch.arange(self.n_classes).to(self.memory_bank.device)
        idx = (cls == class_idx) if positive else (cls != class_idx)
        # get embeddings from the memory and flatten
        return self.memory_bank[idx, :self.counter].view(-1, self.embedding_dim)

    def _sample_random_one(self, class_idx: int, top_k: int, positive: bool):
        memory = self._get_memory(class_idx=class_idx, positive=positive)
        memory_idx = torch.randperm(memory.size(0))[:top_k]
        return memory[memory_idx]

    def _sample_random(self, class_idx: int, k_pos: Optional[int] = None, k_neg: Optional[int] = None) -> Tuple[T, T]:
        # we might want to use custom values
        if k_pos is None:
            k_pos = self.k_pos
        if k_neg is None:
            k_neg = self.k_neg

        pos = self._sample_random_one(class_idx, k_pos, positive=True)
        neg = self._sample_random_one(class_idx, k_neg, positive=False)

        return pos, neg

    def _sample_hardest(self, emb: T, class_idx: int, k_pos: Optional[int] = None, k_neg: Optional[int] = None) -> Tuple[T, T]:
        # we might want to use custom values
        if k_pos is None:
            k_pos = self.k_pos
        if k_neg is None:
            k_neg = self.k_neg

        pos = self._sample_top_k(embedding=emb, class_idx=class_idx, top_k=k_pos, positive=True)
        neg = self._sample_top_k(embedding=emb, class_idx=class_idx, top_k=k_neg, positive=False)

        return pos, neg

    def _sample_harder(self, emb: T, class_idx: int, k_pos: Optional[int] = None, k_neg: Optional[int] = None) -> Tuple[T, T]:
        # we might want to use custom values
        if k_pos is None:
            k_pos = self.k_pos
        if k_neg is None:
            k_neg = self.k_neg

        ten_pct = int(self.memory_size * 0.1)
        pos, neg = self._sample_hardest(emb=emb, class_idx=class_idx,
                                        k_pos=max(ten_pct, k_pos),
                                        k_neg=max(ten_pct, k_neg))

        # randomly sample positives and negatives
        pos_idx = torch.randperm(min(pos.size(0), max(ten_pct, k_pos)))[:k_pos]
        neg_idx = torch.randperm(min(neg.size(0), max(ten_pct, k_neg)))[:k_neg]

        return pos[pos_idx], neg[neg_idx]

    def _sample_anchors(self, emb: T, k: int, y_true: Optional[T] = None, y_pred: Optional[T] = None):
        segmentation_aware = self.segmentation_aware and y_pred is not None
        # flatten embeddings and labels in the same way
        emb_flat = rearrange(emb, "b c h w -> (b h w) c")
        y_true_flat = rearrange(y_true, "b c h w -> (b h w) c")
        # sample randomly
        idx = torch.randperm(emb_flat.size(0))[:k]
        if segmentation_aware:
            # try to replace half the indices with mislabeled regions
            y_pred_flat = rearrange(y_true, "b c h w -> (b h w) c")
            idx_hard = torch.where(y_true_flat != y_pred_flat)[0]
            k_hard = int(min(k * 0.5, len(idx_hard)))
            if k_hard > 0:
                idx[:k_hard] = idx_hard[torch.randperm(len(idx_hard))][:k_hard]

        anchors = emb_flat[idx]
        classes = y_true_flat[idx]

        return anchors, classes









