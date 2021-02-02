import torch
from torch import nn, Tensor as T
from torch.nn.modules.loss import _Loss
from typing import Optional, Tuple
from einops import rearrange, reduce


class PixelContrastLoss(_Loss):
    def __init__(self, n_classes: int, memory_size: int, embedding_dim: int,
                 pixels_per_image: int = 10, k_pos: int = 1024, k_neg: int = 2048, k_anchors: int = 1024,
                 temperature: float = 0.1,
                 sampling: str = "harder", segmentation_aware: bool = True,
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

        # Memory bank
        self.memory_bank = torch.zeros((n_classes, memory_size, pixels_per_image + 1, embedding_dim), dtype=torch.float32)

        # Flags to maintain memory integrity
        self.memory_full = False
        self.counter = 0

    def forward(self, batch_emb: T, y_true: T, y_pred: T) -> T:
        memory_idx = self.counter % self.memory_size
        batch_size, emb_size, h, w = batch_emb.size()

        # Populate region and pixel memory
        for batch_idx, (emb, y) in enumerate(zip(batch_emb, y_true)):
            for class_idx in range(self.n_classes):
                mask_idx = torch.where(y[0] == class_idx)
                class_emb = emb[:, mask_idx[0], mask_idx[1]].T  # pixels_in_class x emb_size
                rand_idx = torch.randperm(class_emb.size(0))[:self.pixels_per_image]
                self.memory_bank[class_idx, memory_idx+batch_idx, 0] = class_emb.mean(dim=0).detach().cpu()
                self.memory_bank[class_idx, memory_idx+batch_idx, 1:] = class_emb[rand_idx].detach().cpu()

        self.counter += batch_size
        if not self.memory_full and self.counter > self.memory_size:
            self.memory_full = True

        anchors, classes = self._sample_anchors(batch_emb, self.k_anchors, y_true, y_pred)
        loss = []
        for anchor, class_idx in zip(anchors.view(-1, self.embedding_dim), classes.view(-1, 1)):
            pos, neg = self._sample(anchor, class_idx.item())
            pos, neg = pos.to(anchor.device), neg.to(anchor.device)
            loss.append(self._info_nce(anchor, pos, neg))

        loss = torch.stack(loss)
        if self.reduction is not None:
            loss = reduce(loss, "b k -> k", self.reduction)

        return loss

    def _info_nce(self, emb: T, pos: T, neg: T):
        # b - batch size, d - embedding dim, k - examples, l - flattened hxw of embedding
        pos_dot = torch.einsum("d,kd->k", emb, pos) / self.temperature
        neg_dot = torch.einsum("d,kd->k", emb, neg) / self.temperature
        max_val = torch.max(torch.cat([pos_dot, neg_dot]))

        numerator = torch.exp(pos_dot - max_val)
        denominator = reduce(torch.exp(neg_dot - max_val), "k -> ()", "sum") + numerator
        loss = -torch.log((numerator / denominator) + 1e-8)

        loss = reduce(loss, "k -> ()", "mean")
        return loss

    def _sample_top_k(self, emb: T, class_idx: int, top_k: int, positive: bool):
        """
        Sample from the memory top-K closest negative or farthest positives

        Args:
            emb: Tensor, embedding vector (1, embedding_dim)
            class_idx: int, embedding label
            top_k: int, to sample from memory
            positive: bool, if sample positive (same class) or negative (other classes)

        Returns: Tensor (k, embedding_dim)
        """
        memory = self._get_memory(class_idx, positive)
        # measure distance to embedding from memory
        dst = torch.norm(emb.detach().cpu() - memory, dim=1, p=None)
        # sample nearest negatives or farthest positives
        knn = dst.topk(top_k, largest=positive)
        return memory[knn.indices]

    def _sample(self, emb: T, class_idx: int, k_pos: Optional[int] = None, k_neg: Optional[int] = None, sampling: str = None):
        # we might want to use custom values
        if k_pos is None:
            k_pos = self.k_pos
        if k_neg is None:
            k_neg = self.k_neg
        if sampling is None:
            sampling = self.sampling

        if sampling == "random":
            return self._sample_random(class_idx, k_pos, k_neg)
        if sampling == "harder":
            return self._sample_harder(emb, class_idx, k_pos, k_neg)
        if sampling == "hardest":
            return self._sample_hardest(emb, class_idx, k_pos, k_neg)

    def _get_memory(self, class_idx: int, positive: bool):
        # class class indices to sample
        cls = torch.arange(self.n_classes)
        idx = (cls == class_idx) if positive else (cls != class_idx)
        # get embeddings from the memory and flatten
        ref = rearrange(self.memory_bank[idx],
                        "c n m d -> (c n m) d",
                        d=self.embedding_dim,
                        n=self.memory_size,
                        m=self.pixels_per_image + 1)  # .view(-1, self.embedding_dim)
        return ref

    def _sample_random_one(self, class_idx: int, top_k: int, positive: bool):
        memory = self._get_memory(class_idx, positive)
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

        pos = self._sample_top_k(emb, class_idx, k_pos, positive=True)
        neg = self._sample_top_k(emb, class_idx, k_neg, positive=False)

        return pos, neg

    def _sample_harder(self, emb: T, class_idx: int, k_pos: Optional[int] = None, k_neg: Optional[int] = None) -> Tuple[T, T]:
        # we might want to use custom values
        if k_pos is None:
            k_pos = self.k_pos
        if k_neg is None:
            k_neg = self.k_neg

        ten_pct = int(self.memory_size * 0.1)
        pos, neg = self._sample_hardest(emb, class_idx, k_pos=ten_pct, k_neg=ten_pct)

        # randomly sample positives and negatives
        pos_idx = torch.randperm(ten_pct)[:k_pos]
        neg_idx = torch.randperm(ten_pct)[:k_neg]

        return pos[pos_idx], neg[neg_idx]

    def _sample_anchors_one(self, emb: T, k: int, y_true: T, y_pred: Optional[T] = None):
        segmentation_aware = self.segmentation_aware and y_pred is not None
        k_random = k // 2 if segmentation_aware else k
        emb_flat = rearrange(emb, "d h w -> (h w) d")
        y_true_flat = rearrange(y_true, "d h w -> (h w) d")

        idx = torch.randperm(emb_flat.size(0))[:k_random]
        anchors = emb_flat[idx]
        classes = y_true_flat[idx]

        # TODO we could occasionally sample twice
        if segmentation_aware:
            y_pred_flat = rearrange(y_pred, "d h w -> (h w) d")
            hard_idx = torch.where(y_true_flat != y_pred_flat)[0]

            k_random = min(k_random, len(hard_idx))  # safe check
            idx = torch.randperm(len(hard_idx))[:k_random]
            anchors = torch.cat([anchors, emb_flat[idx]], dim=0)
            classes = torch.cat([classes, y_true_flat[idx]], dim=0)

        return anchors, classes

    def _sample_anchors(self, emb: T, k: int, y_true: Optional[T] = None, y_pred: Optional[T] = None):
        segmentation_aware = self.segmentation_aware and y_pred is not None

        anchors = []
        classes = []

        for i, e in enumerate(emb):
            anc, cls = self._sample_anchors_one(e, k, y_true[i], y_pred[i] if segmentation_aware else None)
            anchors.append(anc)
            classes.append(cls)

        anchors = torch.stack(anchors, dim=0)
        classes = torch.stack(classes, dim=0)

        return anchors, classes









