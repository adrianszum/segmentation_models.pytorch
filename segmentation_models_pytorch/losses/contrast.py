from typing import Optional, Tuple

import torch
from einops import rearrange, reduce
from torch import Tensor as T
from torch.nn.modules.loss import _Loss


class PixelContrastLoss(_Loss):
    """
    Exploring Cross-Image Pixel Contrast for Semantic Segmentation
    https://arxiv.org/pdf/2101.11939.pdf
    """

    def __init__(
        self,
        n_classes: int,
        memory_size: int,
        embedding_dim: int,
        pixels_per_image: int = 10,
        k_pos: int = 1024,
        k_neg: int = 2048,
        k_anchors: int = 8,
        temperature: float = 0.1,
        sampling: str = "harder",
        segmentation_aware: bool = True,
        hard_anchors_proportion: float = 0.5,
        region_anchors: bool = True,
        allocate_samples: bool = True,
        reduction: str = "mean",
    ):
        """

        Args:
            n_classes: int, number of classes in the dataset
            memory_size: int, size of the memory bank (usually equals to dataset size)
            embedding_dim: int, length of embedding vector
            pixels_per_image: int, pixel embeddings to sample randomly from each training image to the bank;
                              if 0, then only region memory will be used
            k_pos: int, k positive samples
            k_neg: int, k negative samples
            k_anchors: int, k anchors to sample from each training image for the loss computation
            temperature: float, temperature (tau) parameter of InfoNCE loss
            sampling: str {"random", "harder", "hardest"}, see _sample_pos_neg() for details
            segmentation_aware: bool, if sample from mislabeled pixels, see _sample_anchors() for details
            hard_anchors_proportion: float [0, 1], proportion of mislabeled pixels in anchor sampling
            region_anchors: bool, if sample C anchors as average embeddings from each class
            allocate_samples: bool, if allocate space for samples and calculate
                              loss for the batch (consumes memory but faster)
            reduction: loss reduction in batch, default "mean"
        """
        super(PixelContrastLoss, self).__init__(reduction=reduction)

        assert sampling in ("harder", "hardest", "random")
        assert 0 < hard_anchors_proportion < 1

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
        self.hard_anchors_proportion = hard_anchors_proportion
        self.region_anchors = region_anchors
        self.allocate_samples = allocate_samples

        # memory bank
        self.register_buffer(
            "memory_bank",
            torch.zeros(
                (n_classes, memory_size, pixels_per_image + 1, embedding_dim),
                dtype=torch.float32,
            ),
        )

        # flags to maintain memory integrity
        self.memory_full = False
        self.counter = 0

    def forward(self, emb: T, y_true: T, y_pred: Optional[T] = None) -> T:
        """
        Forward pass of pixel contrast loss computation
        Args:
            emb: float tensor (B, D, H, W)
            y_true: long tensor (B, 1, H, W)
            y_pred: long tensor (B, 1, H, W)

        Returns: float tensor loss

        """
        batch_size, *_ = emb.size()

        # Populate region and pixel memory
        for batch_idx, (emb_i, y) in enumerate(zip(emb, y_true)):
            # get index in memory bank
            m = self.counter % self.memory_size

            for c in range(self.n_classes):
                # get embeddings from the current class
                mask = torch.where(y[0] == c)
                emb_c = emb_i[:, mask[0], mask[1]].T
                # skip the class if there are just no pixels
                if emb_c.size(0) == 0:
                    continue
                # add mean region (class) embedding
                self.memory_bank[c, m, 0] = emb_c.mean(dim=0).detach()
                # add random pixel embeddings to pixel memory
                if 0 < self.pixels_per_image < emb_c.size(0):
                    rand_idx = torch.randperm(emb_c.size(0))[: self.pixels_per_image]
                    self.memory_bank[c, m, 1:] = emb_c[rand_idx].detach()

            # increment counter and check if we did a full round on memory bank
            self.counter += 1
            if not self.memory_full and self.counter > self.memory_size:
                self.memory_full = True

        # sample k anchors from each batch image (exactly k from image not guaranteed)
        k_anchors = batch_size * self.k_anchors
        # if self.region_anchors:
        #     k_anchors -= batch_size * self.n_classes
        anchors, classes = self._sample_anchors(
            emb=emb, k=k_anchors, y_true=y_true, y_pred=y_pred
        )

        loss = []
        # allocating samples can consume memory but allows to calculate loss for all anchors at once
        if self.allocate_samples:
            positive_samples = []
            negative_samples = []
        else:
            positive_samples = negative_samples = None

        # sample positive and negative examples for each anchor
        for i, (anchor, c) in enumerate(zip(anchors, classes.long())):
            try:
                pos, neg = self._sample_pos_neg(emb=anchor, class_idx=c)
            except RuntimeError:
                print(classes[classes != 0])
                raise RuntimeError

            # calculate loss right away or populate sample tensor
            if self.allocate_samples:
                positive_samples.append(pos)
                negative_samples.append(neg)
            else:
                loss.append(
                    self._info_nce(
                        emb=anchor[None, ...], pos=pos[None, ...], neg=neg[None, ...]
                    ).mean()
                )

        # calculate loss over collected samples or average results of online calculations
        if self.allocate_samples:
            loss = self._info_nce(
                emb=anchors,
                pos=torch.stack(positive_samples, dim=0),
                neg=torch.stack(negative_samples, dim=0)
            ).mean()
        else:
            loss = torch.stack(loss).mean()

        return loss

    def _info_nce(self, emb: T, pos: T, neg: T) -> T:
        """
        Calculate InfoNCE loss
        Args:
            emb: float tensor (B, D)
            pos: float tensor (B, K, D) of positive examples
            neg: float tensor (B, K, D) of negative examples

        Returns: float tensor loss
        """
        # calculate dot products
        # b - batch size, d - embedding dim, k - examples
        pos_dot = torch.einsum("bd,bkd->bk", emb, pos) / self.temperature
        neg_dot = torch.einsum("bd,bkd->bk", emb, neg) / self.temperature
        # deduct maximal value for numerical stability of exponents
        pos_max_val = torch.max(pos_dot)
        neg_max_val = torch.max(neg_dot)
        max_val = torch.max(torch.stack([pos_max_val, neg_max_val]))

        numerator = torch.exp(pos_dot - max_val)
        denominator = (
            reduce(torch.exp(neg_dot - max_val), "b k -> b ()", "sum") + numerator
        )
        loss = -torch.log((numerator / denominator) + 1e-8)
        loss = reduce(loss, "b k -> b", "mean")
        if self.reduction is not None:
            loss = reduce(loss, "b -> ()", self.reduction)
        return loss

    def _sample_anchors(
        self, emb: T, k: int, y_true: T, y_pred: Optional[T] = None
    ) -> Tuple[T, T]:
        """
        Sample `k` anchors from the given embedding batch.
        May work in segmentation-aware mode given `y_pred`: half the anchors will be sampled from mislabeled areas
        Args:
            emb: float tensor (B, D, H, W) to sample from
            k: int, number of anchors to sample
            y_true: long tensor (B, 1, H, W) of true labels
            y_pred: long tensor (B, 1, H, W) of predicted labels

        Returns: float tensor (K, D) of anchor embeddings, long tensor (K, 1) of classes

        """
        # segmentation-aware mode is active only if predictions were given
        segmentation_aware = self.segmentation_aware and y_pred is not None

        # flatten embeddings and labels in the same way
        emb_flat = rearrange(emb, "b d h w -> (b h w) d")
        y_true_flat = rearrange(y_true, "b d h w -> (b h w) d")

        # sample k random indices from embedding
        idx = torch.randperm(emb_flat.size(0))[:k]
        if segmentation_aware:
            # try to replace half the indices with the ones from mislabeled regions
            y_pred_flat = rearrange(y_true, "b d h w -> (b h w) d")
            idx_hard = torch.where(y_true_flat != y_pred_flat)[0]
            # there can be fewer hard indices than k/2, so we take min
            # TODO resampling could occur here
            k_hard = int(min(k * self.hard_anchors_proportion, len(idx_hard)))
            if k_hard > 0:
                idx[:k_hard] = idx_hard[torch.randperm(len(idx_hard))][:k_hard]

        anchors = emb_flat[idx]
        classes = y_true_flat[idx]

        # get average embeddings of class
        if self.region_anchors:
            emb_flat = rearrange(emb, "b d h w -> b (h w) d")
            y_true_flat = rearrange(y_true, "b d h w -> b (h w) d")

            region_anchors = []
            region_classes = []
            for c in range(self.n_classes):
                # get embeddings from the current class
                for emb_c, y in zip(emb_flat, y_true_flat):
                    mask = torch.where(y == c)[0]
                    emb_c = emb_c[mask, :]
                    if emb_c.size(0) > 0:
                        region_anchors.append(emb_c.mean(dim=0, keepdim=True))
                        region_classes.append(torch.tensor(c))

            region_anchors = torch.cat(region_anchors, dim=0)
            region_classes = torch.stack(region_classes, dim=0)[..., None].to(emb.device)

            try:
                anchors = torch.cat([anchors, region_anchors], dim=0)
                classes = torch.cat([classes, region_classes], dim=0)
            except RuntimeError:
                print(f"anchors: {anchors.shape}, {region_anchors.shape};"
                      f"classes: {classes.shape}, {region_classes.shape}")
                raise RuntimeError

        assert anchors.size(0) == classes.size(0)
        return anchors, classes

    def _sample_from_memory(self, class_idx: int, positive: bool) -> T:
        """
        Sample from memory bank positive or negative examples for the given class
        Args:
            class_idx: int, class index
            positive: bool, if take the given class or all but the given class

        Returns: tensor (P, D), where P = C x (M + 1) x N, C = 1 if positive else (N_classes - 1)

        """
        # class indices to sample (get all but the given one for negative sampling)
        cls = torch.arange(self.n_classes).to(self.memory_bank.device)
        idx = (cls == class_idx) if positive else (cls != class_idx)
        # get embeddings from the memory and flatten
        return self.memory_bank[idx, : self.counter].view(-1, self.embedding_dim)

    def _sample_random_k(self, class_idx: int, k: int, positive: bool) -> T:
        """
        Sample k random positive or negative examples from the memory bank
        Args:
            class_idx: int, class index
            k: int
            positive: bool, if take the given class or all but the given class

        Returns: tensor (K, D)

        """
        memory = self._sample_from_memory(class_idx=class_idx, positive=positive)
        memory_idx = torch.randperm(memory.size(0))[:k]
        return memory[memory_idx]

    def _sample_top_k(self, embedding: T, class_idx: int, k: int, positive: bool) -> T:
        """
        Sample from the memory top-K closest negative or farthest positives to the given embedding

        Args:
            embedding: Tensor, embedding vector (1, embedding_dim)
            class_idx: int, embedding label
            k: int, to sample from memory
            positive: bool, if sample positive (same class) or negative (other classes)

        Returns: Tensor (k, embedding_dim)
        """
        memory = self._sample_from_memory(class_idx=class_idx, positive=positive)
        # measure distance to embedding from memory
        # TODO customize distance function
        dst = torch.norm(memory - embedding, dim=1, p=None)
        # sample nearest negatives or farthest positives
        try:
            # sample min value from top k or available memory size
            knn = dst.topk(min(k, memory.size(0)), largest=positive)
        except RuntimeError:
            print(
                f"\nFailed to sample top {k} from dst={dst.size()}, "
                f"memory={memory.size()}, class_idx={class_idx}, positive={positive}"
            )
            raise RuntimeError
        return memory[knn.indices]

    def _sample_pos_neg(
        self,
        emb: T,
        class_idx: int,
        k_pos: Optional[int] = None,
        k_neg: Optional[int] = None,
        sampling: str = None,
    ) -> Tuple[T, T]:
        """
        Sample positive and negative examples for the given embedding
            - random: _sample_random(), take random instances from the memory bank
            - hardest: _sample_hardest(), take k closest negatives or farthest positives
            - harder: _sample_harder(), take 10% closest negatives or farthest positives, sample random k from them

        Args:
            emb: float tensor (1, C) of a single embedding
            class_idx: int, true class of an embedding
            k_pos: int, k positive samples
            k_neg: int, k negative samples
            sampling: str {"random", "harder", "hardest"}

        Returns: float tensor (K_pos, D), float tensor (K_neg, D)

        """

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
            return self._sample_harder(
                emb=emb, class_idx=class_idx, k_pos=k_pos, k_neg=k_neg
            )
        elif sampling == "hardest":
            return self._sample_hardest(
                emb=emb, class_idx=class_idx, k_pos=k_pos, k_neg=k_neg
            )
        else:
            raise ValueError("Unknown sampling type")

    def _sample_random(
        self, class_idx: int, k_pos: Optional[int] = None, k_neg: Optional[int] = None
    ) -> Tuple[T, T]:
        """
        Sample random positive and negative examples from the memory bank
        Args:
            class_idx: int, true class of an embedding
            k_pos: int
            k_neg: int

        Returns: float tensor (K_pos, D), float tensor (K_neg, D)

        """
        # we might want to use custom values
        if k_pos is None:
            k_pos = self.k_pos
        if k_neg is None:
            k_neg = self.k_neg

        pos = self._sample_random_k(class_idx, k_pos, positive=True)
        neg = self._sample_random_k(class_idx, k_neg, positive=False)

        return pos, neg

    def _sample_hardest(
        self,
        emb: T,
        class_idx: int,
        k_pos: Optional[int] = None,
        k_neg: Optional[int] = None,
    ) -> Tuple[T, T]:
        """
        Sample k hardest examples from the memory bank (farthest positives and closest negatives)
        Args:
            emb: float tensor (1, C) of a single embedding
            class_idx: int, true class of an embedding
            k_pos: int, k positive samples
            k_neg: int, k negative samples

        Returns: float tensor (K_pos, D), float tensor (K_neg, D)

        """

        # we might want to use custom values
        if k_pos is None:
            k_pos = self.k_pos
        if k_neg is None:
            k_neg = self.k_neg

        pos = self._sample_top_k(
            embedding=emb, class_idx=class_idx, k=k_pos, positive=True
        )
        neg = self._sample_top_k(
            embedding=emb, class_idx=class_idx, k=k_neg, positive=False
        )

        return pos, neg

    def _sample_harder(
        self,
        emb: T,
        class_idx: int,
        k_pos: Optional[int] = None,
        k_neg: Optional[int] = None,
    ) -> Tuple[T, T]:
        """
        Sample 10% hardest examples from the memory bank (farthest positives and closest negatives),
        then take k random from them.
        This sampling works best according to the paper.
        Args:
            emb: float tensor (1, C) of a single embedding
            class_idx: int, true class of an embedding
            k_pos: int, k positive samples
            k_neg: int, k negative samples

        Returns: float tensor (K_pos, D), float tensor (K_neg, D)

        """
        # we might want to use custom values
        if k_pos is None:
            k_pos = self.k_pos
        if k_neg is None:
            k_neg = self.k_neg

        ten_pct = int(self.memory_size * 0.1)
        pos, neg = self._sample_hardest(
            emb=emb,
            class_idx=class_idx,
            k_pos=max(ten_pct, k_pos),
            k_neg=max(ten_pct, k_neg),
        )

        # randomly sample positives and negatives
        pos_idx = torch.randperm(min(pos.size(0), max(ten_pct, k_pos)))[:k_pos]
        neg_idx = torch.randperm(min(neg.size(0), max(ten_pct, k_neg)))[:k_neg]

        return pos[pos_idx], neg[neg_idx]
