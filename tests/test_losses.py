import pytest
import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses._functional as F
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, SoftBCEWithLogitsLoss, SoftCrossEntropyLoss, PixelContrastLoss


def test_focal_loss_with_logits():
    input_good = torch.tensor([10, -10, 10]).float()
    input_bad = torch.tensor([-1, 2, 0]).float()
    target = torch.tensor([1, 0, 1])

    loss_good = F.focal_loss_with_logits(input_good, target)
    loss_bad = F.focal_loss_with_logits(input_bad, target)
    assert loss_good < loss_bad


def test_softmax_focal_loss_with_logits():
    input_good = torch.tensor([[0, 10, 0], [10, 0, 0], [0, 0, 10]]).float()
    input_bad = torch.tensor([[0, -10, 0], [0, 10, 0], [0, 0, 10]]).float()
    target = torch.tensor([1, 0, 2]).long()

    loss_good = F.softmax_focal_loss_with_logits(input_good, target)
    loss_bad = F.softmax_focal_loss_with_logits(input_bad, target)
    assert loss_good < loss_bad


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected", "eps"],
    [
        [[1, 1, 1, 1], [1, 1, 1, 1], 1.0, 1e-5],
        [[0, 1, 1, 0], [0, 1, 1, 0], 1.0, 1e-5],
        [[1, 1, 1, 1], [1, 1, 0, 0], 0.5, 1e-5],
    ],
)
def test_soft_jaccard_score(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_jaccard_score(y_pred, y_true, eps=eps)
    assert float(actual) == pytest.approx(expected, eps)


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected", "eps"],
    [
        [[[1, 1, 0, 0], [0, 0, 1, 1]], [[1, 1, 0, 0], [0, 0, 1, 1]], 1.0, 1e-5],
        [[[1, 1, 0, 0], [0, 0, 1, 1]], [[0, 0, 1, 0], [0, 1, 0, 0]], 0.0, 1e-5],
        [[[1, 1, 0, 0], [0, 0, 0, 1]], [[1, 1, 0, 0], [0, 0, 0, 0]], 0.5, 1e-5],
    ],
)
def test_soft_jaccard_score_2(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_jaccard_score(y_pred, y_true, dims=[1], eps=eps)
    actual = actual.mean()
    assert float(actual) == pytest.approx(expected, eps)


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected", "eps"],
    [
        [[1, 1, 1, 1], [1, 1, 1, 1], 1.0, 1e-5],
        [[0, 1, 1, 0], [0, 1, 1, 0], 1.0, 1e-5],
        [[1, 1, 1, 1], [1, 1, 0, 0], 2.0 / 3.0, 1e-5],
    ],
)
def test_soft_dice_score(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_dice_score(y_pred, y_true, eps=eps)
    assert float(actual) == pytest.approx(expected, eps)


@torch.no_grad()
def test_dice_loss_binary():
    eps = 1e-5
    criterion = DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=False)

    # Ideal case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 1, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)


@torch.no_grad()
def test_binary_jaccard_loss():
    eps = 1e-5
    criterion = JaccardLoss(mode=smp.losses.BINARY_MODE, from_logits=False)

    # Ideal case
    y_pred = torch.tensor([1.0]).view(1, 1, 1, 1)
    y_true = torch.tensor(([1])).view(1, 1, 1, 1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)


@torch.no_grad()
def test_multiclass_jaccard_loss():
    eps = 1e-5
    criterion = JaccardLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=False)

    # Ideal case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[0, 0, 1, 1]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[1, 1, 0, 0]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    # 1 - 1/3 case
    y_pred = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])
    y_true = torch.tensor([[1, 1, 0, 0]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0 - 1.0 / 3.0, abs=eps)


@torch.no_grad()
def test_multilabel_jaccard_loss():
    eps = 1e-5
    criterion = JaccardLoss(mode=smp.losses.MULTILABEL_MODE, from_logits=False)

    # Ideal case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = 1 - y_pred
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    # 1 - 1/3 case
    y_pred = torch.tensor([[[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0 - 1.0 / 3.0, abs=eps)


@torch.no_grad()
def test_soft_ce_loss():
    criterion = SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=-100)

    # Ideal case
    y_pred = torch.tensor([[+9, -9, -9, -9], [-9, +9, -9, -9], [-9, -9, +9, -9], [-9, -9, -9, +9]]).float()
    y_true = torch.tensor([0, 1, -100, 3]).long()

    loss = criterion(y_pred, y_true)
    print(loss)


@torch.no_grad()
def test_soft_bce_loss():
    criterion = SoftBCEWithLogitsLoss(smooth_factor=0.1, ignore_index=-100)

    # Ideal case
    y_pred = torch.tensor([-9, 9, 1, 9, -9]).float()
    y_true = torch.tensor([0, 1, -100, 1, 0]).long()

    loss = criterion(y_pred, y_true)
    print(loss)


@pytest.mark.parametrize(
    ["sampling", "segmentation_aware", "device"],
    [
        ["random", False, "cpu"],
        ["hardest", False, "cpu"],
        ["harder", False, "cpu"],
        ["harder", True, "cpu"],
        ["harder", True, "cuda"],
    ])
@torch.no_grad()
def test_pixel_contrast_loss(sampling, segmentation_aware, device):
    k_pos = 16
    k_neg = 32
    k_anchors = 8
    batch_size = 4
    embedding_dim = 128
    h = w = 256

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available")
        return

    criterion = PixelContrastLoss(n_classes=2, memory_size=256, embedding_dim=embedding_dim, pixels_per_image=10,
                                  k_pos=k_pos, k_neg=k_neg, k_anchors=k_anchors, sampling=sampling,
                                  segmentation_aware=segmentation_aware).to(device)

    # test batch
    emb = torch.rand((batch_size, embedding_dim, h, w)).to(device)
    y_true = (torch.rand((batch_size, 1, h, w)).to(device) > 0.5).long()
    y_pred = (torch.rand((batch_size, 1, h, w)).to(device) > 0.5).long()
    print("\nRandom accuracy:", torch.mean((y_true == y_pred).float()))

    # Fill memory with random embeddings
    criterion.memory_bank = torch.rand_like(criterion.memory_bank)
    loss = criterion(emb, y_true, y_pred)
    print(loss)
