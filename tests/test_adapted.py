import torch

from ecgtok.align import FRAMES, PATCH, PatchAlign
from ecgtok.fsq import VOCAB, fsq_ids


def test_fsq_ids_use_base_k_digits() -> None:
    levels = 5
    latent = torch.tensor([[-40.0, -40.0, 40.0, 40.0]])
    ste, ids = fsq_ids(latent, levels)
    assert ste.shape == (1, 4)
    assert int(ids) == (levels - 1) * levels**2 + (levels - 1) * levels**3
    assert 0 <= int(ids) < VOCAB


def test_fsq_straight_through_carries_gradient() -> None:
    latent = torch.nn.Parameter(torch.zeros(2, 4))
    ste, _ids = fsq_ids(latent, 5)
    ste.sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_align_keeps_prototypes_fixed() -> None:
    hidden = 32
    model = PatchAlign(hidden)
    model.prototypes.copy_(torch.randn(model.prototypes.shape))
    windows = torch.randn(3, FRAMES * PATCH)
    out = model(windows)
    assert out.shape == (3, FRAMES, hidden)
    out.sum().backward()
    assert model.embed.weight.grad is not None
    assert model.prototypes.grad is None
