import torch
from anchormil import AnchorMIL, AnchorMILLoss


def test_forward_backward():
    m = AnchorMIL(in_dim=32, dim=32, num_anchors=8, num_heads=4, num_blocks=2)
    x = torch.randn(3, 20, 32)
    mask = torch.zeros(3, 20, dtype=torch.bool)
    mask[1, 10:] = True
    o = m(x, mask)
    assert o["r"].shape == (3, 8) and o["reg"].shape == (3,)
    assert torch.allclose(o["c"].sum(-1), torch.ones(3), atol=1e-5)
    loss, _ = AnchorMILLoss()(o, torch.tensor([10.0, 30.0, 55.0]), m.anchor_scores)
    loss.backward()
    assert torch.isfinite(loss)


def test_padding_invariance():
    m = AnchorMIL(in_dim=16, dim=16, num_anchors=4, num_heads=2).eval()
    x = torch.randn(1, 6, 16)
    xp = torch.cat([x, torch.randn(1, 4, 16)], 1)
    mask = torch.tensor([[False] * 6 + [True] * 4])
    assert torch.allclose(m(x)["reg"], m(xp, mask)["reg"], atol=1e-4)


def test_shared_cross_attn():
    m = AnchorMIL(in_dim=16, dim=16, num_anchors=4, num_heads=2, num_blocks=2)
    assert m.blocks[0].cross is m.blocks[1].cross
