import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorMILLoss(nn.Module):
    """L = beta * L_reg + L_conf + L_BCE  (Sec. 3.5).

    Args:
        beta: weight on regression loss (0.7 in the paper).
        tau: ODX threshold for the binary label (25).
        weight_by_conf: weight per-anchor L1 by predicted confidence (detached).
    """

    def __init__(self, beta: float = 0.7, tau: float = 25.0, weight_by_conf: bool = False, eps: float = 1e-6):
        super().__init__()
        self.beta, self.tau, self.weight_by_conf, self.eps = beta, tau, weight_by_conf, eps

    def forward(self, out, y, anchor_scores):
        r, c = out["r"], out["c"]  # (B, K)
        y = y.float()

        # 1) per-anchor L1 regression loss
        err = (r - y[:, None]).abs()
        if self.weight_by_conf:
            l_reg = (err * c.detach()).sum(-1).mean()
        else:
            l_reg = err.mean()

        # 2) confidence matching: target = softmax(-log((a_k - y)^2))
        target = torch.softmax(-torch.log((anchor_scores[None] - y[:, None]) ** 2 + self.eps), dim=-1)
        # D_KL(c || target), as written in the paper
        log_c = torch.log(c.clamp_min(self.eps))
        l_conf = (c * (log_c - torch.log(target.clamp_min(self.eps)))).sum(-1).mean()

        # 3) BCE on the confidence-weighted logit
        y_bin = (y > self.tau).float()
        l_bce = F.binary_cross_entropy_with_logits(out["logit"], y_bin)

        total = self.beta * l_reg + l_conf + l_bce
        return total, {"reg": l_reg.detach(), "conf": l_conf.detach(), "bce": l_bce.detach()}
