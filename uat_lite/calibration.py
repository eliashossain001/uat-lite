"""Post-hoc temperature scaling."""

import torch
import torch.nn as nn


def fit_temperature(logits, labels, lr=0.01, max_iter=100):
    """Fit a scalar temperature T on held-out logits by minimizing NLL.

    Optimizes log T with L-BFGS (strong Wolfe line search), starting at T=1.
    Apply as ``softmax(logits / T)``.
    """
    log_T = nn.Parameter(torch.zeros(1))
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    x = logits.detach().clone()
    y = labels.detach().clone()

    def closure():
        opt.zero_grad()
        loss = criterion(x / log_T.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())
