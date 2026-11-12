"""CPU unit tests for metrics, temperature scaling and the pooling proxy (no downloads)."""
import numpy as np
import torch

from uat_lite import metrics
from uat_lite.calibration import fit_temperature
from uat_lite.inference import pool_with_uncertainty
from uat_lite.protocol import nli3_to_2


def _toy(n=500, c=3, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, c)) * 2
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    labels = np.where(rng.random(n) < 0.7, probs.argmax(1), rng.integers(0, c, n))
    return probs, labels


def test_perfect_calibration_has_zero_ece():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert metrics.ece(probs, np.array([0, 1])) == 0.0


def test_aurc_grid_converges_to_exact():
    probs, labels = _toy()
    conf, correct = probs.max(1), (probs.argmax(1) == labels).astype(float)
    exact = metrics.aurc(conf, correct)
    assert abs(metrics.aurc_grid(conf, correct, len(labels))[0] - exact) < 1e-2


def test_summary_keys():
    probs, labels = _toy()
    out = metrics.summarize(probs, labels)
    for k in ["accuracy", "ece", "brier", "nll", "aurc", "cov@0.9", "acc@0.9", "n@0.9", "acc@cov90%"]:
        assert k in out


def test_temperature_recovers_scale():
    torch.manual_seed(0)
    true_logits = torch.randn(4000, 3) * 2
    labels = torch.distributions.Categorical(logits=true_logits).sample()
    assert abs(fit_temperature(true_logits * 3.0, labels) - 3.0) < 0.3


def test_q_and_v_proxies_coincide():
    h, u = torch.randn(4, 7, 16), torch.rand(4, 7)
    mask = torch.ones(4, 7)
    assert torch.allclose(pool_with_uncertainty(h, u, mask, 0.5, "q"), pool_with_uncertainty(h, u, mask, 0.5, "v"))


def test_nli_mapping_sums_to_one():
    p = torch.softmax(torch.randn(10, 3), -1)
    assert torch.allclose(nli3_to_2(p).sum(1), torch.ones(10))
