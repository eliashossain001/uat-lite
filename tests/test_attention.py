"""CPU tests for uncertainty-weighted attention on a tiny randomly initialised BERT."""
import torch
from transformers import BertConfig, BertModel

from uat_lite.models.attention import RunningTokenUncertainty, UncertaintyAttention


def _tiny_bert():
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, attn_implementation="eager")
    return BertModel(cfg).eval()


def _run(model, ids, mask, U=None, **kw):
    with UncertaintyAttention(model, **kw) as uat:
        uat.set_uncertainty(U)
        return model(input_ids=ids, attention_mask=mask).last_hidden_state


def test_unmodulated_patch_matches_hf():
    model = _tiny_bert()
    ids, mask = torch.randint(0, 100, (2, 7)), torch.ones(2, 7, dtype=torch.long)
    mask[1, 5:] = 0
    ref = model(input_ids=ids, attention_mask=mask).last_hidden_state
    assert torch.allclose(_run(model, ids, mask, None), ref, atol=1e-5)
    assert torch.allclose(_run(model, ids, mask, torch.rand(2, 7), lam=0.0), ref, atol=1e-5)


def test_patch_is_removed_after_context():
    model = _tiny_bert()
    ids, mask = torch.randint(0, 100, (1, 5)), torch.ones(1, 5, dtype=torch.long)
    ref = model(input_ids=ids, attention_mask=mask).last_hidden_state
    _run(model, ids, mask, torch.rand(1, 5), lam=2.0)
    assert torch.equal(model(input_ids=ids, attention_mask=mask).last_hidden_state, ref)


def test_variants_differ():
    model = _tiny_bert()
    ids, mask = torch.randint(0, 100, (2, 7)), torch.ones(2, 7, dtype=torch.long)
    U = torch.rand(2, 7)
    outs = {v: _run(model, ids, mask, U, lam=2.0, variant=v) for v in ["q", "k", "qk", "v", "qkv"]}
    assert not torch.allclose(outs["q"], outs["v"])
    assert not torch.allclose(outs["q"], outs["k"])


def test_running_uncertainty_matches_batch_std():
    torch.manual_seed(0)
    z = torch.randn(6, 2, 4, 8)
    r = RunningTokenUncertainty()
    assert torch.equal(r.value(z[0]), torch.zeros(2, 4))
    for m in range(6):
        r.update(z[m])
    assert torch.allclose(r.value(z[0]), z.std(dim=0).mean(dim=-1), atol=1e-6)
