import pytest
import torch
from torch.optim import AdamW
from transformers import GPT2LMHeadModel

from src.identity_experiment import (
    ExperimentCfg,
    IdentityModel,
    compute_loss,
    synthetic_batch_generator,
    tokenizer,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gpt2():
    m = GPT2LMHeadModel.from_pretrained("gpt2")
    m.eval()
    return m


@pytest.fixture
def cfg():
    return ExperimentCfg(split_layer=6, new_hidden_dim=64, new_n_heads=8, num_new_blocks=1)


@pytest.fixture
def model(gpt2, cfg):
    return IdentityModel(gpt2, cfg)


# ── freeze / grad isolation ───────────────────────────────────────────────────


class TestFreezeAndGrads:
    def test_right_side_frozen(self, model):
        for param in model.right_blocks.parameters():
            assert not param.requires_grad
        for param in model.ln_f.parameters():
            assert not param.requires_grad
        for param in model.lm_head.parameters():
            assert not param.requires_grad

    def test_left_side_trainable(self, model):
        for p in model.new_wte.parameters():
            assert p.requires_grad
        for p in model.new_wpe.parameters():
            assert p.requires_grad
        for p in model.new_blocks.parameters():
            assert p.requires_grad
        if not isinstance(model.proj, torch.nn.Identity):
            for p in model.proj.parameters():
                assert p.requires_grad

    def test_backward_grads_only_on_trainable(self, model, cfg):
        model.train()
        ids = torch.randint(0, 50257, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)

        loss = compute_loss(model(ids, mask), ids, mask, cfg)
        loss.backward()

        for param in model.right_blocks.parameters():
            assert param.grad is None
        for param in model.ln_f.parameters():
            assert param.grad is None
        for param in model.lm_head.parameters():
            assert param.grad is None
        for param in model.trainable_params():
            assert param.grad is not None

    def test_frozen_weights_unchanged_after_step(self, model, cfg):
        model.train()
        snapshot = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if not param.requires_grad
        }

        optimizer = AdamW(model.trainable_params(), lr=1e-3)
        ids = torch.randint(0, 50257, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)

        compute_loss(model(ids, mask), ids, mask, cfg).backward()
        optimizer.step()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert torch.equal(param.data, snapshot[name]), f"{name} changed after step"


# ── loss ──────────────────────────────────────────────────────────────────────


class TestComputeLoss:
    def test_no_shift(self):
        """Loss should be near zero when logits are confident about the token at the same position."""
        batch, seq, vocab = 2, 8, 50257
        ids = torch.randint(0, vocab, (batch, seq))
        mask = torch.ones(batch, seq, dtype=torch.long)

        logits = torch.full((batch, seq, vocab), -100.0)
        for b in range(batch):
            for i in range(seq):
                logits[b, i, ids[b, i]] = 100.0  # confident about token at position i (not i+1)

        assert compute_loss(logits, ids, mask, ExperimentCfg()).item() < 0.01

    def test_padding_tokens_not_counted(self):
        """Swapping tokens at masked positions must not change the loss."""
        batch, seq, vocab = 2, 8, 50257
        ids = torch.randint(0, vocab, (batch, seq))
        mask = torch.zeros(batch, seq, dtype=torch.long)
        mask[:, :4] = 1
        logits = torch.randn(batch, seq, vocab)
        cfg = ExperimentCfg()

        ids_mod = ids.clone()
        ids_mod[:, 4:] = (ids_mod[:, 4:] + 1) % vocab

        assert torch.equal(
            compute_loss(logits, ids, mask, cfg),
            compute_loss(logits, ids_mod, mask, cfg),
        )

    def test_fully_masked_sequence_ignored(self):
        """A completely masked sequence must not affect the loss."""
        batch, seq, vocab = 2, 8, 50257
        ids = torch.randint(0, vocab, (batch, seq))
        logits = torch.randn(batch, seq, vocab)
        mask = torch.ones(batch, seq, dtype=torch.long)
        mask[1, :] = 0
        cfg = ExperimentCfg()

        ids_mod = ids.clone()
        ids_mod[1, :] = torch.randint(0, vocab, (seq,))

        assert torch.equal(
            compute_loss(logits, ids, mask, cfg),
            compute_loss(logits, ids_mod, mask, cfg),
        )

    def test_soft_kl_non_negative(self, gpt2):
        batch, seq, vocab = 1, 8, 50257
        ids = torch.randint(0, vocab, (batch, seq))
        mask = torch.ones(batch, seq, dtype=torch.long)
        logits = torch.randn(batch, seq, vocab)

        loss = compute_loss(logits, ids, mask, ExperimentCfg(use_soft_kl=True), gpt2)
        assert loss.item() >= 0.0

    def test_soft_kl_requires_model(self):
        logits = torch.randn(1, 8, 50257)
        ids = torch.randint(0, 50257, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.long)

        with pytest.raises(AssertionError):
            compute_loss(logits, ids, mask, ExperimentCfg(use_soft_kl=True), gpt2_model=None)


# ── _build_causal_4d_mask ─────────────────────────────────────────────────────


class TestBuildCausal4dMask:
    def test_output_shape(self):
        mask = torch.ones(3, 10, dtype=torch.long)
        out = IdentityModel._build_causal_4d_mask(mask, torch.float32, seq_len=10)
        assert out.shape == (3, 1, 10, 10)

    def test_causal_valid_positions_are_zero(self):
        # Diagonal and below (past/present) with no padding → 0.0
        mask = torch.ones(1, 4, dtype=torch.long)
        out = IdentityModel._build_causal_4d_mask(mask, torch.float32, seq_len=4)
        assert out[0, 0, 0, 0] == 0.0   # query=0, key=0  (self)
        assert out[0, 0, 2, 0] == 0.0   # query=2, key=0  (past)
        assert out[0, 0, 2, 2] == 0.0   # query=2, key=2  (self)

    def test_future_positions_are_large_negative(self):
        # Upper triangle (future tokens) → very negative
        mask = torch.ones(1, 4, dtype=torch.long)
        out = IdentityModel._build_causal_4d_mask(mask, torch.float32, seq_len=4)
        assert out[0, 0, 0, 1] <= -1e4  # query=0, key=1  (future)
        assert out[0, 0, 1, 3] <= -1e4  # query=1, key=3  (future)

    def test_padding_positions_are_large_negative(self):
        # Padded key positions → very negative regardless of causal order
        mask = torch.tensor([[1, 1, 0, 0]])  # last 2 are padding
        out = IdentityModel._build_causal_4d_mask(mask, torch.float32, seq_len=4)
        assert out[0, 0, 3, 2] <= -1e4  # key=2 is padding, even though j<=i


# ── synthetic_batch_generator ─────────────────────────────────────────────────


class TestSyntheticBatchGenerator:
    def test_yields_correct_count(self, gpt2):
        batches = list(synthetic_batch_generator(gpt2, batch_size=2, seq_len=8, num_batches=3))
        assert len(batches) == 3

    def test_output_shapes(self, gpt2):
        ids, mask = next(iter(synthetic_batch_generator(gpt2, batch_size=3, seq_len=10)))
        assert ids.shape == (3, 10)
        assert mask.shape == (3, 10)

    def test_token_ids_in_range(self, gpt2):
        ids, _ = next(iter(synthetic_batch_generator(gpt2, batch_size=4, seq_len=12)))
        assert (ids >= 0).all() and (ids < tokenizer.vocab_size).all()

    def test_mask_starts_true(self, gpt2):
        # GPT2 uses EOS as BOS, so every sequence begins with a valid token
        _, mask = next(iter(synthetic_batch_generator(gpt2, batch_size=4, seq_len=12)))
        assert mask[:, 0].all()


# ── oracle integration test ───────────────────────────────────────────────────


class TestOracle:
    def test_split0_with_gpt2_embeddings_matches_gpt2(self, gpt2):
        """
        With split_layer=0 and no new blocks, the model is:
            [new_wte + new_wpe] → [all 12 frozen GPT2 blocks] → ln_f → lm_head
        If we copy GPT2's wte/wpe into new_wte/new_wpe, the output must be
        identical to the original GPT2 — testing the full routing end-to-end.
        """
        cfg = ExperimentCfg(split_layer=0, new_hidden_dim=768, new_n_heads=12, num_new_blocks=0)
        model = IdentityModel(gpt2, cfg)
        model.eval()

        with torch.no_grad():
            model.new_wte.weight.copy_(gpt2.transformer.wte.weight)
            model.new_wpe.weight.copy_(gpt2.transformer.wpe.weight)

        ids = torch.randint(0, 50257, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)

        with torch.no_grad():
            assert torch.allclose(
                model(ids, mask),
                gpt2(ids, attention_mask=mask).logits,
                atol=1e-5,
            )
