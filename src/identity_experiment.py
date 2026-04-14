import copy
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.optim import AdamW
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import wandb

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# ── config ────────────────────────────────────────────────────────────────────


@dataclass
class ExperimentCfg:
    project: str = "identity-experiment"
    # training
    num_batches: int = 1000
    batch_size: int = 4
    max_seq_len: int = 128
    lr: float = 5e-5
    sample_every: int = 50
    # architecture
    split_layer: int = 6  # GPT2 blocks from this index onwards are frozen (0–11)
    new_hidden_dim: int = 768  # hidden dim for new left-side layers
    new_n_heads: int = 8  # attention heads in new blocks; must divide new_hidden_dim
    num_new_blocks: int = (
        0  # number of new trainable transformer blocks (0 = embedding only)
    )
    # loss
    # False → cross-entropy with actual tokens (one-hot KL)
    # True  → KL divergence against original GPT2's output distribution (soft targets)
    use_soft_kl: bool = False


# ── model ─────────────────────────────────────────────────────────────────────


class IdentityModel(nn.Module):
    """
    Replaces the first `split_layer` layers of GPT2 with new trainable layers,
    keeps the remaining layers frozen, and learns to reproduce GPT2's output.

    Architecture:
        [new wte + new wpe]  ← trainable
        [num_new_blocks × GPT2Block(new_hidden_dim)]  ← trainable
        [Linear(new_hidden_dim → gpt2_hidden)]  ← trainable (Identity if dims match)
        [GPT2 blocks split_layer..11]  ← frozen
        [GPT2 ln_f]  ← frozen
        [GPT2 lm_head]  ← frozen
    """

    def __init__(self, gpt2_model: GPT2LMHeadModel, cfg: ExperimentCfg):
        super().__init__()
        gpt2_cfg = gpt2_model.config
        vocab_size = gpt2_cfg.vocab_size
        max_pos = gpt2_cfg.n_positions
        gpt2_hidden = gpt2_cfg.n_embd  # 768 for gpt2-small

        # ── trainable left side ──────────────────────────────────────────────
        self.new_wte = nn.Embedding(vocab_size, cfg.new_hidden_dim)
        self.new_wpe = nn.Embedding(max_pos, cfg.new_hidden_dim)

        if cfg.num_new_blocks > 0:
            block_cfg = copy.deepcopy(gpt2_cfg)
            block_cfg.n_embd = cfg.new_hidden_dim
            block_cfg.n_head = cfg.new_n_heads
            block_cfg.n_inner = 4 * cfg.new_hidden_dim
            self.new_blocks = nn.ModuleList(
                [GPT2Block(block_cfg, layer_idx=i) for i in range(cfg.num_new_blocks)]
            )
        else:
            self.new_blocks = nn.ModuleList()

        if cfg.new_hidden_dim != gpt2_hidden:
            self.proj: nn.Module = nn.Linear(
                cfg.new_hidden_dim, gpt2_hidden, bias=False
            )
        else:
            self.proj = nn.Identity()

        # ── frozen right side ────────────────────────────────────────────────
        # The blocks are moved into this module (shared objects with gpt2_model).
        # call gpt2_model.to(device) before model.to(device) to move everything.
        self.right_blocks = nn.ModuleList(gpt2_model.transformer.h[cfg.split_layer :])
        self.ln_f = gpt2_model.transformer.ln_f
        self.lm_head = gpt2_model.lm_head

        for param in self.right_blocks.parameters():
            param.requires_grad = False
        for param in self.ln_f.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def _build_causal_4d_mask(
        attention_mask: torch.Tensor, dtype: torch.dtype, seq_len: int
    ) -> torch.Tensor:
        """
        Combined causal + padding mask in additive format.
        Shape: (batch, 1, seq_len, seq_len)
        Position i may attend to position j only if j <= i (causal) AND attention_mask[j] == 1.
        Valid: 0.0 — Masked: finfo(dtype).min
        """
        batch_size, device = attention_mask.shape[0], attention_mask.device
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        padding = attention_mask[:, None, None, :].bool()
        valid = causal[None, None] & padding
        mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=dtype, device=device)
        return mask.masked_fill(~valid, torch.finfo(dtype).min)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        hidden = self.new_wte(input_ids) + self.new_wpe(position_ids)

        if attention_mask is not None:
            causal_mask = self._build_causal_4d_mask(attention_mask, hidden.dtype, seq_len)
        else:
            causal_mask = None

        for block in self.new_blocks:
            hidden = block(hidden, attention_mask=causal_mask, use_cache=False)[0]

        hidden = self.proj(hidden)

        # Rebuild mask after proj — dtype may differ
        if attention_mask is not None:
            causal_mask = self._build_causal_4d_mask(attention_mask, hidden.dtype, seq_len)

        for block in self.right_blocks:
            hidden = block(hidden, attention_mask=causal_mask, use_cache=False)[0]

        hidden = self.ln_f(hidden)
        return self.lm_head(hidden)


# ── loss ──────────────────────────────────────────────────────────────────────


def compute_loss(
    logits: torch.Tensor,
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: ExperimentCfg,
    gpt2_model: GPT2LMHeadModel | None = None,
) -> torch.Tensor:
    if cfg.use_soft_kl:
        assert gpt2_model is not None, "gpt2_model required for soft KL loss"
        with torch.no_grad():
            teacher_logits = gpt2_model(ids, attention_mask=attention_mask).logits
        target = F.softmax(teacher_logits, dim=-1)
        log_q = F.log_softmax(logits, dim=-1)
        # KL(teacher || model), summed over vocab, averaged over valid tokens
        kl = F.kl_div(log_q, target, reduction="none").sum(dim=-1)  # (batch, seq)
        kl = kl * attention_mask.float()
        return kl.sum() / attention_mask.float().sum()
    else:
        # Cross-entropy with the input token at the same position (no shift)
        labels = ids.clone()
        labels[attention_mask == 0] = -100
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100
        )


# ── data generation ───────────────────────────────────────────────────────────


def synthetic_batch_generator(gpt2_model, batch_size=1, seq_len=20, num_batches=1):
    for _ in range(num_batches):
        with torch.inference_mode():
            ids = gpt2_model.generate(
                input_ids=None,
                attention_mask=None,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                max_new_tokens=seq_len - 1,
                num_return_sequences=batch_size,
            )
        # Attend from position 0 up to and including the first terminating EOS.
        # GPT2 uses EOS as BOS, so valid sequences have eos_cumsum <= 2.
        eos_cumsum = (ids == tokenizer.eos_token_id).cumsum(dim=1)
        attention_mask = eos_cumsum <= 2
        yield ids.clone(), attention_mask


# ── fixed test sequences ──────────────────────────────────────────────────────

NATURAL_SEQ = "Hi, I'm Bert"
_rng = torch.Generator()
_rng.manual_seed(42)
RANDOM_SEQ_IDS = torch.randint(0, tokenizer.vocab_size, (1, 20), generator=_rng)


# ── training ──────────────────────────────────────────────────────────────────


def train(cfg: ExperimentCfg):
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    gpt2_model.to(device)

    model = IdentityModel(gpt2_model, cfg)
    model.to(device)

    n_trainable = sum(p.numel() for p in model.trainable_params())
    n_frozen = sum(p.numel() for p in model.parameters()) - n_trainable
    print(f"Trainable: {n_trainable:,}  |  Frozen: {n_frozen:,}")

    assert (
        cfg.new_hidden_dim % cfg.new_n_heads == 0
    ), f"new_hidden_dim ({cfg.new_hidden_dim}) must be divisible by new_n_heads ({cfg.new_n_heads})"

    optimizer = AdamW(model.trainable_params(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(device.type)

    test_seq_rows = []

    def log_test_sequences(step):
        model.eval()
        with torch.inference_mode():
            nat_ids = tokenizer.encode(NATURAL_SEQ, return_tensors="pt").to(device)
            nat_mask = torch.ones_like(nat_ids)
            nat_logits = model(nat_ids, nat_mask)
            nat_out = tokenizer.decode(nat_logits[0].argmax(dim=-1))

            rand_ids = RANDOM_SEQ_IDS.to(device)
            rand_mask = torch.ones_like(rand_ids)
            rand_logits = model(rand_ids, rand_mask)
            rand_out = tokenizer.decode(rand_logits[0].argmax(dim=-1))
            rand_in_text = tokenizer.decode(rand_ids[0])

        test_seq_rows.append([step, "natural", NATURAL_SEQ, nat_out])
        test_seq_rows.append([step, "random", rand_in_text, rand_out])
        table = wandb.Table(
            columns=["step", "name", "input_text", "model_output"],
            data=test_seq_rows,
        )
        wandb.log({"test_sequences": table}, step=step)
        model.train()

    wandb_key = os.getenv("WANDB_API_KEY")
    assert wandb_key is not None, "WANDB_API_KEY must be set in .env"
    wandb.login(key=wandb_key)

    with wandb.init(project=cfg.project, config=asdict(cfg)) as run:
        run.watch(model, log_freq=cfg.sample_every)
        wandb.log({"train/trainable_params": n_trainable}, step=0)

        pbar = tqdm(
            synthetic_batch_generator(
                gpt2_model,
                batch_size=cfg.batch_size,
                seq_len=cfg.max_seq_len,
                num_batches=cfg.num_batches,
            ),
            total=cfg.num_batches,
        )

        step = 0
        for ids, attention_mask in pbar:
            ids = ids.to(device)
            attention_mask = attention_mask.to(device)
            model.train()

            gpt2_for_loss = gpt2_model if cfg.use_soft_kl else None
            with torch.amp.autocast(device.type):
                logits = model(ids, attention_mask)
                loss = compute_loss(logits, ids, attention_mask, cfg, gpt2_for_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            wandb.log({"train/loss": loss.item()}, step=step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            step += 1

            if step % cfg.sample_every == 0:
                log_test_sequences(step)


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(ExperimentCfg())
