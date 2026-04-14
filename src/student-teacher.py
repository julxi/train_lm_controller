import os
from dataclasses import asdict, dataclass
from dotenv import load_dotenv
from tqdm import tqdm
import wandb

import transformers
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === load teacher model ===
model_name = "gpt2"

teacher_model = GPT2LMHeadModel.from_pretrained("gpt2")
teacher_model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # <- this should get rid of some warnings

# === prepare student ===
student_model = transformers.models.GPT2LMHeadModel(teacher_model.config)

# === wandb login ===
wandb_key = os.getenv("WANDB_API_KEY")
assert wandb_key is not None, "you have to have an api key for wandb in your .env"
wandb.login(key=wandb_key)


# === data generation ===
def synthetic_batch_generator(batch_size=1, seq_len=20, num_batches=1):
    for _ in range(num_batches):
        with torch.inference_mode():
            ids = teacher_model.generate(
                input_ids=None,
                attention_mask=None,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                max_new_tokens=seq_len - 1,
                num_return_sequences=batch_size,
            )

        # attention from 0th pos to first pad_token appearing after the 0th pos
        # the first appearing pad_token is included
        # it's a bit overly clever designed
        # GPT-2 uses EOS as BOS, so generated sequences start with one EOS.
        # cumsum <= 2 attends to everything up to and including the terminating EOS,
        # and correctly handles sequences that hit max_new_tokens without a terminating EOS.
        eos_cumsum = (ids == tokenizer.eos_token_id).cumsum(dim=1)
        attention_mask = eos_cumsum <= 2

        yield ids.clone(), attention_mask


# === traing ===
@dataclass
class TrainCfg:
    project: str = "teacher-student"
    num_batches: int = 10
    batch_size: int = 7  # exprimental max on 4x Tesla V100
    max_seq_len: int = 1024
    lr: float = 5e-5
    sample_every: int = 1


def train(cfg: TrainCfg):

    samples_rows = []  # plain list, not a wandb object

    def log_samples(step):
        student_model.eval()
        with torch.inference_mode():
            greedy = tokenizer.decode(
                student_model.generate(
                    input_ids=None,
                    attention_mask=None,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
            )
            sampled = tokenizer.decode(
                student_model.generate(
                    input_ids=None,
                    attention_mask=None,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                )[0]
            )

        samples_rows.append([step, "greedy", greedy])
        samples_rows.append([step, "sampled", sampled])

        # reconstruct a fresh table from all rows so far
        table = wandb.Table(columns=["step", "type", "text"], data=samples_rows)
        wandb.log({"samples": table}, step=step)
        student_model.train()

    # === training loop ===

    batch_size = cfg.batch_size
    seq_len = cfg.max_seq_len
    num_batches = cfg.num_batches

    # Move models to GPU if available

    teacher_model.to(device)
    student_model.to(device)

    # Optimizer
    optimizer = AdamW(student_model.parameters(), lr=cfg.lr)

    # Scaler
    # scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler(device.type)

    with wandb.init(project=cfg.project, config=asdict(cfg)) as run:
        run.watch(student_model, log_freq=cfg.sample_every)

        pbar = tqdm(
            synthetic_batch_generator(
                batch_size=batch_size, seq_len=seq_len, num_batches=num_batches
            ),
            total=num_batches,
        )

        loss = 0
        step = 0
        for ids, attention_mask in pbar:
            student_model.train()

            labels = ids.clone()
            labels[attention_mask == 0] = -100  # ignore loss for paddedd tokens

            # Forward pass: student model
            with torch.amp.autocast(device.type):
                outputs = student_model(
                    ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            wandb.log({"train/loss": loss.item()}, step=step)
            step += 1

            if step % cfg.sample_every == 0:
                print("logging samples")
                log_samples(step)


# === do train ===
train(
    TrainCfg(
        num_batches=1000,
        batch_size=7,
        max_seq_len=1024,
    )
)
