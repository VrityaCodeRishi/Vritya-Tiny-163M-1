import os
import time
import random
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from huggingface_hub import hf_hub_download

HF_REPO = "curious-techie/Vritya-Tiny-163M"

cfg = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "context_length": 1024,
    "dropout": 0.1,
    "n_layers": 12,
    "n_heads": 12,
    "qkv_bias": False,
}

tokenizer = tiktoken.get_encoding("gpt2")
PAD_TOKEN = tokenizer.eot_token
MAX_LENGTH = 512


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            nn.GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False, n_heads=8):
        super().__init__()
        assert d_out % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.dropout_p = dropout

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_out = nn.Linear(d_out, d_out, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiheadAttention(
            cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"],
            cfg["dropout"], cfg["qkv_bias"], cfg["n_heads"]
        )
        self.ffn = FeedForward(cfg)
        self.ln1 = LayerNorm(cfg["emb_dim"])
        self.ln2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.linear_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.drop_embed = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        _, seq_len = x.shape
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(
            torch.arange(seq_len, device=x.device)
        )
        x = token_embeddings + position_embeddings
        x = self.drop_embed(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.linear_head(x)
        return logits


PROMPT_TEMPLATE = (
    "Write a short story with the following details.\n\n"
    "### Summary:\n{summary}\n\n"
    "### Words to include:\n{words}\n\n"
    "### Features:\n{features}\n\n"
    "### Story:\n"
)


def format_prompt(sample):
    return PROMPT_TEMPLATE.format(
        summary=sample.get("summary", ""),
        words=sample.get("words", ""),
        features=sample.get("features", ""),
    )


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.input_ids = []
        self.labels = []

        for sample in data:
            prompt = format_prompt(sample)
            full_text = prompt + sample["story"] + "<|endoftext|>"

            prompt_ids = tokenizer.encode(
                prompt, allowed_special={"<|endoftext|>"}
            )
            full_ids = tokenizer.encode(
                full_text, allowed_special={"<|endoftext|>"}
            )

            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            self.input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            self.labels.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def dynamic_pad_collate(batch):
    input_ids, labels = zip(*batch)
    max_len = max(seq.size(0) for seq in input_ids)

    padded_input_ids = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (ids, lbl) in enumerate(zip(input_ids, labels)):
        padded_input_ids[i, :ids.size(0)] = ids
        padded_labels[i, :lbl.size(0)] = lbl

    return padded_input_ids, padded_labels


def parse_tinystories_instruct(rows):
    """Parse raw TinyStoriesInstruct rows into structured samples."""
    samples = []
    current = {}
    story_lines = []
    in_story = False

    for row in rows:
        text = row["text"].strip()

        if text == "<|endoftext|>":
            if in_story and story_lines:
                current["story"] = "\n".join(story_lines)
                story_lines = []
                in_story = False
            if current.get("story"):
                samples.append(current)
            current = {}
            continue

        if text == "":
            if in_story:
                story_lines.append("")
            continue

        if text.startswith("Summary:"):
            current["summary"] = text[len("Summary:"):].strip()
        elif text.startswith("Words:"):
            current["words"] = text[len("Words:"):].strip()
        elif text.startswith("Features:"):
            current["features"] = text[len("Features:"):].strip()
        elif text.startswith("Story:"):
            in_story = True
            remainder = text[len("Story:"):].strip()
            if remainder:
                story_lines.append(remainder)
        elif in_story:
            story_lines.append(text)

    if in_story and story_lines:
        current["story"] = "\n".join(story_lines)
    if current.get("story"):
        samples.append(current)

    return samples


def load_tinystories_instruct(max_samples=50000, train_frac=0.85, val_frac=0.05, seed=42):
    print(f"Loading TinyStoriesInstruct (streaming up to {max_samples:,} samples)...")

    ds = load_dataset("roneneldan/TinyStoriesInstruct", split="train", streaming=True)

    raw_rows = []
    sample_count = 0
    for row in ds:
        raw_rows.append(row)
        if row["text"].strip() == "<|endoftext|>":
            sample_count += 1
            if sample_count >= max_samples:
                break

    all_data = parse_tinystories_instruct(raw_rows)
    print(f"  Parsed {len(all_data):,} samples")

    random.seed(seed)
    random.shuffle(all_data)

    n = len(all_data)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]

    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")
    return train_data, val_data, test_data


def load_pretrained_for_instruction(cfg, pretrained_path):
    model = GPTModel(cfg)
    model.load_state_dict(
        torch.load(pretrained_path, map_location="cpu", weights_only=True)
    )
    print(f"Loaded pretrained weights from {pretrained_path}")

    for param in model.parameters():
        param.requires_grad = False

    for block in model.transformer_blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.linear_head.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    return model


def calc_loss_batch(input_ids, labels, model, device):
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    return loss


def evaluate(model, val_loader, device):
    model.eval()
    total_loss, total_steps = 0.0, 0
    with torch.no_grad():
        for input_ids, labels in val_loader:
            loss = calc_loss_batch(input_ids, labels, model, device)
            total_loss += loss.item()
            total_steps += 1
    return total_loss / max(total_steps, 1)


def generate_sample(model, tokenizer, device, max_tokens=200):
    model.eval()
    sample = {
        "summary": "A little bear finds a magical flower in the forest.",
        "words": "bear, flower, forest",
        "features": "Dialogue",
    }
    prompt = format_prompt(sample)
    input_ids = torch.tensor(
        tokenizer.encode(prompt, allowed_special={"<|endoftext|>"}),
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            if input_ids.size(1) > cfg["context_length"]:
                input_ids = input_ids[:, -cfg["context_length"]:]
            logits = model(input_ids)
            logits = logits[:, -1, :] / 0.7
            top_k_logits, _ = torch.topk(logits, 40)
            logits[logits < top_k_logits[:, [-1]]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.item() == PAD_TOKEN:
                break
            input_ids = torch.cat((input_ids, next_id), dim=1)

    full_text = tokenizer.decode(input_ids.squeeze(0).tolist())
    response = full_text.split("### Story:\n")[-1].strip()
    return response


def train(model, train_loader, val_loader, optimizer, device, num_epochs,
          tokenizer, eval_every=500):
    train_losses, val_losses = [], []
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        t0 = time.time()
        epoch_loss, epoch_steps = 0.0, 0

        for input_ids, labels in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_ids, labels, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % 100 == 0:
                print(f"  Step {global_step:,} | Loss {loss.item():.4f}", flush=True)

            if global_step % eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                train_losses.append(epoch_loss / epoch_steps)
                val_losses.append(val_loss)
                print(f"  Step {global_step:,} | "
                      f"Train Loss: {epoch_loss / epoch_steps:.4f} | "
                      f"Val Loss: {val_loss:.4f}", flush=True)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "instruction_finetuned.pth")

                model.train()

        epoch_time = time.time() - t0
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(epoch_loss / epoch_steps)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} done in {epoch_time/60:.1f} min | "
              f"Avg Loss: {epoch_loss/epoch_steps:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "instruction_finetuned.pth")

        print("--- Sample ---")
        response = generate_sample(model, tokenizer, device)
        print(f"  {response[:400]}")
        print("--------------")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print("Best model saved to instruction_finetuned.pth")
    return train_losses, val_losses


def plot_losses(train_losses, val_losses):
    steps = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_losses, label="Train Loss")
    plt.plot(steps, val_losses, label="Val Loss")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Loss")
    plt.title("Instruction Finetuning Loss (TinyStoriesInstruct)")
    plt.legend()
    plt.savefig("instruction_finetuning_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot saved to instruction_finetuning_loss.png")


if __name__ == "__main__":
    train_data, val_data, test_data = load_tinystories_instruct(max_samples=50000)

    print("Tokenizing train set...")
    train_dataset = InstructionDataset(train_data, tokenizer, MAX_LENGTH)
    print("Tokenizing val set...")
    val_dataset = InstructionDataset(val_data, tokenizer, MAX_LENGTH)
    print("Tokenizing test set...")
    test_dataset = InstructionDataset(test_data, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, drop_last=True,
        collate_fn=dynamic_pad_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, drop_last=False,
        collate_fn=dynamic_pad_collate,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, drop_last=False,
        collate_fn=dynamic_pad_collate,
    )
    print(f"Train batches: {len(train_loader):,} | Val: {len(val_loader):,} | Test: {len(test_loader):,}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    pretrained_path = "best_model.pth"
    if not os.path.exists(pretrained_path):
        pretrained_path = hf_hub_download(HF_REPO, pretrained_path)
    model = load_pretrained_for_instruction(cfg, pretrained_path)
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5, weight_decay=0.1,
    )

    train_losses, val_losses = train(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=3, tokenizer=tokenizer, eval_every=500,
    )

    test_loss = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}")

    plot_losses(train_losses, val_losses)
