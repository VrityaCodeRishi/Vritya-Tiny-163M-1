import os
import time
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

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
        position_embeddings = self.position_embedding(torch.arange(seq_len, device=x.device))
        x = token_embeddings + position_embeddings
        x = self.drop_embed(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.linear_head(x)
        return logits


class GPTDataset(Dataset):
    def __init__(self, token_ids, context_length, stride):
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride
        self.n_samples = max(0, (len(token_ids) - context_length) // stride)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.context_length
        return self.token_ids[start:end], self.token_ids[start + 1:end + 1]


def generate_text_simple(model, idx, max_new_tokens, context_size,
                         temperature=1.0, top_k=50):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < top_k_logits[:, [-1]]] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def text_to_token_ids(text, tokenizer):
    return torch.tensor(
        tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    ).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    total_samples = 0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        loss = calc_loss_batch(inputs, targets, model, device)
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
    return total_loss / total_samples


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    with torch.no_grad():
        token_ids = generate_text_simple(
            model, text_to_token_ids(start_context, tokenizer).to(device),
            50, cfg["context_length"]
        )
        print("Generated text:\n", token_ids_to_text(token_ids, tokenizer))


def save_checkpoint(model, optimizer, epoch, global_step, tokens_seen,
                    best_val_loss, train_losses, val_losses, track_tokens_seen,
                    path="checkpoint.pth"):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "track_tokens_seen": track_tokens_seen,
    }, path)
    print(f"  Checkpoint saved at step {global_step}")


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (checkpoint["epoch"], checkpoint["global_step"],
            checkpoint["tokens_seen"], checkpoint["best_val_loss"],
            checkpoint["train_losses"], checkpoint["val_losses"],
            checkpoint["track_tokens_seen"])


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context, tokenizer,
                resume_from=None, checkpoint_every=1000):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float('inf')
    start_epoch = 0
    patience, patience_counter = 5, 0

    if resume_from and os.path.exists(resume_from):
        start_epoch, global_step, tokens_seen, best_val_loss, \
            train_losses, val_losses, track_tokens_seen = \
            load_checkpoint(model, optimizer, resume_from)
        print(f"Resumed from epoch {start_epoch + 1}, step {global_step}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0
        epoch_steps = 0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            tokens_seen += input_batch.size(0) * input_batch.size(1)
            epoch_loss += loss.item()
            epoch_steps += 1

            if global_step % 100 == 0 and global_step > 0:
                elapsed = time.time() - t0
                tps = (epoch_steps * input_batch.size(0) * input_batch.size(1)) / max(elapsed, 1e-6)
                print(f"  Step {global_step:,} | Loss {loss.item():.4f} | "
                      f"{tps:,.0f} tok/s", flush=True)

            if global_step % checkpoint_every == 0 and global_step > 0:
                save_checkpoint(model, optimizer, epoch, global_step,
                                tokens_seen, best_val_loss,
                                train_losses, val_losses, track_tokens_seen)

            if global_step % eval_freq == 0:
                elapsed = time.time() - t0
                tok_per_sec = tokens_seen / max(elapsed, 1e-6) if epoch == start_epoch else 0
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                track_tokens_seen.append(tokens_seen)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} | Step {global_step:,} | "
                      f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                      f"Tokens {tokens_seen:,}")
                generate_and_print_sample(model, tokenizer, device, start_context)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best_model.pth")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1} "
                              f"— best val loss: {best_val_loss:.4f}")
                        return train_losses, val_losses, track_tokens_seen

                model.train()

        epoch_time = time.time() - t0
        print(f"--- Epoch {epoch + 1} done in {epoch_time / 60:.1f} min | "
              f"Avg loss: {epoch_loss / epoch_steps:.4f} ---")

        save_checkpoint(model, optimizer, epoch + 1, global_step,
                        tokens_seen, best_val_loss,
                        train_losses, val_losses, track_tokens_seen)

    return train_losses, val_losses, track_tokens_seen


def load_tinystories(tokenizer, cache_dir="data"):
    """Load and tokenize TinyStories. Caches tokenized tensors to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "tinystories_train.pt")
    val_cache = os.path.join(cache_dir, "tinystories_val.pt")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print("Loading cached tokenized data...")
        train_tokens = torch.load(train_cache, weights_only=True)
        val_tokens = torch.load(val_cache, weights_only=True)
        print(f"  Train: {len(train_tokens):,} tokens | Val: {len(val_tokens):,} tokens")
        return train_tokens, val_tokens

    print("Downloading TinyStories and tokenizing (first time only)...")
    ds = load_dataset("roneneldan/TinyStories")
    eot = tokenizer.eot_token

    train_token_list = []
    for i, story in enumerate(ds["train"]):
        train_token_list.extend(tokenizer.encode(story["text"]))
        train_token_list.append(eot)
        if (i + 1) % 200_000 == 0:
            print(f"  Tokenized {i + 1:,} / {len(ds['train']):,} train stories")

    val_token_list = []
    for story in ds["validation"]:
        val_token_list.extend(tokenizer.encode(story["text"]))
        val_token_list.append(eot)

    train_tokens = torch.tensor(train_token_list, dtype=torch.long)
    val_tokens = torch.tensor(val_token_list, dtype=torch.long)
    del train_token_list, val_token_list

    torch.save(train_tokens, train_cache)
    torch.save(val_tokens, val_cache)
    print(f"  Train: {len(train_tokens):,} tokens | Val: {len(val_tokens):,} tokens")
    print(f"  Cached to {cache_dir}/")
    return train_tokens, val_tokens


if __name__ == "__main__":
    train_tokens, val_tokens = load_tinystories(tokenizer)

    train_loader = DataLoader(
        GPTDataset(train_tokens, cfg["context_length"], cfg["context_length"]),
        batch_size=64, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        GPTDataset(val_tokens, cfg["context_length"], cfg["context_length"]),
        batch_size=64, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True,
    )
    print(f"Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")


    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GPTModel(cfg)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params * 4 / 1024**3:.2f} GB fp32)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=0.1)

    train_losses, val_losses, track_tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=2,
        eval_freq=500,
        eval_iter=20,
        start_context="Once upon a time",
        tokenizer=tokenizer,
        resume_from="checkpoint.pth",
        checkpoint_every=1000,
    )

    if train_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(track_tokens_seen, train_losses, label="Train Loss")
        plt.plot(track_tokens_seen, val_losses, label="Val Loss")
        plt.xlabel("Tokens Seen")
        plt.ylabel("Loss")
        plt.title("TinyStories Pretraining Loss")
        plt.legend()
        plt.savefig("loss_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Loss plot saved to loss_plot.png")
    else:
        print("No loss data to plot.")
