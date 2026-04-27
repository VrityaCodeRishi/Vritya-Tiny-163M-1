import os
import urllib.request
import zipfile
import random
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

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



def download_sms_spam(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "smsspamcollection.zip")
    extracted_path = os.path.join(data_dir, "SMSSpamCollection")

    if not os.path.exists(extracted_path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        print(f"Downloading SMS Spam Collection from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        print("Download complete.")

    ham, spam = [], []
    with open(extracted_path, "r", encoding="utf-8") as f:
        for line in f:
            label, msg = line.strip().split("\t", maxsplit=1)
            if label == "ham":
                ham.append(msg)
            else:
                spam.append(msg)

    print(f"Loaded {len(ham)} ham, {len(spam)} spam messages")
    return ham, spam




def balance_and_split(ham, spam, train_frac=0.7, val_frac=0.15, seed=42):
    """Downsample ham to match spam count, then split into train/val/test."""
    random.seed(seed)
    random.shuffle(ham)
    ham_balanced = ham[:len(spam)]

    texts = ham_balanced + spam
    labels = [0] * len(ham_balanced) + [1] * len(spam)

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts, labels = list(texts), list(labels)

    n = len(texts)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    splits = {
        "train": (texts[:train_end], labels[:train_end]),
        "val":   (texts[train_end:val_end], labels[train_end:val_end]),
        "test":  (texts[val_end:], labels[val_end:]),
    }
    for name, (t, l) in splits.items():
        n_spam = sum(l)
        print(f"  {name:5s}: {len(t):4d} samples  (ham {len(t)-n_spam}, spam {n_spam})")
    return splits



class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.token_ids = []
        self.attention_masks = []

        for text in texts:
            ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

            if len(ids) > max_length:
                ids = ids[:max_length]

            attn_mask = [1] * len(ids) + [0] * (max_length - len(ids))
            ids = ids + [PAD_TOKEN] * (max_length - len(ids))

            self.token_ids.append(torch.tensor(ids, dtype=torch.long))
            self.attention_masks.append(torch.tensor(attn_mask, dtype=torch.long))

        self.token_ids = torch.stack(self.token_ids)
        self.attention_masks = torch.stack(self.attention_masks)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.attention_masks[idx], self.labels[idx]




def get_max_token_length(texts, tokenizer):
    lengths = [len(tokenizer.encode(t)) for t in texts]
    return max(lengths)



def load_pretrained_classifier(cfg, pretrained_path, num_classes=2):
    model = GPTModel(cfg)
    model.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True))
    print(f"Loaded pretrained weights from {pretrained_path}")


    for param in model.parameters():
        param.requires_grad = False

    for param in model.transformer_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    model.linear_head = nn.Linear(cfg["emb_dim"], num_classes, bias=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    return model


def calc_loss_and_acc(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for token_ids, attn_masks, labels in loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            attn_masks = attn_masks.to(device)

            logits = model(token_ids)

            last_real_idx = attn_masks.sum(dim=1) - 1  # (B,)
            last_logits = logits[torch.arange(logits.size(0)), last_real_idx]  # (B, num_classes)

            loss = torch.nn.functional.cross_entropy(last_logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = last_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, optimizer, device, num_epochs):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for token_ids, attn_masks, labels in train_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            attn_masks = attn_masks.to(device)

            logits = model(token_ids)

            last_real_idx = attn_masks.sum(dim=1) - 1
            last_logits = logits[torch.arange(logits.size(0)), last_real_idx]

            loss = torch.nn.functional.cross_entropy(last_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            preds = last_logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_loss, val_acc = calc_loss_and_acc(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Accuracy")
    ax2.plot(epochs, val_accs, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("spam_finetuning_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot saved to spam_finetuning_metrics.png")



if __name__ == "__main__":
    ham, spam = download_sms_spam()
    splits = balance_and_split(ham, spam)

    all_texts = splits["train"][0] + splits["val"][0] + splits["test"][0]
    max_length = min(get_max_token_length(all_texts, tokenizer), cfg["context_length"])
    print(f"\nMax token length (capped at context_length): {max_length}")

    train_dataset = SpamDataset(*splits["train"], tokenizer, max_length)
    val_dataset   = SpamDataset(*splits["val"],   tokenizer, max_length)
    test_dataset  = SpamDataset(*splits["test"],  tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, drop_last=False)

    print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_pretrained_classifier(cfg, "/Users/curious_techie/Desktop/best_model.pth")
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=0.1,
    )

    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, val_loader, optimizer, device, num_epochs=5
    )

    test_loss, test_acc = calc_loss_and_acc(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), "finetuned_for_classification.pth")
    print("Model saved to finetuned_for_classification.pth")

    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    label_map = {0: "Ham", 1: "Spam"}
    print("\n--- Spam Classifier ---")
    print("Type a message to classify (or 'quit' to exit):\n")

    test_messages = [
        "Hey, are you free for lunch tomorrow?",
        "CONGRATULATIONS! You've won a $1000 gift card! Click here to claim NOW!",
        "Can you pick up milk on your way home?",
        "FREE entry to win a brand new iPhone! Text WIN to 80888",
        "Meeting moved to 3pm, see you there",
    ]

    model.eval()
    for msg in test_messages:
        ids = tokenizer.encode(msg, allowed_special={"<|endoftext|>"})
        if len(ids) > max_length:
            ids = ids[:max_length]
        attn_mask = [1] * len(ids) + [0] * (max_length - len(ids))
        ids = ids + [PAD_TOKEN] * (max_length - len(ids))

        token_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        mask_tensor = torch.tensor(attn_mask, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(token_tensor)
            last_idx = mask_tensor.sum(dim=1) - 1
            last_logits = logits[0, last_idx[0]]
            pred = last_logits.argmax().item()
            probs = torch.softmax(last_logits, dim=-1)

        print(f"  [{label_map[pred]}] ({probs[pred]:.1%}) \"{msg}\"")

    print()
    while True:
        msg = input(">> ").strip()
        if msg.lower() in ("quit", "exit", "q"):
            break
        ids = tokenizer.encode(msg, allowed_special={"<|endoftext|>"})
        if len(ids) > max_length:
            ids = ids[:max_length]
        attn_mask = [1] * len(ids) + [0] * (max_length - len(ids))
        ids = ids + [PAD_TOKEN] * (max_length - len(ids))

        token_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        mask_tensor = torch.tensor(attn_mask, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(token_tensor)
            last_idx = mask_tensor.sum(dim=1) - 1
            last_logits = logits[0, last_idx[0]]
            pred = last_logits.argmax().item()
            probs = torch.softmax(last_logits, dim=-1)

        print(f"  [{label_map[pred]}] ({probs[pred]:.1%})")
