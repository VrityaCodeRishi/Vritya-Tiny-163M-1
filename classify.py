import torch
import torch.nn as nn
import tiktoken

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
MAX_LENGTH = 257
NUM_CLASSES = 2
LABEL_MAP = {0: "Ham", 1: "Spam"}


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


def load_model(path="finetuned_for_classification.pth"):
    model = GPTModel(cfg)
    model.linear_head = nn.Linear(cfg["emb_dim"], NUM_CLASSES, bias=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def classify(model, text, device):
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    if len(ids) > MAX_LENGTH:
        ids = ids[:MAX_LENGTH]
    n_real = len(ids)
    ids = ids + [PAD_TOKEN] * (MAX_LENGTH - n_real)

    token_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(token_tensor)
        last_logits = logits[0, n_real - 1]
        pred = last_logits.argmax().item()
        probs = torch.softmax(last_logits, dim=-1)

    return LABEL_MAP[pred], probs[pred].item()


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model()
    model.to(device)
    print(f"Model loaded on {device}\n")

    demos = [
        "Hey, are you free for lunch tomorrow?",
        "CONGRATULATIONS! You've won a $1000 gift card! Click here to claim NOW!",
        "Can you pick up milk on your way home?",
        "FREE entry to win a brand new iPhone! Text WIN to 80888",
        "Meeting moved to 3pm, see you there",
    ]

    print("--- Demo ---")
    for msg in demos:
        label, conf = classify(model, msg, device)
        print(f"  [{label}] ({conf:.1%}) \"{msg}\"")

    print("\n--- Type a message (or 'quit' to exit) ---\n")
    while True:
        msg = input(">> ").strip()
        if msg.lower() in ("quit", "exit", "q"):
            break
        if not msg:
            continue
        label, conf = classify(model, msg, device)
        print(f"  [{label}] ({conf:.1%})")
