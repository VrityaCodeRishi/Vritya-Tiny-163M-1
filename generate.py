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


PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_prompt(instruction, input_text=""):
    if input_text.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def generate(model, prompt, device, max_tokens=300, temperature=0.7,
             top_k=40, top_p=0.9, repetition_penalty=1.2):
    model.eval()
    input_ids = torch.tensor(
        tokenizer.encode(prompt, allowed_special={"<|endoftext|>"}),
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            if input_ids.size(1) > cfg["context_length"]:
                input_ids = input_ids[:, -cfg["context_length"]:]
            logits = model(input_ids)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                generated = input_ids[0].tolist()
                for token_id in set(generated):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / temperature

            if top_k > 0:
                top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_logits[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.item() == PAD_TOKEN:
                break
            input_ids = torch.cat((input_ids, next_id), dim=1)

    full_text = tokenizer.decode(input_ids.squeeze(0).tolist())
    response = full_text.split("### Response:\n")[-1].strip()
    return response


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = GPTModel(cfg)
    model.load_state_dict(
        torch.load("/Users/curious_techie/Desktop/instruction_finetuned.pth", map_location="cpu", weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}\n")

    demos = [
        "Give three tips for staying healthy.",
        "What is the capital of France?",
        "Write a short poem about the ocean.",
    ]

    print("--- Demo ---")
    for instr in demos:
        prompt = format_prompt(instr)
        response = generate(model, prompt, device)
        print(f"Instruction: {instr}")
        print(f"Response: {response}\n")

    print("--- Interactive mode (type 'quit' to exit) ---\n")
    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            break
        if not instruction:
            continue

        input_text = input("Input (optional, press Enter to skip): ").strip()
        prompt = format_prompt(instruction, input_text)
        response = generate(model, prompt, device)
        print(f"\nResponse: {response}\n")
