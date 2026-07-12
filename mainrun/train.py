import utils
import math, random, time
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog

@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 64

    # Optimization 26: Tune batch size to 32
    # batch_size: int = 32

    vocab_size: int = 16_000

    # Optimization 24: Tune vocab size to 20k
    # vocab_size: int = 20_000

    # Optimization 25: Tune vocab size to 12k
    # vocab_size: int = 12_000

    """
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    """

    # Optimization 32: Deeper and narrower architecture
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 448

    # Optimization 33: Test 10L-8H-416D architecture
    """
    n_layer: int = 10
    n_head: int = 8
    d_model: int = 416
    """

    # Optimization 34: Test 10L-8H-448D architecture
    """
    n_layer: int = 10
    n_head: int = 8
    d_model: int = 448
    """

    # dropout: float = 0.1

    # Optimization 09: Reduce dropout to 0.05
    # dropout: float = 0.05

    # Optimization 10: Reduce dropout to 0.02
    # dropout: float = 0.02

    # Optimization 11: Remove dropout
    # dropout: float = 0.0

    # Optimization 30: SwiGLU with 0.02 dropout
    # dropout: float = 0.02

    # Optimization 31: SwiGLU with 0.05 dropout
    dropout: float = 0.05

    # lr: float = 6e-3
    # weight_decay: float = 0.0

    # Optimization 01: Switch to AdamW
    # lr: float = 3e-4
    # weight_decay: float = 0.1

    # Optimization 13: Tune weight decay to 0.05
    # weight_decay: float = 0.05

    # Optimization 14: Tune weight decay to 0.15
    weight_decay: float = 0.15

    # Optimization 15: Tune weight decay to 0.2
    # weight_decay: float = 0.2

    # Optimization 16: Tune weight decay to 0.18
    # weight_decay: float = 0.18

    # Optimization 03: Tune learning rate to 2e-4
    # lr: float = 2e-4

    # Optimization 04: Tune learning rate to 5e-4
    # lr: float = 5e-4

    # Optimization 05: Tune learning rate to 8e-4
    # lr: float = 8e-4

    # Optimization 06: Tune learning rate to 1e-3
    lr: float = 1e-3

    # Optimization 07: Tune learning rate to 1.2e-3
    # lr: float = 1.2e-3

    # Optimization 08: Tune learning rate to 1.1e-3
    # lr: float = 1.1e-3

    # Optimization 02: Add linear learning-rate warmup before cosine decay
    # warmup_ratio: float = 0.05

    # Optimization 17: Tune warmup ratio to 0.02
    # warmup_ratio: float = 0.02

    # Optimization 18: Tune warmup ratio to 0.08
    # warmup_ratio: float = 0.08

    # Optimization 19: Tune warmup ratio to 0.1
    # warmup_ratio: float = 0.1

    # Optimization 20: Tune warmup ratio to 0.12
    # warmup_ratio: float = 0.12

    # Optimization 21: Tune warmup ratio to 0.15
    # warmup_ratio: float = 0.15

    # Optimization 22: Tune warmup ratio to 0.18
    warmup_ratio: float = 0.18

    # Optimization 23: Tune warmup ratio to 0.2
    # warmup_ratio: float = 0.2

    evals_per_epoch: int = 3
    
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = open(log_file, 'w')
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()
            
        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)
    
    return DualLogger(file_handler)

logger = None

def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_batch(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float

# Optimization 35: Replace LayerNorm with RMSNorm
"""
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()

        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)

        return (x_norm * self.weight.float()).to(input_dtype)
"""

# Optimization 28: Replace learned positional embeddings with RoPE
def apply_rotary_embeddings(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to x.

    x shape:   (batch, heads, sequence, head_dim)
    cos/sin:   (1, 1, sequence, head_dim / 2)
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        assert cfg.d_model % cfg.n_head == 0

        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head = cfg.n_head
        
        self.dropout_p = cfg.dropout # For Optimization 36

        assert self.head_dim % 2 == 0, (
            "RoPE requires an even attention head dimension"
        )

        self.qkv = nn.Linear(
            cfg.d_model,
            3 * cfg.d_model,
        )
        self.proj = nn.Linear(
            cfg.d_model,
            cfg.d_model,
        )

        # self.attn_drop = nn.Dropout(cfg.dropout) -> Remove for Optimization 36
        self.resid_drop = nn.Dropout(cfg.dropout)
        
        # Remove for Optimization 36
        """
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    cfg.block_size,
                    cfg.block_size,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )
        """

        # Optimization 28: Replace learned positional embeddings with RoPE
        inv_freq = 1.0 / (
            10_000
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    dtype=torch.float32,
                )
                / self.head_dim
            )
        )

        positions = torch.arange(
            cfg.block_size,
            dtype=torch.float32,
        )

        frequencies = torch.outer(
            positions,
            inv_freq,
        )

        self.register_buffer(
            "rope_cos",
            frequencies.cos(),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            frequencies.sin(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()

        qkv = self.qkv(x).view(
            B,
            T,
            3,
            self.n_head,
            self.head_dim,
        )

        # Shape: (3, B, n_head, T, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        # Shape: (1, 1, T, head_dim / 2)
        cos = self.rope_cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[:T].unsqueeze(0).unsqueeze(0)

        # RoPE is applied only to queries and keys
        q = apply_rotary_embeddings(q, cos, sin)
        k = apply_rotary_embeddings(k, cos, sin)

        # Optimization 36: PyTorch SDPA
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        """
        att = (
            q @ k.transpose(-2, -1)
        ) * (1.0 / math.sqrt(self.head_dim))

        att = att.masked_fill(
            ~self.tril[:T, :T],
            float("-inf"),
        )

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        """

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.resid_drop(
            self.proj(y)
        )

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        """
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        """

        # Optimization 29: Switch to SwiGLU MLP
        hidden_dim = int(8 * cfg.d_model / 3)
        hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.gate_proj = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x): # return self.net(x)
        # Optimization 29: Switch to SwiGLU MLP
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)

        x = gate * value
        x = self.down_proj(x)

        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        # Optimization 35: Replace LayerNorm with RMSNorm
        # self.ln1 = RMSNorm(cfg.d_model)
        # self.ln2 = RMSNorm(cfg.d_model)

        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model)) -> Remove for Optimization 28
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        self.ln_f      = nn.LayerNorm(cfg.d_model)

        # Optimization 35: Replace final LayerNorm with RMSNorm
        # self.ln_f = RMSNorm(cfg.d_model)

        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Optimization 27: Add residual initialization scaling
        residual_init_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, param in self.named_parameters():
            # if name.endswith("attn.proj.weight") or name.endswith("mlp.net.2.weight"): -> For Opti 29
            if name.endswith("attn.proj.weight") or name.endswith("mlp.down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_init_std)

        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx)
        # pos = self.pos_emb[:, :T, :] -> Remove for Optimization 28
        # x = self.drop(tok + pos)
        x = self.drop(tok)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

def main():
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    global logger
    logger = configure_logging(args.log_file)
    
    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    
    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token))
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    
    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)
    
    # opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Optimization 01: Switch to AdamW
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    # Optimization 12: Exclude LayerNorm and bias parameters from weight decay
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Do not apply weight decay to biases or LayerNorm parameters
        if name.endswith(".bias") or "ln" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    """

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    # Optimization 02: Add linear learning-rate warmup before cosine decay
    warmup_steps = max(1, int(args.warmup_ratio * max_steps))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max_steps - warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    logger.log(
        "scheduler_info",
        scheduler="linear_warmup_cosine_decay",
        warmup_steps=warmup_steps,
        warmup_ratio=args.warmup_ratio
    )

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr = 0
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            elapsed = time.time() - t0
            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=loss.item(),
                      elapsed_time=elapsed,
                      prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)

if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
