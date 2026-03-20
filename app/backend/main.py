from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Encoder Attention API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/response schemas ────────────────────────────────────────

class InferenceRequest(BaseModel):
    text: str
    model_name: str = "bert-base-uncased"

class LimitsRequest(BaseModel):
    model_name: str = "bert-base-uncased"
    seq_lengths: list[int] = [64, 128, 256, 512]

# ── Model cache ─────────────────────────────────────────────────────

_cache: dict = {}

def get_model(name: str):
    if name not in _cache:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name, output_attentions=True)
            mdl.eval()
            _cache[name] = (tok, mdl)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Model load error: {exc}")
    return _cache[name]

# ── Routes ──────────────────────────────────────────────────────────

@app.post("/api/attention")
def get_attention(req: InferenceRequest):
    """
    Run encoder inference and return per-layer, per-head attention matrices.

    Returns
    -------
    tokens : list[str]
    attention : float[][][][] — [layer][head][row][col]
    """
    tokenizer, model = get_model(req.model_name)
    inputs = tokenizer(
        req.text, return_tensors="pt", truncation=True, max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    if not outputs.attentions:
        raise HTTPException(status_code=500, detail="Model returned no attentions")

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # [layer][head][seq][seq]
    attention_data = [
        [layer_attn[0, h].tolist() for h in range(layer_attn.size(1))]
        for layer_attn in outputs.attentions
    ]

    return {"tokens": tokens, "attention": attention_data}


@app.post("/api/limits")
def probe_limits(req: LimitsRequest):
    """
    For each sequence length in req.seq_lengths, time a forward pass and
    measure peak allocated memory.  Returns latency_ms and memory_mb per length.
    """
    tokenizer, model = get_model(req.model_name)
    results = []

    # Use a max-length tokenizer config, we'll pad/truncate to exact length
    base_ids = tokenizer.encode(
        "the " * 1024, add_special_tokens=True, truncation=False
    )

    for seq_len in req.seq_lengths:
        # Truncate or pad to exact length
        ids = base_ids[:seq_len]
        if len(ids) < seq_len:
            pad_id = tokenizer.pad_token_id or 0
            ids = ids + [pad_id] * (seq_len - len(ids))

        input_ids = torch.tensor([ids])
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == (tokenizer.pad_token_id or 0)] = 0

        try:
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            latency_ms = (time.perf_counter() - t0) * 1000

            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            else:
                mem_mb = None

            results.append({
                "seq_len": seq_len,
                "latency_ms": round(latency_ms, 2),
                "memory_mb": round(mem_mb, 1) if mem_mb is not None else None,
                "status": "ok",
            })
        except RuntimeError as exc:
            results.append({
                "seq_len": seq_len,
                "latency_ms": None,
                "memory_mb": None,
                "status": "oom" if "out of memory" in str(exc).lower() else "error",
            })

    return {"model": req.model_name, "results": results}
