import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_paths import (
    DATA_CSV, EARLIER_OVERLAP, CACHE, RESULTS, STIMULI, REFERENCE, FIGURES,
    PIPELINE, load_nrc_lexicon, text_open, apply_figure_style, ensure_runtime_dirs,
)
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(STIMULI))
ensure_runtime_dirs()

import sys, time, json, numpy as np, torch
from common import *
if len(sys.argv) < 2 or sys.argv[1] not in ("zh", "en"):
    raise SystemExit("Usage: python extract_baseline.py zh|en [resume_row]")
lang = sys.argv[1]  # zh or en
df = load_df()
col = "chinese_text" if lang == "zh" else "english_text"
texts = [strip_brackets(str(t)) for t in df[col].values]
N = len(texts)
tok, m = load_model(); H = Hooks(m)
tok.padding_side = "right"
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # resume point (multiple of BS) after an interruption
if START:
    out = np.lib.format.open_memmap(str(CACHE / f"gate_{lang}.npy"), mode="r+")
else:
    out = np.lib.format.open_memmap(str(CACHE / f"gate_{lang}.npy"), mode="w+", dtype=np.float16, shape=(N, len(LAYERS), FFN))
lens = np.array([len(tok(t).input_ids) for t in texts])
order = np.argsort(lens)
BS = 32; t0 = time.time()
for b in range(START, N, BS):
    idx = order[b:b+BS]
    enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    with torch.no_grad(): m(**enc)
    pooled = np.stack([((H.out[l]*mask).sum(1)/mask.sum(1).clamp(min=1)).numpy() for l in LAYERS], 1)
    out[idx] = pooled.astype(np.float16)
    if (b//BS) % 50 == 0:
        print(f"{lang} {b+len(idx)}/{N} {time.time()-t0:.0f}s maxlen={enc.input_ids.shape[1]}", flush=True)
out.flush()
np.save(str(CACHE / "labels.npy"), df["primary_emotion"].values.astype(str))
json.dump({"lang": lang, "N": N, "mean_tokens": float(lens.mean()), "max_tokens": int(lens.max()),
           "seconds": time.time()-t0, "dtype": "bfloat16 CPU", "pooling": "attention-masked mean over all tokens",
           "torch": torch.__version__}, text_open(str(CACHE / f"meta_{lang}.json"),"w"), indent=1)
print("DONE", lang, time.time()-t0, flush=True)
