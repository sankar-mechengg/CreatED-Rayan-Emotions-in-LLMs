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

import sys, logging, numpy as np, pandas as pd
from preprocessing import load_and_clean
from utils import strip_brackets
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LAYERS = list(range(14, 23)); FFN = 8960
EM = ["ANGER","DISGUST","FEAR","HAPPINESS","SADNESS","SURPRISE"]
CSV = str(DATA_CSV)

def load_df():
    lg = logging.getLogger("x"); lg.addHandler(logging.NullHandler())
    df = load_and_clean(CSV, lg).reset_index(drop=True)
    df["primary_emotion"] = df["primary_emotion"].str.upper().str.strip()
    return df

def load_model():
    import torch
    torch.set_num_threads(2)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    return tok, m

class Hooks:
    """Capture post-SwiGLU product per layer, full sequence (float32)."""
    def __init__(self, m):
        self.out = {}; self.h = []
        for i in LAYERS:
            def mk(idx):
                def f(mod, inp, o):
                    x = inp[0]
                    self.out[idx] = (mod.act_fn(mod.gate_proj(x)) * mod.up_proj(x)).float()
                return f
            self.h.append(m.model.layers[i].mlp.register_forward_hook(mk(i)))
    def remove(self):
        for h in self.h: h.remove()
