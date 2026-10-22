import sys, logging, numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/claude/pipe/emotion_neuron_analysis")
from preprocessing import load_and_clean
from utils import strip_brackets
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LAYERS = list(range(14, 23)); FFN = 8960
EM = ["ANGER","DISGUST","FEAR","HAPPINESS","SADNESS","SURPRISE"]
CSV = "/mnt/user-data/uploads/emotion_dataset_translated.csv"
torch.set_num_threads(2)

def load_df():
    lg = logging.getLogger("x"); lg.addHandler(logging.NullHandler())
    df = load_and_clean(CSV, lg).reset_index(drop=True)
    df["primary_emotion"] = df["primary_emotion"].str.upper().str.strip()
    return df

def load_model():
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
