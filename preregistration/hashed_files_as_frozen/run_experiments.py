"""
Context-dependent reconfiguration experiments (R4), Families 1-4.
Usage: python3 run_experiments.py F1|F2|F3|F4|PILOT
Model: Qwen/Qwen2.5-1.5B-Instruct, bfloat16, CPU, chat template,
greedy decoding for generation (max 80 new tokens), seed 42.
Internal measures use the English reference map (derive_map.py):
per-token emotion score s_e = sum_l z_l @ W_l, where z = (act - mu)/sd on
the derivation split and W_l spreads 1/|S_e| over the FDR-corrected
signature neurons of emotion e. Behaviour: next-token log-probabilities of
the six label words after the assistant prefix "Answer:" (forced choice),
or an independent emotion classifier applied to generated text.
"""
import sys, os, json, time, re
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/claude/r4"); sys.path.insert(0, "/home/claude/r4/stimuli")
from common import EM, LAYERS, FFN, load_model, Hooks
torch.manual_seed(42)
FAM = sys.argv[1]
OUTD = "/home/claude/r4/results/experiments"; os.makedirs(OUTD, exist_ok=True)
NOUN = {"ANGER": "anger", "DISGUST": "disgust", "FEAR": "fear", "HAPPINESS": "happiness",
        "SADNESS": "sadness", "SURPRISE": "surprise"}
tok, m = load_model(); H = Hooks(m)
LABEL_IDS = [tok.encode(" " + NOUN[e], add_special_tokens=False)[0] for e in EM]
ref = np.load("/home/claude/r4/cache/reference_map_en.npz")
MU = torch.tensor(ref["mu"]); SD = torch.tensor(ref["sd"]); W = torch.tensor(ref["W"])
t0 = time.time()

def chat(user, prefix=""):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True) + prefix

def char_span(full, sub, start_hint=0):
    i = full.find(sub, start_hint)
    assert i >= 0, sub[:40]
    return (i, i + len(sub))

def forward(strings, spans_list):
    """strings: list of prompts; spans_list: list of dicts name->(c0,c1) char ranges.
    Returns per sample: label logprobs (6), pooled scores per span (6), final-token
    scores (6), pooled raw activations per span and final token (9 x FFN, fp16)."""
    tok.padding_side = "right"
    enc = tok(strings, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    with torch.no_grad():
        out = m(**enc)
    lens = enc["attention_mask"].sum(1)
    res = []
    for b in range(len(strings)):
        L = int(lens[b]); last = L - 1
        lp = torch.log_softmax(out.logits[b, last].float(), -1)[LABEL_IDS].numpy()
        acts = torch.stack([H.out[l][b, :L] for l in LAYERS])          # 9 x L x FFN
        z = (acts - MU[:, None, :]) / SD[:, None, :]
        tokscore = torch.einsum("ltf,lfe->te", z, W)                    # L x 6
        r = {"lp": lp.tolist(), "final": tokscore[last].tolist(), "spans": {}, "raw": {}}
        r["raw"]["final"] = acts[:, last].half().numpy()
        for name, (c0, c1) in spans_list[b].items():
            o = offs[b, :L]
            msk = (o[:, 0] >= c0) & (o[:, 1] <= c1) & (o[:, 1] > o[:, 0])
            if msk.sum() == 0:
                continue
            r["spans"][name] = tokscore[msk].mean(0).tolist()
            r["raw"][name] = acts[:, msk].mean(1).half().numpy()
        res.append(r)
    return res

def generate(prompts, max_new=80):
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None,
                       top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    return [tok.decode(g[i, enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for i in range(len(prompts))]

def save(rows, raws, tag):
    with open(f"{OUTD}/{tag}.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for k, v in raws.items():
        np.save(f"{OUTD}/raw_{tag}_{k}_{time.time_ns()}.npy", np.stack(v))

def run_trials(trials, tag, bs=8):
    done = set()
    if os.path.exists(f"{OUTD}/{tag}.jsonl"):
        done = {json.loads(l)["trial"] for l in open(f"{OUTD}/{tag}.jsonl")}
    todo = [t for t in trials if t["trial"] not in done]
    print(tag, "trials", len(trials), "remaining", len(todo), flush=True)
    for s in range(0, len(todo), bs):
        chunk = todo[s:s + bs]
        res = forward([c["prompt"] for c in chunk], [c["spans"] for c in chunk])
        rows, raws = [], {}
        for c, r in zip(chunk, res):
            row = {k: v for k, v in c.items() if k not in ("spans",)}
            row.update(lp=r["lp"], final=r["final"], span_scores=r["spans"])
            rows.append(row)
            for k, v in r["raw"].items():
                raws.setdefault(k, []).append(v)
            raws.setdefault("trialids", [])
        save(rows, {}, tag)
        np.savez_compressed(f"{OUTD}/raw_{tag}_{s:06d}.npz", trial=np.array([c["trial"] for c in chunk]),
                            **{k: np.stack(v) for k, v in raws.items() if k != "trialids"})
        if (s // bs) % 25 == 0:
            print(f"{tag} {s+len(chunk)}/{len(todo)} {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------ families
R_INSTR = [
    "Read the following text and identify every emotion that may be present in it.",
    "Which emotions could the writer of the following text be feeling? Consider all that apply.",
    "List all the emotions that might be expressed in the text below.",
    "Consider the text below and name any emotions it may convey.",
]
D_INSTR = [
    "Read the following text and decide whether the writer feels {a} or {b}. Choose exactly one.",
    "Is the emotion in the text below {a} or {b}? Pick the single better answer.",
    "Classify the text below as expressing either {a} or {b}, choosing one.",
    "Between {a} and {b}, which emotion best describes the writer of the text below?",
]

def f1_trials():
    items = pd.read_csv("/home/claude/r4/stimuli/f1_items.csv"); T = []
    for _, it in items.iterrows():
        a, b = it.pair.split("-")
        for cond in ("RECOGNISE", "DISCRIMINATE"):
            for p in range(4):
                if cond == "RECOGNISE":
                    ins = R_INSTR[p]
                else:
                    x, y = (a, b) if p % 2 == 0 else (b, a)
                    ins = D_INSTR[p].format(a=NOUN[x], b=NOUN[y])
                user = f"{ins}\n\nText: {it.text}"
                full = chat(user, "Answer:")
                T.append({"trial": f"{it['item']}|{cond}|{p}", "item": it["item"], "pair": it.pair, "gold": it.gold,
                          "cond": cond, "para": p, "prompt": full,
                          "spans": {"stim": char_span(full, it.text, full.find("Text: "))}})
    return T

def f2_trials():
    import f2_disambiguation as S
    T = []
    for (a, b), items in S.PAIRS.items():
        for k, (base, ev_a, ev_b) in enumerate(items):
            item = f"F2_{a[:3]}{b[:3]}_{k:02d}"
            conts = {"NONE": "", "FILLER": S.NEUTRAL[k % len(S.NEUTRAL)], "EV_A": ev_a, "EV_B": ev_b}
            for cond, cont in conts.items():
                for p in range(4):
                    x, y = (a, b) if p % 2 == 0 else (b, a)
                    acc = base + (" " + cont if cont else "")
                    user = S.INSTR[p].format(a=NOUN[x], b=NOUN[y]) + f"\n\nAccount: {acc}"
                    full = chat(user, "Answer:")
                    sp = {"base": char_span(full, base)}
                    if cont:
                        sp["cont"] = char_span(full, cont, sp["base"][1])
                    T.append({"trial": f"{item}|{cond}|{p}", "item": item, "pair": f"{a}-{b}", "a": a, "b": b,
                              "cond": cond, "para": p, "prompt": full, "spans": sp})
    return T

def gen_trials(fam):
    import f3_f4_generation as G
    T = []
    if fam == "F3":
        for k, fact in enumerate(G.FACTS):
            for e in EM:
                for p in range(4):
                    user = G.TONE_INSTR[p].format(e=NOUN[e]) + f"\n\nStatement: {fact}"
                    T.append({"trial": f"F3_{k:02d}|{e}|{p}", "item": f"F3_{k:02d}", "target": e, "para": p,
                              "user": user})
    else:
        for e, sits in (("ANGER", G.ANGER_SIT), ("FEAR", G.FEAR_SIT)):
            for k, sit in enumerate(sits):
                for cond in ("EXPRESS", "SUPPRESS", "NEUTRAL"):
                    for p in range(4):
                        user = f"Situation: {sit}\n\n" + G.F4_INSTR[cond][p].format(e=NOUN[e])
                        T.append({"trial": f"F4_{e[:3]}_{k:02d}|{cond}|{p}", "item": f"F4_{e[:3]}_{k:02d}",
                                  "target": e, "cond": cond, "para": p, "user": user})
    return T

def run_generation(trials, tag, bs=8):
    from transformers import pipeline
    clf = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    done = set()
    if os.path.exists(f"{OUTD}/{tag}.jsonl"):
        done = {json.loads(l)["trial"] for l in open(f"{OUTD}/{tag}.jsonl")}
    todo = [t for t in trials if t["trial"] not in done]
    print(tag, "trials", len(trials), "remaining", len(todo), flush=True)
    for s in range(0, len(todo), bs):
        chunk = todo[s:s + bs]
        prompts = [chat(c["user"]) for c in chunk]
        outs = generate(prompts)
        fulls = [p + o for p, o in zip(prompts, outs)]
        spans = [{"gen": (len(p), len(p) + len(o))} for p, o in zip(prompts, outs)]
        res = forward(fulls, spans)
        # planning state: final prompt token (the newline after the assistant tag), from a
        # separate forward pass over the prompt alone, i.e. the state before any output token
        resp = forward(prompts, [{} for _ in prompts])
        cls = clf([o if o else "." for o in outs], truncation=True)
        rows, raws = [], {}
        for c, o, r, rp, cl in zip(chunk, outs, res, resp, cls):
            row = {k: v for k, v in c.items() if k != "user"}
            row.update(output=o, gen_score=r["spans"].get("gen"), plan_score=rp["final"],
                       clf={d["label"]: d["score"] for d in cl})
            rows.append(row)
            raws.setdefault("gen", []).append(r["raw"].get("gen", np.zeros((len(LAYERS), FFN), np.float16)))
            raws.setdefault("prompt_last", []).append(rp["raw"]["final"])
        save(rows, {}, tag)
        np.savez_compressed(f"{OUTD}/raw_{tag}_{s:06d}.npz", trial=np.array([c["trial"] for c in chunk]),
                            **{k: np.stack(v) for k, v in raws.items()})
        print(f"{tag} {s+len(chunk)}/{len(todo)} {time.time()-t0:.0f}s", flush=True)

if FAM == "F1":
    run_trials(f1_trials(), "F1")
elif FAM == "F2":
    run_trials(f2_trials(), "F2")
elif FAM in ("F3", "F4"):
    run_generation(gen_trials(FAM), FAM)
elif FAM == "PILOT":   # pipeline check on items that are NOT study stimuli
    full = chat("Read the following text and decide whether the writer feels fear or sadness. Choose exactly one.\n\nText: The pilot sentence about a rainy afternoon.", "Answer:")
    r = forward([full], [{"stim": char_span(full, "The pilot sentence about a rainy afternoon.")}])[0]
    print("pilot lp", np.round(r["lp"], 2), "span", np.round(r["spans"]["stim"], 3), f"{time.time()-t0:.1f}s")
    tg = time.time(); o = generate([chat("Write one sentence about a pilot test of a bicycle bell.")] * 8)
    print("gen 8 in", round(time.time() - tg, 1), "s :", o[0][:120])
