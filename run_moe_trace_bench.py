"""MoE routing traces for garlic's default benchmark: 20 prompts from each of GSM8K, MT-Bench, ShareGPT and HumanEval
(see ../garlic-private/benchmark_runners/run_bench.py), run through inference.py in batches of 10, greedy.

Prompts are fetched from the HuggingFace datasets-server REST API and cleaned the same way as run_bench.py (whitespace
flattened, prompts over 4000 chars skipped), or reused from a prompts.json via --manifest. They are templated like
garlic's runner: its system message + the user turn through the model's chat template. Decoding is greedy.

Per batch, written to --outdir:
    {dataset}_b{k}.jsonl   MoE trace (moe_trace.py format); batch_id i is prompt batch_start + i of that dataset
    {dataset}_b{k}.log     the per-step token-id prints of inference.py
    {dataset}_b{k}.json    the prompts' indices, prompt texts and completions

    python run_moe_trace_bench.py --outdir moe_trace_bench
"""

import argparse
import contextlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

from inference import run
from transformers import AutoModelForCausalLM, AutoTokenizer


# Same system message as garlic's runner (inference.cpp)
SYSTEM_PROMPT = (
    "This is a conversation between a user and an AI chatbot. You are a highly capable, precise, and efficient AI "
    "assistant. Read carefully and understand the user's core intent quickly and avoid unnecessary or overly "
    "exhaustive planning loops. Then provide with them with the best answer for their query."
)


# ---- prompt fetching (stdlib only, mirrors garlic's run_bench.py) ---------------------------------------------------
DATASETS_SERVER = "https://datasets-server.huggingface.co"


def http_get_json(url, timeout=30, retries=3):
    last_err = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            last_err = e
    raise RuntimeError(f"GET {url} failed: {last_err}")


def clean_prompt(text, max_len=4000):
    if not text:
        return None
    flat = " ".join(text.split())
    if not flat or len(flat) > max_len:
        return None
    return flat


def fetch_rows(dataset, config, split, n, row_to_prompt, page_size=50, max_pages=20):
    prompts, offset = [], 0
    for _ in range(max_pages):
        if len(prompts) >= n:
            break
        url = f"{DATASETS_SERVER}/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={page_size}"
        rows = http_get_json(url).get("rows", [])
        if not rows:
            break
        for r in rows:
            p = row_to_prompt(r["row"])
            p = clean_prompt(p) if p is not None else None
            if p:
                prompts.append(p)
                if len(prompts) >= n:
                    break
        offset += page_size
    return prompts[:n]


def first_human_turn(row):
    for turn in row.get("conversations") or []:
        if turn.get("from") == "human":
            return turn.get("value")
    return None


DATASET_FETCHERS = {
    "gsm8k": lambda n: fetch_rows("openai/gsm8k", "main", "test", n, lambda row: row.get("question")),
    # only the first turn of each MT-Bench conversation
    "mtbench": lambda n: fetch_rows(
        "HuggingFaceH4/mt_bench_prompts", "default", "train", n, lambda row: (row.get("prompt") or [None])[0]
    ),
    "sharegpt": lambda n: fetch_rows("shibing624/sharegpt_gpt4", "default", "train", n, first_human_turn),
    "humaneval": lambda n: fetch_rows(
        "openai/openai_humaneval", "openai_humaneval", "test", n, lambda row: row.get("prompt")
    ),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/home/gaylord/qwen35-35B-fp8/")
    ap.add_argument("--n", type=int, default=20, help="prompts per dataset")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_FETCHERS), choices=list(DATASET_FETCHERS))
    ap.add_argument("--manifest", default=None, help="reuse a prompts.json instead of re-fetching")
    ap.add_argument("--outdir", default="moe_trace_bench")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--lookahead", type=int, default=4)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Prompts: same datasets / fetching / cleaning as garlic's run_bench.py
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        all_prompts = {name: manifest[name][: args.n] for name in args.datasets}
    else:
        all_prompts = {}
        for name in args.datasets:
            print(f"[bench] fetching {args.n} prompts for {name}...", file=sys.stderr)
            all_prompts[name] = DATASET_FETCHERS[name](args.n)
            print(f"[bench]   got {len(all_prompts[name])}", file=sys.stderr)
    with open(outdir / "prompts.json", "w") as f:
        json.dump(all_prompts, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto").eval()

    for name, prompts in all_prompts.items():
        for k, start in enumerate(range(0, len(prompts), args.batch_size)):
            batch = prompts[start : start + args.batch_size]
            templated = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for p in batch
            ]
            stem = outdir / f"{name}_b{k}"
            print(f"[bench] {stem} ({len(batch)} prompts) ...", file=sys.stderr, end=" ", flush=True)
            t0 = time.time()
            with open(f"{stem}.log", "w") as log, contextlib.redirect_stdout(log):
                outputs = run(
                    model, tokenizer, templated, args.max_new_tokens, moe_trace=f"{stem}.jsonl", lookahead=args.lookahead
                )
            with open(f"{stem}.json", "w") as f:
                json.dump(
                    {
                        "dataset": name,
                        "prompt_index": list(range(start, start + len(batch))),
                        "prompts": batch,
                        "completion_tokens": outputs,
                        "completions": [tokenizer.decode(o, skip_special_tokens=True) for o in outputs],
                    },
                    f,
                    indent=2,
                )
            n_tok = sum(len(o) for o in outputs)
            print(f"done in {time.time() - t0:.0f}s, {n_tok} tokens generated", file=sys.stderr)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
