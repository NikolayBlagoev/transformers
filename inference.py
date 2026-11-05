"""Minimal batched greedy inference that prints, for every forward pass, the token id(s) each batch element processes.
Optionally logs MoE routing (routed + speculated experts per token) with `--moe-trace trace.jsonl`, see moe_trace.py.

Two engines:
    cb        transformers' continuous batching (paged KV cache, flattened batches). Attention-only models.
    generate  `model.generate` with a padded batch and the model's own cache. Works for hybrid models
              (Qwen3.5-MoE linear attention, Nemotron-H Mamba, DeepSeek-V4 compressed attention).
`--engine auto` (default) picks `cb` when every layer is full / sliding attention, `generate` otherwise.

    python inference.py --model Qwen/Qwen3-0.6B --max-new-tokens 16 "Hello, my name is" "The capital of France is"
    python inference.py --model Qwen/Qwen3.5-35B-A3B --moe-trace trace.jsonl
"""

import argparse

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, ContinuousBatchingConfig, GenerationConfig
from transformers.generation.continuous_batching import ContinuousBatchingManager


CB_LAYER_TYPES = {"full_attention", "sliding_attention"}


def print_step(header, items):
    """items: one entry per batch element, either a list of token ids, a (prefill) list, or a status string."""
    print(header)
    for b, (status, ids) in enumerate(items):
        if status == "prefill":
            print(f"batch_id {b}: [prefill, {len(ids)} tokens] {ids}")
        elif status == "decode":
            print(f"batch_id {b}: {ids}")
        else:
            print(f"batch_id {b}: <{status}>")


# ---- continuous batching engine ---------------------------------------------------------------------------------------
class PrintingManager(ContinuousBatchingManager):
    """Continuous batching manager whose per-step loop prints the token ids each request processes."""

    def __init__(self, *args, num_requests, tracer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_requests = num_requests
        self.tracer = tracer
        self.step = 0

    def _generation_step(self):
        # Called once per forward, after the scheduler has built the flattened batch and before the model runs
        processor = self.batch_processor
        io = processor.inputs_and_outputs
        flat_ids = io.input_ids[: io.num_q_tokens].tolist()
        scheduled, offset = {}, 0
        meta = {"flat_index": [], "batch_id": [], "position": [], "token_id": [], "phase": []}
        for future_state in io.requests_in_batch:
            state, q_len = future_state.state, future_state.query_length
            ids = flat_ids[offset : offset + q_len]
            phase = "prefill" if state.generated_len() == 0 else "decode"
            b = int(state.request_id)
            scheduled[b] = (phase, ids)
            # position_offset was already advanced past this step's tokens by the scheduler
            meta["flat_index"] += range(offset, offset + q_len)
            meta["batch_id"] += [b] * q_len
            meta["position"] += range(state.position_offset - q_len, state.position_offset)
            meta["token_id"] += ids
            meta["phase"] += [phase] * q_len
            offset += q_len

        def status(b):
            if b in scheduled:
                return scheduled[b]
            pending = str(b) in processor.scheduler.active_requests or str(b) in processor.scheduler.waiting_requests
            return ("waiting" if pending else "finished", None)

        print_step(
            f"=== step {self.step} ({offset} tokens in the flattened batch) ===",
            [status(b) for b in range(self.num_requests)],
        )
        if self.tracer is not None:
            self.tracer.annotate(**meta)
        super()._generation_step()
        self.step += 1


def generate_cb(model, tokenizer, prompts, generation_config, tracer):
    # Hooks (printing / tracing) don't run inside replayed CUDA graphs, and async batching fills in input ids on device
    cb_config = ContinuousBatchingConfig(use_cuda_graph=False, use_async_batching=False)
    manager = PrintingManager(
        model=model,
        generation_config=generation_config,
        continuous_batching_config=cb_config,
        num_requests=len(prompts),
        tracer=tracer,
    )
    # Queue every request before starting, so the first step schedules all of them together
    for b, prompt in enumerate(prompts):
        manager.add_request(tokenizer(prompt).input_ids, request_id=str(b))
    manager.start()
    results = {}
    try:
        while len(results) < len(prompts):
            result = manager.get_result(timeout=1)
            if result is not None and result.error is not None:
                raise RuntimeError(f"Request {result.request_id} failed: {result.error}")
            if result is not None and result.is_finished():
                results[result.request_id] = result
            elif result is None and not manager.is_running():
                raise RuntimeError("Generation thread terminated unexpectedly")
    finally:
        manager.stop(block=True, timeout=5.0)
    return [results[str(b)].generated_tokens for b in range(len(prompts))]


# ---- generate engine --------------------------------------------------------------------------------------------------
def generate_padded(model, tokenizer, prompts, generation_config, tracer):
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    eos = generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    finished = set()
    step = [0]

    def hook(module, args, kwargs):
        # Called before every forward of `generate`: input_ids holds what each row processes in this step
        input_ids, mask = kwargs["input_ids"], kwargs.get("attention_mask")
        seq = input_ids.shape[1]
        if mask is None:  # generate drops the mask when nothing is padded
            past = kwargs.get("past_key_values")
            past_len = past.get_seq_length() if past is not None else 0
            mask = torch.ones(input_ids.shape[0], past_len + seq, dtype=torch.long, device=input_ids.device)
        positions = (mask.long().cumsum(-1) - 1)[:, -seq:].tolist()
        valid = mask[:, -seq:].bool().tolist()
        items = []
        meta = {"flat_index": [], "batch_id": [], "position": [], "token_id": [], "phase": []}
        phase = "prefill" if step[0] == 0 else "decode"
        for b, row in enumerate(input_ids.tolist()):
            if b in finished:  # generate keeps feeding pad tokens for finished rows
                items.append(("finished", None))
                continue
            cols = [c for c in range(seq) if valid[b][c]]
            ids = [row[c] for c in cols]
            items.append((phase, ids))
            meta["flat_index"] += [b * seq + c for c in cols]
            meta["batch_id"] += [b] * len(cols)
            meta["position"] += [positions[b][c] for c in cols]
            meta["token_id"] += ids
            meta["phase"] += [phase] * len(cols)
            if phase == "decode" and ids[-1] in eos:
                finished.add(b)  # the EOS token itself is processed this step, pads from the next one on
        print_step("=== prefill ===" if step[0] == 0 else f"=== decode step {step[0]} ===", items)
        if tracer is not None:
            tracer.annotate(**meta)
        step[0] += 1

    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        out = model.generate(**batch, generation_config=generation_config)
    finally:
        handle.remove()
    new_tokens = out[:, batch["input_ids"].shape[1] :].tolist()
    # Strip what comes after the first EOS (generate pads finished rows)
    results = []
    for row in new_tokens:
        end = next((i + 1 for i, t in enumerate(row) if t in eos), len(row))
        results.append(row[:end])
    return results


# ---- main -------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts", nargs="*", default=["Hello, my name is", "The capital of France is a city that"])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--engine", choices=["auto", "cb", "generate"], default="auto")
    parser.add_argument("--attn", default=None, help="attention implementation, e.g. kernels-community/flash-attn")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--moe-trace", default=None, help="write MoE routing logs (JSONL) to this path")
    parser.add_argument("--lookahead", type=int, default=4, help="how many MoE layers ahead to speculate")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"attn_implementation": args.attn} if args.attn else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto", **model_kwargs).eval()
    run(model, tokenizer, args.prompts, args.max_new_tokens, args.engine, args.moe_trace, args.lookahead)


def run(model, tokenizer, prompts, max_new_tokens, engine="auto", moe_trace=None, lookahead=4):
    if engine == "auto":
        layer_types = getattr(model.config.get_text_config(), "layer_types", None) or ["full_attention"]
        engine = "cb" if set(layer_types) <= CB_LAYER_TYPES else "generate"

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=model.generation_config.eos_token_id or tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    tracer = None
    if moe_trace:
        from moe_trace import MoeRoutingTracer

        tracer = MoeRoutingTracer(model, moe_trace, lookahead=lookahead)
        print(f"Tracing {len(tracer.layers)} MoE layers: {[moe.layer_idx for moe in tracer.layers]}")

    try:
        with torch.inference_mode():
            fn = generate_cb if engine == "cb" else generate_padded
            outputs = fn(model, tokenizer, prompts, generation_config, tracer)
    finally:
        if tracer is not None:
            tracer.close()

    print(f"=== outputs ({engine}) ===")
    for b, (prompt, ids) in enumerate(zip(prompts, outputs)):
        print(f"batch_id {b}: {prompt!r} -> {tokenizer.decode(ids, skip_special_tokens=True)!r}")
    if tracer is not None:
        print(f"MoE trace written to {moe_trace}; speculation hit rate by distance: {tracer.summary()}")
    return outputs


if __name__ == "__main__":
    main()
