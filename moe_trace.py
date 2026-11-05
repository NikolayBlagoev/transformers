"""MoE routing tracer: logs routed experts per token for every MoE router, plus speculated experts for the next
1..K MoE layers, for every forward pass. Written to study expert caching / prefetching (frequency, recency and
speculative signals).

Speculation for MoE layer `j` from distance `k`: take the hidden states that enter the pre-router norm of the MoE layer
`k` MoE layers *earlier* (i.e. the residual stream right before that layer's `post_attention_layernorm`), and run them
through layer `j`'s own norm and router. "Layers ahead" counts MoE layers (on Nemotron-H, Mamba / attention blocks have
no router); the absolute layer index of every MoE layer is in the log.

Supported layouts (anything else matching them works too, e.g. Qwen3-MoE, DeepSeek-V3):
    - decoder layer with `post_attention_layernorm` and `mlp.gate`   (Qwen3.5-MoE, DeepSeek-V4)
    - block with `norm` and `mixer.gate`, `block_type == "moe"`      (Nemotron-H / Nemotron 3)
Routers must return `(logits, weights, indices)`. DeepSeek-V4 hash routers (`tid2eid`) are called with the input ids.

Output: one JSON line per forward pass:
    {"forward": 3, "time": 12.34, "tokens": {"batch_id": [...], "position": [...], "token_id": [...], "phase": [...]},
     "layers": [{"layer": 5, "moe_idx": 2, "router": "topk",
                 "experts": [[e, ...] per token], "weights": [[w, ...] per token],
                 "spec": {"1": {"from_layer": 4, "experts": [[...]], "weights": [[...]]}, "2": ...}}, ...]}
`spec["k"]` in layer j's record holds the prediction made `k` MoE layers before j. Token lists are aligned with the
rows of `experts` / `weights`. Pad tokens and anything not annotated by the driver are dropped. `time` is seconds
since the tracer was attached, taken when the forward pass finished (logging the records adds a host sync).
"""

import json
import re
import time
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MoeLayer:
    layer_idx: int  # absolute decoder layer index
    moe_idx: int  # index among MoE layers only
    layer: nn.Module
    norm: nn.Module
    router: nn.Module

    @property
    def is_hash(self) -> bool:
        return hasattr(self.router, "tid2eid")


def find_moe_layers(model: nn.Module) -> list[MoeLayer]:
    found = []
    for name, module in model.named_modules():
        match = re.search(r"layers\.(\d+)$", name)
        if match is None:
            continue
        if hasattr(module, "post_attention_layernorm") and hasattr(getattr(module, "mlp", None), "gate"):
            norm, router = module.post_attention_layernorm, module.mlp.gate
        elif getattr(module, "block_type", None) == "moe" and hasattr(module, "norm"):
            norm, router = module.norm, module.mixer.gate
        else:
            continue
        if not hasattr(router, "top_k"):  # dense MLP with a `gate` projection, not a router
            continue
        found.append((int(match.group(1)), module, norm, router))
    found.sort(key=lambda x: x[0])
    return [MoeLayer(idx, i, layer, norm, router) for i, (idx, layer, norm, router) in enumerate(found)]


class MoeRoutingTracer:
    """Attach with `tracer = MoeRoutingTracer(model, "trace.jsonl")`, detach with `tracer.close()`.

    The driver may call `annotate(...)` before (or during) each forward to say which flattened tokens are real and
    which batch element / position they belong to; otherwise every non-pad token is logged with batch_id = row.
    """

    def __init__(self, model: nn.Module, path: str, lookahead: int = 4):
        self.layers = find_moe_layers(model)
        if not self.layers:
            raise ValueError("No MoE routers found in the model")
        self.lookahead = lookahead
        self.file = open(path, "w")
        self.forward_idx = 0
        self.start_time = time.perf_counter()
        self.annotation = None
        self._reset_forward()
        # overlap[k] = [sum over tokens of |predicted ∩ routed|, sum of |routed|] for a quick sanity summary
        self.overlap = {k: [0, 0] for k in range(1, lookahead + 1)}

        self.handles = [
            model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True),
            model.register_forward_hook(self._model_post_hook),
        ]
        for moe in self.layers:
            self.handles.append(moe.norm.register_forward_pre_hook(self._make_norm_hook(moe)))
            self.handles.append(moe.router.register_forward_hook(self._make_router_hook(moe)))

    # ---- driver API -------------------------------------------------------------------------------------------------
    def annotate(self, flat_index, batch_id, position, token_id, phase):
        """Lists (aligned) describing the tokens to log for the next forward, indexed into the flattened [B*S] tokens."""
        self.annotation = {
            "flat_index": list(flat_index),
            "batch_id": list(batch_id),
            "position": list(position),
            "token_id": list(token_id),
            "phase": list(phase),
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.file.close()

    def summary(self) -> dict[int, float]:
        """Fraction of routed experts that the k-ahead speculation got right, over everything logged."""
        return {k: hit / total for k, (hit, total) in self.overlap.items() if total}

    # ---- hooks ------------------------------------------------------------------------------------------------------
    def _reset_forward(self):
        self.routed = {}  # moe_idx -> (indices [T, k], weights [T, k])
        self.spec = {}  # target moe_idx -> {distance: (indices, weights)}
        self.input_ids = None
        self.default_tokens = None

    def _model_pre_hook(self, module, args, kwargs):
        self._reset_forward()
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return
        self.input_ids = input_ids
        # Default token metadata: every non-pad token, batch_id = row, position from the attention mask
        batch, seq = input_ids.shape
        mask = kwargs.get("attention_mask")
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            positions = (mask.long().cumsum(-1) - 1)[:, -seq:]
            valid = mask[:, -seq:].bool()
        else:
            cache_position = kwargs.get("cache_position")
            start = int(cache_position[0]) if cache_position is not None else 0
            positions = torch.arange(start, start + seq, device=input_ids.device).expand(batch, seq)
            valid = torch.ones_like(input_ids, dtype=torch.bool)
        rows, cols = valid.nonzero(as_tuple=True)
        self.default_tokens = {
            "flat_index": (rows * seq + cols).tolist(),
            "batch_id": rows.tolist(),
            "position": positions[rows, cols].tolist(),
            "token_id": input_ids[rows, cols].tolist(),
            "phase": ["prefill" if seq > 1 else "decode"] * len(rows),
        }

    def _route(self, moe: MoeLayer, hidden_states):
        """Runs a layer's norm + router on `hidden_states`, bypassing hooks. Returns (indices, weights) as [T, k]."""
        device = next(moe.norm.parameters(), hidden_states).device
        if device.type != "meta":
            hidden_states = hidden_states.to(device)
        normed = moe.norm.forward(hidden_states)
        flat = normed.reshape(-1, normed.shape[-1])
        if moe.is_hash:
            out = moe.router.forward(flat, self.input_ids.to(flat.device))
        else:
            out = moe.router.forward(flat)
        return out[-1].reshape(flat.shape[0], -1), out[-2].reshape(flat.shape[0], -1)

    def _make_norm_hook(self, moe: MoeLayer):
        def hook(module, args):
            hidden_states = args[0]
            with torch.no_grad():
                for k in range(1, self.lookahead + 1):
                    target_idx = moe.moe_idx + k
                    if target_idx >= len(self.layers):
                        break
                    self.spec.setdefault(target_idx, {})[k] = self._route(self.layers[target_idx], hidden_states)

        return hook

    def _make_router_hook(self, moe: MoeLayer):
        def hook(module, args, output):
            indices, weights = output[-1], output[-2]
            self.routed[moe.moe_idx] = (indices.reshape(-1, indices.shape[-1]), weights.reshape(-1, weights.shape[-1]))

        return hook

    def _model_post_hook(self, module, args, output):
        tokens = self.annotation or self.default_tokens
        self.annotation = None
        if tokens is None or not self.routed:
            return
        keep = tokens["flat_index"]
        records = []
        for moe in self.layers:
            if moe.moe_idx not in self.routed:
                continue
            indices, weights = self.routed[moe.moe_idx]
            keep_t = torch.tensor(keep, device=indices.device, dtype=torch.long)
            indices, weights = indices[keep_t], weights[keep_t]
            record = {
                "layer": moe.layer_idx,
                "moe_idx": moe.moe_idx,
                "router": "hash" if moe.is_hash else "topk",
                "experts": indices.tolist(),
                "weights": [[round(w, 5) for w in row] for row in weights.float().tolist()],
                "spec": {},
            }
            for k, (spec_idx, spec_w) in sorted(self.spec.get(moe.moe_idx, {}).items()):
                spec_idx, spec_w = spec_idx.to(indices.device)[keep_t], spec_w[keep_t.to(spec_w.device)]
                record["spec"][str(k)] = {
                    "from_layer": self.layers[moe.moe_idx - k].layer_idx,
                    "experts": spec_idx.tolist(),
                    "weights": [[round(w, 5) for w in row] for row in spec_w.float().tolist()],
                }
                hits = (spec_idx.unsqueeze(-1) == indices.unsqueeze(-2)).any(-2).sum().item()
                self.overlap[k][0] += hits
                self.overlap[k][1] += indices.numel()
            records.append(record)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        line = {
            "forward": self.forward_idx,
            "time": round(time.perf_counter() - self.start_time, 4),
            "tokens": {k: v for k, v in tokens.items() if k != "flat_index"},
        }
        line["layers"] = records
        self.file.write(json.dumps(line) + "\n")
        self.file.flush()
        self.forward_idx += 1
        self._reset_forward()
