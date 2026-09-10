from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

ATTENTION_MEASURE_KEY = "attention_measure_v1"
ATTENTION_MEASURE_CAPABILITIES_KEY = "attention_measure_capabilities_v1"
ATTENTION_MEASURE_API = 1
KEY_LOG_MEASURE_OPERATOR = "key_log_measure"
H3_SOURCE_CARRIER_NORMALIZATION = "h3_native_source_carrier_v1"
MIXED_GRID_TOPOLOGY = "mixed_grid_low_suffix"
H3_COORDINATE_POLICY = "minimax_h3_native_frame_grid_v1"


@dataclass(frozen=True, slots=True)
class BoundMeasurePlan:
    semantic_digest: str
    implementation_profile: str
    owner_generation: str
    numerical_route: str
    q_rows: int
    kv_rows: int
    exact_k_block_range: tuple[int, int]
    exact_range_digest: str
    key_log_measure: torch.Tensor


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"attention measure {name} must be an integer")
    if value < minimum:
        raise ValueError(f"attention measure {name} must be >= {minimum}")
    return value


def _grid(name: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"attention measure {name} must be a two-element grid")
    return _integer(f"{name}[0]", value[0], 1), _integer(f"{name}[1]", value[1], 1)


def normalize(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise TypeError("attention_measure_v1 must be a mapping")
    literals = {
        "api": ATTENTION_MEASURE_API,
        "operator": KEY_LOG_MEASURE_OPERATOR,
        "normalization": H3_SOURCE_CARRIER_NORMALIZATION,
        "topology": MIXED_GRID_TOPOLOGY,
        "coordinate_policy": H3_COORDINATE_POLICY,
    }
    for key, expected in literals.items():
        if request.get(key) != expected:
            raise ValueError(f"attention measure {key} must be {expected!r}")
    q_rows = _integer("q_rows", request.get("q_rows"), 1)
    kv_rows = _integer("kv_rows", request.get("kv_rows"), 1)
    video_start = _integer("video_start", request.get("video_start"))
    temporal = _integer("temporal", request.get("temporal"), 2)
    prefix_t = _integer("prefix_t", request.get("prefix_t"), 1)
    source_grid = _grid("source_grid", request.get("source_grid"))
    prefix_grid = _grid("prefix_grid", request.get("prefix_grid"))
    if q_rows != kv_rows:
        raise ValueError("attention measure requires the all-row self-attention domain")
    if video_start >= kv_rows or prefix_t >= temporal:
        raise ValueError("attention measure video geometry is invalid")
    source_rows = math.prod(source_grid)
    prefix_rows = math.prod(prefix_grid)
    if source_rows >= prefix_rows:
        raise ValueError("mixed-grid measure requires a strictly denser protected prefix")
    if q_rows != video_start + prefix_t * prefix_rows + (temporal - prefix_t) * source_rows:
        raise ValueError("attention measure row count disagrees with mixed-grid geometry")
    raw_segments = request.get("segments")
    if not isinstance(raw_segments, (list, tuple)) or not raw_segments:
        raise TypeError("attention measure segments must be a nonempty list")
    segments = []
    cursor = 0
    for i, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise TypeError(f"attention measure segments[{i}] must be a mapping")
        start = _integer(f"segments[{i}].start", raw.get("start"))
        stop = _integer(f"segments[{i}].stop", raw.get("stop"))
        num = _integer(f"segments[{i}].mass_num", raw.get("mass_num"), 1)
        den = _integer(f"segments[{i}].mass_den", raw.get("mass_den"), 1)
        if start != cursor or stop <= start or stop > kv_rows:
            raise ValueError("attention measure segments must be sorted, contiguous, nonempty and in range")
        ratio = Fraction(num, den)
        segments.append({"start": start, "stop": stop, "mass_num": ratio.numerator, "mass_den": ratio.denominator})
        cursor = stop
    if cursor != kv_rows:
        raise ValueError("attention measure segments must cover every key row exactly once")
    weighted_start = video_start
    weighted_stop = video_start + prefix_t * prefix_rows
    wanted = Fraction(source_rows, prefix_rows)
    for segment in segments:
        start, stop = segment["start"], segment["stop"]
        if start < weighted_start < stop or start < weighted_stop < stop:
            raise ValueError("attention measure boundaries must align with the protected-prefix interval")
        overlap = max(start, weighted_start) < min(stop, weighted_stop)
        ratio = Fraction(segment["mass_num"], segment["mass_den"])
        if overlap and ratio != wanted:
            raise ValueError("protected-prefix keys use the wrong spatial measure")
        if not overlap and ratio != 1:
            raise ValueError("non-prefix keys must retain unit measure")
        if not math.isfinite(math.log(float(ratio))):
            raise ValueError("attention measure derived bias must be finite")
    return {**literals, "q_rows": q_rows, "kv_rows": kv_rows, "video_start": video_start,
            "temporal": temporal, "prefix_t": prefix_t, "source_grid": list(source_grid),
            "prefix_grid": list(prefix_grid), "segments": segments}


def semantic_digest(request: Mapping[str, Any]) -> str:
    encoded = json.dumps(normalize(request), sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_h3(request, *, layout, q_rows, kv_rows, external_sequence=None):
    normalized = normalize(request)
    if normalized["q_rows"] != q_rows or normalized["kv_rows"] != kv_rows:
        raise ValueError("attention measure row counts are stale")
    if layout is None or getattr(layout, "seq_len", None) != q_rows:
        raise ValueError("attention measure requires the actual H3 mixed layout")
    video = next(((a, b) for a, b, kind in getattr(layout, "segments", ()) if kind == "video"), None)
    if video != (normalized["video_start"], q_rows):
        raise ValueError("attention measure video interval disagrees with the H3 layout")
    if external_sequence is not None:
        checks = {"api": 2, "sequence_rows": q_rows, "video_start": normalized["video_start"],
                  "temporal": normalized["temporal"], "prefix_t": normalized["prefix_t"],
                  "source_rows_per_frame": math.prod(normalized["source_grid"]),
                  "prefix_rows_per_frame": math.prod(normalized["prefix_grid"])}
        if not isinstance(external_sequence, Mapping) or any(external_sequence.get(k) != v for k, v in checks.items()):
            raise ValueError("attention measure disagrees with the VDN external-sequence contract")
    return normalized


def materialize(request, *, device, dtype=torch.float32):
    normalized = normalize(request)
    if not dtype.is_floating_point:
        raise TypeError("attention measure bias requires floating point")
    out = torch.empty(normalized["kv_rows"], device=device, dtype=dtype)
    for segment in normalized["segments"]:
        out[segment["start"]:segment["stop"]] = math.log(segment["mass_num"] / segment["mass_den"])
    if not torch.isfinite(out).all():
        raise ValueError("attention measure produced non-finite bias")
    return out.contiguous()


def merge_exact_k_blocks(request, block_size: int, existing=(0, 0)) -> tuple[int, int]:
    normalized = normalize(request)
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if len(existing) != 2 or any(type(x) is not int for x in existing) or existing[0] < 0 or existing[1] < existing[0]:
        raise ValueError("existing sink range is invalid")
    weighted = [s for s in normalized["segments"] if (s["mass_num"], s["mass_den"]) != (1, 1)]
    first = min(s["start"] for s in weighted) // block_size
    last = (max(s["stop"] for s in weighted) + block_size - 1) // block_size
    if existing[0] == existing[1]:
        return first, last
    return min(existing[0], first), max(existing[1], last)


def bind(request, *, layout, q_rows, kv_rows, block_size, existing_sink, device,
         implementation_profile, owner_generation, numerical_route, external_sequence=None):
    normalized = validate_h3(request, layout=layout, q_rows=q_rows, kv_rows=kv_rows, external_sequence=external_sequence)
    if implementation_profile not in {"dense_exact_v1", "weighted_exact_blocks_v1"}:
        raise ValueError("unsupported attention-measure implementation profile")
    blocks = merge_exact_k_blocks(normalized, block_size, existing_sink)
    range_digest = hashlib.sha256(f"{blocks[0]}:{blocks[1]}".encode("ascii")).hexdigest()
    return BoundMeasurePlan(semantic_digest(normalized), implementation_profile, str(owner_generation),
                            str(numerical_route), q_rows, kv_rows, blocks, range_digest,
                            materialize(normalized, device=device))


def _mask(mask, *, batch, q_rows, kv_rows, dtype, device):
    if mask.device != device:
        raise ValueError("attention measure mask is on the wrong device")
    if mask.ndim == 2 and tuple(mask.shape) == (q_rows, kv_rows):
        mask = mask[None, None]
    elif mask.ndim == 2 and tuple(mask.shape) == (batch, kv_rows):
        mask = mask[:, None, None, :]
    elif mask.ndim == 3 and tuple(mask.shape) == (batch, q_rows, kv_rows):
        mask = mask[:, None]
    elif mask.ndim != 4:
        raise ValueError("unsupported attention mask shape for key-measure composition")
    if mask.dtype == torch.bool:
        return torch.zeros(mask.shape, dtype=dtype, device=device).masked_fill(~mask, float("-inf"))
    if not mask.dtype.is_floating_point:
        raise TypeError("attention mask must be boolean or floating point")
    return mask.to(dtype=dtype)


def weighted_dense(q, k, v, heads, *, key_bias, mask=None, attn_precision=None,
                   skip_reshape=False, skip_output_reshape=False, scale=None):
    if skip_reshape:
        if q.ndim != 4 or q.shape[1] != heads:
            raise ValueError("weighted dense skip_reshape expects BHND Q/K/V")
        qh, kh, vh = q, k, v
        batch, _, q_rows, dim_head = qh.shape
    else:
        batch, q_rows, inner = q.shape
        if inner % heads:
            raise ValueError("weighted dense head geometry is invalid")
        dim_head = inner // heads
        qh = q.view(batch, q_rows, heads, dim_head).transpose(1, 2)
        kh = k.view(batch, k.shape[1], heads, dim_head).transpose(1, 2)
        vh = v.view(batch, v.shape[1], heads, dim_head).transpose(1, 2)
    kv_rows = kh.shape[2]
    if kh.shape != vh.shape or qh.shape[-1] != kh.shape[-1] or key_bias.ndim != 1 or key_bias.shape[0] != kv_rows:
        raise ValueError("weighted dense Q/K/V or key-bias geometry is inconsistent")
    dtype = torch.float32 if attn_precision == torch.float32 else qh.dtype
    bias = key_bias.to(device=q.device, dtype=dtype).view(1, 1, 1, kv_rows)
    if mask is not None:
        bias = _mask(mask, batch=batch, q_rows=q_rows, kv_rows=kv_rows, dtype=dtype, device=q.device) + bias
    out = F.scaled_dot_product_attention(qh.to(dtype), kh.to(dtype), vh.to(dtype), attn_mask=bias, scale=scale).to(v.dtype)
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(batch, q_rows, heads * dim_head)


def register_capability(transformer_options, provider_identity, capability):
    current = transformer_options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
    registry = dict(current) if isinstance(current, Mapping) else {}
    existing = registry.get(provider_identity)
    if existing is not None and existing is not capability:
        raise RuntimeError(f"attention-measure provider {provider_identity!r} already has another owner")
    registry[provider_identity] = capability
    transformer_options[ATTENTION_MEASURE_CAPABILITIES_KEY] = registry
