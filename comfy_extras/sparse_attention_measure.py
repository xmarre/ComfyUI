from __future__ import annotations

import inspect

import torch

from comfy.attention_measure import (
    ATTENTION_MEASURE_KEY,
    bind,
    register_capability,
    semantic_digest,
    weighted_dense,
)

PROVIDER_IDENTITY = "comfy.core.block_sparse_attention"
VDN_EPILOGUE_KEY = "vdn_h3_external_softmax_epilogue_v1"


def _supports_key_bias(provider) -> bool:
    try:
        parameters = inspect.signature(provider).parameters
    except (TypeError, ValueError):
        return False
    return "key_bias" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _cache(patch):
    cache = getattr(patch, "measure_plans", None)
    if cache is None:
        cache = {}
        patch.measure_plans = cache
    return cache


def prepare(
    patch,
    transformer_options,
    *,
    q_rows,
    kv_rows,
    device,
    existing_sink,
    provider,
    profile,
    numerical_route,
):
    request = transformer_options.get(ATTENTION_MEASURE_KEY)
    if request is None:
        return None
    if patch.vsa:
        raise RuntimeError("Mixed-Grid attention measure is incompatible with VSA tile planning")
    if profile == "weighted_exact_blocks_v1" and not _supports_key_bias(provider):
        raise RuntimeError(
            "Mixed-Grid attention measure selected a sparse provider without key_bias support; "
            "install a comfy-kitchen build with weighted sparse attention support"
        )
    digest = semantic_digest(request)
    key = (digest, int(q_rows), int(kv_rows), str(torch.device(device)), tuple(existing_sink), profile, numerical_route)
    cache = _cache(patch)
    result = cache.get(key)
    if result is None:
        result = bind(
            request,
            layout=transformer_options.get("minimax_h3_layout"),
            q_rows=int(q_rows),
            kv_rows=int(kv_rows),
            block_size=64,
            existing_sink=existing_sink,
            device=device,
            implementation_profile=profile,
            owner_generation=patch.measure_owner_generation,
            numerical_route=numerical_route,
            external_sequence=transformer_options.get("vdn_h3_external_sequence_v1"),
        )
        cache[key] = result
    return result


def dense_call(patch, transformer_options, q, k, v, heads, *, mask, attn_precision,
               skip_reshape, skip_output_reshape, scale):
    q_rows = q.shape[2] if skip_reshape else q.shape[1]
    kv_rows = k.shape[2] if skip_reshape else k.shape[1]
    sink, _ = patch.sinks(transformer_options, q_rows)
    plan = prepare(
        patch,
        transformer_options,
        q_rows=q_rows,
        kv_rows=kv_rows,
        device=q.device,
        existing_sink=sink,
        provider=weighted_dense,
        profile="dense_exact_v1",
        numerical_route="core_dense_sdpa",
    )
    if plan is None:
        return None
    return weighted_dense(
        q, k, v, heads,
        key_bias=plan.key_log_measure,
        mask=mask,
        attn_precision=attn_precision,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        scale=scale,
    )


class Capability:
    api = 1
    operator = "key_log_measure"
    provider_identity = PROVIDER_IDENTITY

    def __init__(self, patch):
        self.patch = patch

    def prepare(self, request, execution_context):
        if request is not execution_context.get("transformer_options", {}).get(ATTENTION_MEASURE_KEY):
            raise RuntimeError("attention-measure request is not owned by the current transformer options")
        return prepare(
            self.patch,
            execution_context["transformer_options"],
            q_rows=execution_context["q_rows"],
            kv_rows=execution_context["kv_rows"],
            device=execution_context["device"],
            existing_sink=execution_context.get("existing_sink", (0, 0)),
            provider=execution_context["provider"],
            profile=execution_context["implementation_profile"],
            numerical_route=execution_context["numerical_route"],
        )


def register(patch, transformer_options):
    if not hasattr(patch, "measure_capability"):
        patch.measure_capability = Capability(patch)
    register_capability(transformer_options, PROVIDER_IDENTITY, patch.measure_capability)
