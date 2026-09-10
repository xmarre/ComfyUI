from __future__ import annotations

import inspect

import torch

from comfy.attention_measure import (
    ATTENTION_MEASURE_KEY,
    BoundMeasurePlan,
    MeasureExecutionContext,
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
    return "key_bias" in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


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
    block_index,
    q_rows,
    kv_rows,
    device,
    dtype,
    head_dim,
    mask_class,
    existing_sink,
    provider,
    profile,
    numerical_route,
    preprocess_digest,
) -> BoundMeasurePlan | None:
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
    context = MeasureExecutionContext(
        provider_identity=PROVIDER_IDENTITY,
        block_index=int(block_index),
        owner=patch,
        owner_generation=patch.measure_owner_generation,
        layout=transformer_options.get("minimax_h3_layout"),
        q_rows=int(q_rows),
        kv_rows=int(kv_rows),
        dtype=dtype,
        device=torch.device(device),
        head_dim=int(head_dim),
        mask_class=str(mask_class),
        preprocess_digest=str(preprocess_digest),
        numerical_route=str(numerical_route),
        existing_sink=tuple(existing_sink),
        external_sequence=transformer_options.get("vdn_h3_external_sequence_v1"),
    )
    key = (
        digest,
        context.block_index,
        context.q_rows,
        context.kv_rows,
        str(context.device),
        str(context.dtype),
        context.head_dim,
        context.mask_class,
        context.existing_sink,
        profile,
        context.numerical_route,
        context.preprocess_digest,
        context.owner_generation,
    )
    cache = _cache(patch)
    result = cache.get(key)
    if result is None:
        result = bind(
            request,
            context=context,
            block_size=64,
            implementation_profile=profile,
        )
        cache[key] = result
    return result


def dense_call(
    patch,
    transformer_options,
    q,
    k,
    v,
    heads,
    *,
    mask,
    attn_precision,
    skip_reshape,
    skip_output_reshape,
    scale,
    block_index,
):
    q_rows = q.shape[2] if skip_reshape else q.shape[1]
    kv_rows = k.shape[2] if skip_reshape else k.shape[1]
    sink, _ = patch.sinks(transformer_options, q_rows)
    plan = prepare(
        patch,
        transformer_options,
        block_index=block_index,
        q_rows=q_rows,
        kv_rows=kv_rows,
        device=q.device,
        dtype=q.dtype,
        head_dim=q.shape[-1] if skip_reshape else q.shape[-1] // heads,
        mask_class="none" if mask is None else ("boolean" if mask.dtype == torch.bool else "additive"),
        existing_sink=sink,
        provider=weighted_dense,
        profile="dense_exact_v1",
        numerical_route="core_dense_sdpa",
        preprocess_digest="caller_attention_domain_v1",
    )
    if plan is None:
        return None
    return weighted_dense(
        q,
        k,
        v,
        heads,
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
        if not isinstance(execution_context, MeasureExecutionContext):
            raise TypeError("core sparse attention requires MeasureExecutionContext")
        if execution_context.owner is not self.patch:
            raise RuntimeError("core sparse attention measure capability is bound to another patch owner")
        return bind(
            request,
            context=execution_context,
            block_size=64,
            implementation_profile=(
                "weighted_exact_blocks_v1"
                if execution_context.numerical_route.startswith("core_bsa")
                else "dense_exact_v1"
            ),
        )


def register(patch, transformer_options):
    if not hasattr(patch, "measure_capability"):
        patch.measure_capability = Capability(patch)
    register_capability(transformer_options, PROVIDER_IDENTITY, patch.measure_capability)
