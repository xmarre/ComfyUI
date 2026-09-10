from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from comfy import attention_measure as m


def request(*, equal=False):
    source = [2, 3]
    prefix = source if equal else [3, 4]
    source_rows = 6
    prefix_rows = 6 if equal else 12
    rows = 6 + prefix_rows + source_rows
    ratio = (1, 1) if equal else (1, 2)
    return {
        "api": 1,
        "operator": "key_log_measure",
        "normalization": "h3_native_source_carrier_v1",
        "topology": "mixed_grid_low_suffix",
        "q_rows": rows,
        "kv_rows": rows,
        "video_start": 6,
        "temporal": 2,
        "prefix_t": 1,
        "source_grid": source,
        "prefix_grid": prefix,
        "segments": [
            {"start": 0, "stop": 6, "mass_num": 1, "mass_den": 1},
            {"start": 6, "stop": 6 + prefix_rows, "mass_num": ratio[0], "mass_den": ratio[1]},
            {"start": 6 + prefix_rows, "stop": rows, "mass_num": 1, "mass_den": 1},
        ],
        "coordinate_policy": "minimax_h3_native_frame_grid_v1",
    }


def layout(rows):
    return SimpleNamespace(seq_len=rows, segments=((0, 6, "text"), (6, rows, "video")))


def context(r):
    return m.MeasureExecutionContext(
        provider_identity="test.provider",
        block_index=3,
        owner=object(),
        owner_generation="gen-1",
        layout=layout(r["q_rows"]),
        q_rows=r["q_rows"],
        kv_rows=r["kv_rows"],
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        head_dim=128,
        mask_class="none",
        preprocess_digest="pre-1",
        numerical_route="test",
    )


def test_normalize_is_semantic_not_segment_spelling():
    r = request()
    canonical = m.normalize(r)
    split = {**r, "segments": [
        {"start": 0, "stop": 2, "mass_num": 4, "mass_den": 4},
        {"start": 2, "stop": 6, "mass_num": 1, "mass_den": 1},
        {"start": 6, "stop": 12, "mass_num": 2, "mass_den": 4},
        {"start": 12, "stop": 18, "mass_num": 1, "mass_den": 2},
        {"start": 18, "stop": 24, "mass_num": 3, "mass_den": 3},
    ]}
    assert m.normalize(split) == canonical
    assert m.semantic_digest(split) == m.semantic_digest(r)


def test_equal_grid_is_unit_measure_and_preserves_existing_sink():
    r = request(equal=True)
    normalized = m.normalize(r)
    assert normalized["segments"] == [{"start": 0, "stop": 18, "mass_num": 1, "mass_den": 1}]
    assert m.is_unit_measure(r)
    assert m.merge_exact_k_blocks(r, 64, existing=(0, 1)) == (0, 1)
    assert torch.equal(m.materialize(r, device="cpu"), torch.zeros(18))


def test_ragged_weighted_interval_expands_to_physical_k_blocks():
    r = {
        "api": 1,
        "operator": "key_log_measure",
        "normalization": "h3_native_source_carrier_v1",
        "topology": "mixed_grid_low_suffix",
        "q_rows": 303,
        "kv_rows": 303,
        "video_start": 63,
        "temporal": 3,
        "prefix_t": 1,
        "source_grid": [8, 8],
        "prefix_grid": [8, 14],
        "segments": [
            {"start": 0, "stop": 63, "mass_num": 1, "mass_den": 1},
            {"start": 63, "stop": 175, "mass_num": 4, "mass_den": 7},
            {"start": 175, "stop": 303, "mass_num": 1, "mass_den": 1},
        ],
        "coordinate_policy": "minimax_h3_native_frame_grid_v1",
    }
    assert m.merge_exact_k_blocks(r, 64) == (0, 3)
    assert m.merge_exact_k_blocks(r, 64, existing=(0, 1)) == (0, 3)


def test_h3_validation_cross_checks_actual_layout_and_vdn_when_present():
    r = request()
    rows = r["q_rows"]
    external = {
        "api": 2,
        "mode": "dense_gate_no_linear",
        "topology": "mixed_grid_low_suffix",
        "sequence_rows": rows,
        "video_start": 6,
        "temporal": 2,
        "prefix_t": 1,
        "source_rows_per_frame": 6,
        "prefix_rows_per_frame": 12,
    }
    assert m.validate_h3(r, layout=layout(rows), q_rows=rows, kv_rows=rows, external_sequence=external) == m.normalize(r)
    with pytest.raises(ValueError):
        m.validate_h3(r, layout=SimpleNamespace(seq_len=rows, segments=((0, rows, "video"),)), q_rows=rows, kv_rows=rows)
    with pytest.raises(ValueError):
        m.validate_h3(r, layout=layout(rows), q_rows=rows, kv_rows=rows, external_sequence={**external, "prefix_t": 2})


def test_bind_and_capability_validate_concrete_owner_identity():
    r = request()
    ctx = context(r)

    class Capability:
        def prepare(self, request, execution_context):
            return m.bind(
                request,
                context=execution_context,
                block_size=64,
                implementation_profile="weighted_exact_blocks_v1",
            )

    options = {}
    cap = Capability()
    m.register_capability(options, "test.provider", cap)
    plan = m.prepare_capability(options, r, ctx)
    assert plan.semantic_digest == m.semantic_digest(r)
    assert plan.exact_k_block_range == (0, 1)
    assert plan.provider_identity == "test.provider"
    with pytest.raises(RuntimeError):
        m.prepare_capability(options, r, replace(ctx, provider_identity="other"))


def test_weighted_dense_sdpa_and_streaming_match_and_preserve_masks():
    torch.manual_seed(12)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 24, 8)
    v = torch.randn(1, 2, 24, 8)
    bias = m.materialize(request(), device="cpu")
    mask = torch.ones(1, 1, 5, 24, dtype=torch.bool)
    mask[..., 1, :] = False
    mask[..., 3, 17:] = False
    got = m.weighted_dense(q, k, v, 2, key_bias=bias, mask=mask, skip_reshape=True, skip_output_reshape=True)
    streamed = m.weighted_dense_streaming(
        q, k, v, 2, key_bias=bias, mask=mask, skip_reshape=True, skip_output_reshape=True, key_chunk_size=7
    )
    torch.testing.assert_close(streamed, got, rtol=2e-5, atol=2e-5)
    assert torch.count_nonzero(got[:, :, 1]) == 0


def test_malformed_values_and_additive_nan_rejected():
    r = request()
    with pytest.raises(TypeError):
        m.normalize({**r, "q_rows": True})
    bad = {**r, "segments": [dict(x) for x in r["segments"]]}
    bad["segments"][1]["mass_den"] = False
    with pytest.raises(TypeError):
        m.normalize(bad)
    q = torch.zeros(1, 1, 1, 8)
    k = torch.zeros(1, 1, 24, 8)
    v = torch.zeros_like(k)
    bias = m.materialize(r, device="cpu")
    mask = torch.zeros(1, 1, 1, 24)
    mask[..., 0] = float("nan")
    with pytest.raises(ValueError):
        m.weighted_dense(q, k, v, 1, key_bias=bias, mask=mask, skip_reshape=True)
