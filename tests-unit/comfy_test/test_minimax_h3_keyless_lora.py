from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.lora as lora  # noqa: E402


class _ExplodingStateDictModel:
    def __init__(self, contract):
        self.diffusion_model = SimpleNamespace(
            **{lora.KEYLESS_H3_CONTRACT_KEY: contract}
        )

    def state_dict(self):
        raise AssertionError("generic Keyless LoRA guard must run before state_dict mapping")


def _contract(**overrides):
    values = {
        "api": 1,
        "architecture": "h3_keyless_core50_v1",
        "core_blocks": 50,
        "token_refiner": "native_qkv",
        "token_refiner_blocks": 2,
        "routing_source": "value",
        "retrieval_source": "raw_projected_value",
        "projection_attr": "qv_proj",
        "qv_order": "q_effective;v",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_generic_lora_key_map_rejects_keyless_before_state_dict_mapping():
    model = _ExplodingStateDictModel(_contract())

    with pytest.raises(ValueError, match="Generic LoRA/DoRA loading is not supported"):
        lora.model_lora_keys_unet(model, {})


def test_generic_lora_guard_rejects_malformed_keyless_contract():
    model = _ExplodingStateDictModel(_contract(routing_source="key"))

    with pytest.raises(
        ValueError,
        match="malformed minimax_h3_keyless_contract_v1.*routing_source",
    ):
        lora.model_lora_keys_unet(model, {})


def test_generic_lora_guard_ignores_models_without_keyless_marker():
    model = SimpleNamespace(diffusion_model=SimpleNamespace())

    assert lora._reject_unsupported_keyless_h3_adapter(model) is None
