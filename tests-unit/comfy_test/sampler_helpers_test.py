import comfy.hooks
import comfy.sampler_helpers as sampler_helpers


class _FakeControl:
    def __init__(self, extra_hooks=None):
        self._extra_hooks = extra_hooks or []

    def get_extra_hooks(self):
        return self._extra_hooks


class TestSamplerHelpers:
    def test_transformer_options_hooks_do_not_require_non_dynamic_patcher(self):
        hooks = comfy.hooks.HookGroup()
        hooks.add(comfy.hooks.TransformerOptionsHook())
        cond = [(None, {"hooks": hooks})]

        assert sampler_helpers.cond_requires_non_dynamic_patcher(cond) is False

    def test_additional_model_hooks_do_not_require_non_dynamic_patcher(self):
        hooks = comfy.hooks.HookGroup()
        hooks.add(comfy.hooks.AdditionalModelsHook())
        cond = [(None, {"control": _FakeControl(extra_hooks=[hooks])})]

        assert sampler_helpers.cond_requires_non_dynamic_patcher(cond) is False

    def test_weight_hooks_require_non_dynamic_patcher(self):
        hooks = comfy.hooks.HookGroup()
        hooks.add(comfy.hooks.WeightHook())
        cond = [(None, {"hooks": hooks})]

        assert sampler_helpers.cond_requires_non_dynamic_patcher(cond) is True

    def test_control_extra_weight_hooks_require_non_dynamic_patcher(self):
        hooks = comfy.hooks.HookGroup()
        hooks.add(comfy.hooks.WeightHook())
        cond = [(None, {"control": _FakeControl(extra_hooks=[hooks])})]

        assert sampler_helpers.cond_requires_non_dynamic_patcher(cond) is True
