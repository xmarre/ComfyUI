"""Diagnostic compatibility shim for ComfyUI v0.34.3 + current H3 Flow.

This branch is intentionally a regression-bisect environment. ComfyUI v0.34.3
predates the MiniMax-H3 block-replacement ABI change that began including
``layout`` in every block callback. Current MiniMax-H3-Flow-Aligned-Regenerate
uses that field for layout metrics, so a plain old-core checkout raises
``KeyError: 'layout'`` before the first transformer evaluation.

Do not backport newer H3 numerics into this baseline. Instead, patch only the
Flow layout-metrics factory at import time: when an old-core block callback has
no layout, pass through to the wrapped/original block unchanged. Mixed-Grid's
own high-resolution boundary wrapper explicitly supplies its synthetic layout,
so that path retains its current behavior.

The file is named sitecustomize.py so Python imports it automatically when
ComfyUI is launched from this checkout. It installs a narrowly scoped import
hook and does nothing unless h3_flow_regenerate.attention is imported.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys

_TARGET = "h3_flow_regenerate.attention"
_MARKER = "_comfy_v0343_layout_compat"


def _patch_flow_attention(module) -> None:
    if getattr(module, _MARKER, False):
        return

    def make_layout_block_wrapper(layer, metrics, previous=None, *, record_layout=True):
        def wrapper(args, extra):
            layout = args.get("layout")

            # ComfyUI <= v0.34.3 does not include layout in ordinary H3
            # block-replacement callback arguments. Layout recording is
            # diagnostic-only, so preserve the old-core numerical path exactly.
            if layout is None:
                if previous is not None:
                    return previous(args, extra)
                return extra["original_block"](args)

            transformer = args["transformer_options"]
            old = transformer.get("h3_flow_attention_context")
            transformer["h3_flow_attention_context"] = {"layout": layout, "layer": layer}
            if layer == 0 and record_layout:
                metrics.event("packed_layout", **module.layout_summary(layout))
            try:
                if previous is not None:
                    return previous(args, extra)
                return extra["original_block"](args)
            finally:
                if old is None:
                    transformer.pop("h3_flow_attention_context", None)
                else:
                    transformer["h3_flow_attention_context"] = old

        return wrapper

    module.make_layout_block_wrapper = make_layout_block_wrapper
    setattr(module, _MARKER, True)
    print("[v0.34.3 H3 compat] Flow layout wrapper accepts legacy block callbacks without layout")


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        _patch_flow_attention(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or isinstance(spec.loader, _PatchLoader):
            return spec
        spec.loader = _PatchLoader(spec.loader)
        return spec


sys.meta_path.insert(0, _PatchFinder())
