"""Unit tests for cache-signature sanitization and hash conversion hardening."""

import asyncio
import importlib
import sys
import types

import pytest


class _DummyNode:
    """Minimal node stub used to satisfy cache-signature class lookups."""

    @staticmethod
    def INPUT_TYPES():
        """Return a minimal empty input schema for unit tests."""
        return {"required": {}}


class _FakeDynPrompt:
    """Small DynamicPrompt stand-in with only the methods these tests need."""

    def __init__(self, nodes_by_id):
        """Store test nodes by id."""
        self._nodes_by_id = nodes_by_id

    def has_node(self, node_id):
        """Return whether the fake prompt contains the requested node."""
        return node_id in self._nodes_by_id

    def get_node(self, node_id):
        """Return the stored node payload for the requested id."""
        return self._nodes_by_id[node_id]


class _FakeIsChangedCache:
    """Async stub for `is_changed` lookups used by cache-key generation."""

    def __init__(self, values):
        """Store canned `is_changed` responses keyed by node id."""
        self._values = values

    async def get(self, node_id):
        """Return the canned `is_changed` value for a node."""
        return self._values[node_id]


class _OpaqueValue:
    """Hashable opaque object used to exercise fail-closed unordered hashing paths."""


def _contains_unhashable(value, unhashable_type):
    """Return whether a nested built-in structure contains an Unhashable sentinel."""
    if isinstance(value, unhashable_type):
        return True

    value_type = type(value)
    if value_type is dict:
        return any(
            _contains_unhashable(key, unhashable_type) or _contains_unhashable(item, unhashable_type)
            for key, item in value.items()
        )
    if value_type in (list, tuple, set, frozenset):
        return any(_contains_unhashable(item, unhashable_type) for item in value)
    return False


@pytest.fixture
def caching_module(monkeypatch):
    """Import `comfy_execution.caching` with lightweight stub dependencies."""
    torch_module = types.ModuleType("torch")
    psutil_module = types.ModuleType("psutil")
    nodes_module = types.ModuleType("nodes")
    nodes_module.NODE_CLASS_MAPPINGS = {}
    graph_module = types.ModuleType("comfy_execution.graph")

    class DynamicPrompt:
        """Placeholder graph type so the caching module can import cleanly."""

        pass

    graph_module.DynamicPrompt = DynamicPrompt

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "psutil", psutil_module)
    monkeypatch.setitem(sys.modules, "nodes", nodes_module)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", graph_module)
    monkeypatch.delitem(sys.modules, "comfy_execution.caching", raising=False)

    module = importlib.import_module("comfy_execution.caching")
    module = importlib.reload(module)
    return module, nodes_module


def test_sanitize_signature_input_handles_shared_builtin_substructures(caching_module):
    """Shared built-in substructures should sanitize without collapsing to Unhashable."""
    caching, _ = caching_module
    shared = [{"value": 1}, {"value": 2}]

    sanitized = caching._sanitize_signature_input([shared, shared])

    assert isinstance(sanitized, list)
    assert sanitized[0] == sanitized[1]
    assert sanitized[0][0]["value"] == 1
    assert sanitized[0][1]["value"] == 2


def test_sanitize_signature_input_snapshots_list_before_recursing(caching_module, monkeypatch):
    """List sanitization should read a point-in-time snapshot before recursive descent."""
    caching, _ = caching_module
    original = caching._sanitize_signature_input
    marker = object()
    values = [marker, 2]

    def mutating_sanitize(obj, *args, **kwargs):
        """Mutate the live list during recursion to verify snapshot-based traversal."""
        if obj is marker:
            values[1] = 3
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_sanitize_signature_input", mutating_sanitize)

    sanitized = original(values)

    assert isinstance(sanitized, list)
    assert sanitized[1] == 2


def test_sanitize_signature_input_snapshots_dict_before_recursing(caching_module, monkeypatch):
    """Dict sanitization should read a point-in-time snapshot before recursive descent."""
    caching, _ = caching_module
    original = caching._sanitize_signature_input
    marker = object()
    values = {"first": marker, "second": 2}

    def mutating_sanitize(obj, *args, **kwargs):
        """Mutate the live dict during recursion to verify snapshot-based traversal."""
        if obj is marker:
            values["second"] = 3
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_sanitize_signature_input", mutating_sanitize)

    sanitized = original(values)

    assert isinstance(sanitized, dict)
    assert sanitized["second"] == 2


@pytest.mark.parametrize(
    "container_factory",
    [
        lambda marker: [marker],
        lambda marker: (marker,),
        lambda marker: {marker},
        lambda marker: frozenset({marker}),
        lambda marker: {marker: "value"},
    ],
)
def test_sanitize_signature_input_fails_closed_on_runtimeerror(caching_module, monkeypatch, container_factory):
    """Traversal RuntimeError should degrade sanitization to Unhashable."""
    caching, _ = caching_module
    original = caching._sanitize_signature_input
    marker = object()

    def raising_sanitize(obj, *args, **kwargs):
        """Raise a traversal RuntimeError for the marker value and delegate otherwise."""
        if obj is marker:
            raise RuntimeError("container changed during iteration")
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_sanitize_signature_input", raising_sanitize)

    sanitized = original(container_factory(marker))

    assert isinstance(sanitized, caching.Unhashable)


def test_to_hashable_handles_shared_builtin_substructures(caching_module):
    """Repeated sanitized content should hash stably for shared substructures."""
    caching, _ = caching_module
    shared = [{"value": 1}, {"value": 2}]

    sanitized = caching._sanitize_signature_input([shared, shared])
    hashable = caching.to_hashable(sanitized)

    assert hashable[0] == "list"
    assert hashable[1][0] == hashable[1][1]
    assert hashable[1][0][0] == "list"


@pytest.mark.parametrize(
    "container_factory",
    [
        set,
        frozenset,
    ],
)
def test_to_hashable_fails_closed_on_runtimeerror(caching_module, monkeypatch, container_factory):
    """Traversal RuntimeError should degrade unordered hash conversion to Unhashable."""
    caching, _ = caching_module

    def raising_sort_key(obj, *args, **kwargs):
        """Raise a traversal RuntimeError while unordered values are canonicalized."""
        raise RuntimeError("container changed during iteration")

    monkeypatch.setattr(caching, "_sanitized_sort_key", raising_sort_key)

    hashable = caching.to_hashable(container_factory({"value"}))

    assert isinstance(hashable, caching.Unhashable)


def test_sanitize_signature_input_fails_closed_for_ambiguous_dict_ordering(caching_module):
    """Ambiguous dict sort ties should fail closed instead of depending on input order."""
    caching, _ = caching_module
    ambiguous = {
        _OpaqueValue(): _OpaqueValue(),
        _OpaqueValue(): _OpaqueValue(),
    }

    sanitized = caching._sanitize_signature_input(ambiguous)

    assert isinstance(sanitized, caching.Unhashable)


@pytest.mark.parametrize(
    "container_factory",
    [
        set,
        frozenset,
    ],
)
def test_to_hashable_fails_closed_for_ambiguous_unordered_values(caching_module, container_factory):
    """Ambiguous unordered values should fail closed instead of depending on iteration order."""
    caching, _ = caching_module
    container = container_factory({_OpaqueValue(), _OpaqueValue()})

    hashable = caching.to_hashable(container)

    assert isinstance(hashable, caching.Unhashable)


def test_get_node_signature_sanitizes_full_signature(caching_module, monkeypatch):
    """Recursive `is_changed` payloads should be sanitized inside the full node signature."""
    caching, nodes_module = caching_module
    monkeypatch.setitem(nodes_module.NODE_CLASS_MAPPINGS, "UnitTestNode", _DummyNode)

    is_changed_value = []
    is_changed_value.append(is_changed_value)

    dynprompt = _FakeDynPrompt(
        {
            "node": {
                "class_type": "UnitTestNode",
                "inputs": {"value": 5},
            }
        }
    )
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        ["node"],
        _FakeIsChangedCache({"node": is_changed_value}),
    )

    signature = asyncio.run(key_set.get_node_signature(dynprompt, "node"))

    assert signature[0] == "list"
    assert _contains_unhashable(signature, caching.Unhashable)
