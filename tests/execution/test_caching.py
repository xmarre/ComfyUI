import asyncio

from comfy_execution import caching


class _StubDynPrompt:
    def __init__(self, nodes):
        self._nodes = nodes

    def has_node(self, node_id):
        return node_id in self._nodes

    def get_node(self, node_id):
        return self._nodes[node_id]


class _StubIsChangedCache:
    async def get(self, node_id):
        return None


class _StubNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


def test_get_immediate_node_signature_canonicalizes_non_link_inputs(monkeypatch):
    live_value = [1, {"nested": [2, 3]}]
    dynprompt = _StubDynPrompt(
        {
            "1": {
                "class_type": "TestCacheNode",
                "inputs": {"value": live_value},
            }
        }
    )

    monkeypatch.setitem(caching.nodes.NODE_CLASS_MAPPINGS, "TestCacheNode", _StubNode)
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})

    keyset = caching.CacheKeySetInputSignature(dynprompt, [], _StubIsChangedCache())
    signature = asyncio.run(keyset.get_immediate_node_signature(dynprompt, "1", {}))

    assert signature == [
        "TestCacheNode",
        None,
        ("value", ("list", (1, ("dict", (("nested", ("list", (2, 3))),))))),
    ]


def test_get_immediate_node_signature_fails_closed_for_opaque_non_link_input(monkeypatch):
    class OpaqueRuntimeValue:
        pass

    live_value = OpaqueRuntimeValue()
    dynprompt = _StubDynPrompt(
        {
            "1": {
                "class_type": "TestCacheNode",
                "inputs": {"value": live_value},
            }
        }
    )

    monkeypatch.setitem(caching.nodes.NODE_CLASS_MAPPINGS, "TestCacheNode", _StubNode)
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})

    keyset = caching.CacheKeySetInputSignature(dynprompt, [], _StubIsChangedCache())
    signature = asyncio.run(keyset.get_immediate_node_signature(dynprompt, "1", {}))

    assert signature[:2] == ["TestCacheNode", None]
    assert signature[2][0] == "value"
    assert type(signature[2][1]) is caching.Unhashable


def test_get_node_signature_never_visits_raw_non_link_input(monkeypatch):
    live_value = [1, 2, 3]
    dynprompt = _StubDynPrompt(
        {
            "1": {
                "class_type": "TestCacheNode",
                "inputs": {"value": live_value},
            }
        }
    )

    monkeypatch.setitem(caching.nodes.NODE_CLASS_MAPPINGS, "TestCacheNode", _StubNode)
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})

    original_impl = caching._signature_to_hashable_impl

    def guarded_impl(obj, *args, **kwargs):
        if obj is live_value:
            raise AssertionError("raw non-link input reached outer signature canonicalizer")
        return original_impl(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_signature_to_hashable_impl", guarded_impl)

    keyset = caching.CacheKeySetInputSignature(dynprompt, [], _StubIsChangedCache())
    signature = asyncio.run(keyset.get_node_signature(dynprompt, "1"))

    assert isinstance(signature, tuple)
