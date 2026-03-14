from comfy_execution.caching import Unhashable, to_hashable


def test_to_hashable_returns_unhashable_for_cyclic_builtin_containers():
    """Ensure self-referential built-in containers terminate as Unhashable."""
    cyclic_list = []
    cyclic_list.append(cyclic_list)

    result = to_hashable(cyclic_list)

    assert result[0] == "list"
    assert len(result[1]) == 1
    assert isinstance(result[1][0], Unhashable)


def test_to_hashable_returns_unhashable_when_max_depth_is_reached():
    """Ensure deeply nested built-in containers stop at the configured depth limit."""
    nested = current = []
    for _ in range(32):
        next_item = []
        current.append(next_item)
        current = next_item

    result = to_hashable(nested)

    depth = 0
    current = result
    while isinstance(current, tuple):
        assert current[0] == "list"
        assert len(current[1]) == 1
        current = current[1][0]
        depth += 1

    assert depth == 32
    assert isinstance(current, Unhashable)
