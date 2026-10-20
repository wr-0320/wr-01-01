"""Deterministic perfect-matching utilities for the frozen v3 graph space."""

from __future__ import annotations

from typing import TypeAlias


Graph: TypeAlias = tuple[tuple[int, int], ...]


def canonical_graph(edges) -> Graph:
    return tuple(sorted((min(a, b), max(a, b)) for a, b in edges))


def enumerate_matchings(nodes: tuple[int, ...] = tuple(range(8))) -> tuple[Graph, ...]:
    """Enumerate perfect matchings in deterministic lexicographic order."""
    if len(nodes) % 2:
        raise ValueError("a perfect matching requires an even number of nodes")
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique")

    def recurse(remaining: tuple[int, ...]):
        if not remaining:
            yield ()
            return
        first = remaining[0]
        for index in range(1, len(remaining)):
            second = remaining[index]
            rest = remaining[1:index] + remaining[index + 1 :]
            for suffix in recurse(rest):
                yield canonical_graph(((first, second), *suffix))

    return tuple(sorted(set(recurse(tuple(sorted(nodes))))))


def graph_id(graph: Graph) -> str:
    return "|".join(f"{a}-{b}" for a, b in canonical_graph(graph))


def edge_distance(graph: Graph, truth: Graph) -> float:
    graph_edges = set(canonical_graph(graph))
    truth_edges = set(canonical_graph(truth))
    if len(graph_edges) != len(truth_edges):
        raise ValueError("graphs must have the same edge count")
    return 1.0 - len(graph_edges & truth_edges) / len(truth_edges)

