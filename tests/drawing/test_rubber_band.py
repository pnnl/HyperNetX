"""Tests for convex and concave rubber-band drawing."""

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.patches import PathPatch
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from hypernetx import Hypergraph
from hypernetx.drawing.rubber_band import create_rubber_band, draw, get_concave_hull


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close figures created while drawing."""
    yield
    plt.close("all")


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    """Return polygon components of a shapely geometry.

    Parameters
    ----------
    geometry : BaseGeometry
        Polygon or multipolygon produced by a rubber band.

    Returns
    -------
    list of Polygon
        Polygon parts. A single polygon is returned as a one-item list.
    """
    if isinstance(geometry, Polygon):
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return []


def _u_shaped_points() -> np.ndarray:
    """Return a point set with a wide rectangular bay.

    Local spacing is 1, so an alpha of 0.5 keeps those triangles
    (circumradius about 0.7) and drops triangles that span the bay.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2).
    """
    points = []
    for x in (0, 1):
        for y in range(4):
            points.append((x, y))
    for x in (2, 3, 4):
        for y in (2, 3):
            points.append((x, y))
    for x in (5, 6):
        for y in range(4):
            points.append((x, y))
    return np.asarray(points, dtype=float)


def test_get_concave_hull_rejects_bad_shape() -> None:
    """Reject point arrays that are not two-dimensional coordinates."""
    with pytest.raises(ValueError, match="shape"):
        get_concave_hull(np.ones((4, 3)))


def test_nonpositive_alpha_is_the_convex_hull() -> None:
    """An alpha of zero returns the convex hull."""
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    hull = get_concave_hull(square, alpha=0.0)
    assert hull.area == pytest.approx(1.0)


def test_alpha_shape_is_more_concave_than_the_convex_hull() -> None:
    """A wide bay is excluded from the alpha shape and kept by the convex hull."""
    points = _u_shaped_points()
    concave = get_concave_hull(points, alpha=0.5)
    convex = get_concave_hull(points, alpha=0.0)
    assert concave.area < convex.area


def test_rubber_band_covers_members_and_cuts_out_interior_nodes() -> None:
    """Member corners stay covered and a node in the middle is removed."""
    positions = {
        "a": (0.0, 0.0),
        "b": (2.0, 0.0),
        "c": (2.0, 2.0),
        "d": (0.0, 2.0),
        "outsider": (1.0, 1.0),
    }
    radii = {node: 0.05 for node in positions}
    geometry = create_rubber_band(
        positions,
        ["a", "b", "c", "d"],
        buffer_distance=0.1,
        alpha=0.0,
        other_nodes=["outsider"],
        node_radius=radii,
    )
    for node in ("a", "b", "c", "d"):
        assert geometry.contains(Point(positions[node]))
    assert not geometry.covers(Point(positions["outsider"]))
    assert any(len(polygon.interiors) > 0 for polygon in _polygons(geometry))


def test_rubber_band_is_deterministic() -> None:
    """Repeated calls with the same arguments return the same geometry."""
    positions = {"a": (0.0, 0.0), "b": (1.0, 0.2), "c": (0.2, 1.0), "d": (0.4, 0.4)}
    radii = {node: 0.02 for node in positions}
    first = create_rubber_band(
        positions,
        ["a", "b", "c"],
        buffer_distance=0.15,
        alpha=1.0,
        other_nodes=["d"],
        node_radius=radii,
    )
    second = create_rubber_band(
        positions,
        ["a", "b", "c"],
        buffer_distance=0.15,
        alpha=1.0,
        other_nodes=["d"],
        node_radius=radii,
    )
    assert first.equals(second)


def test_one_and_two_node_edges_cover_their_nodes() -> None:
    """Edges with fewer than three nodes become a disk or a capsule."""
    positions = {"a": (0.0, 0.0), "b": (1.0, 0.0)}
    disk = create_rubber_band(positions, ["a"], buffer_distance=0.2, other_nodes=[])
    capsule = create_rubber_band(
        positions, ["a", "b"], buffer_distance=0.2, other_nodes=[]
    )
    assert disk.contains(Point(positions["a"]))
    assert capsule.contains(Point(positions["a"]))
    assert capsule.contains(Point(positions["b"]))
    assert create_rubber_band(positions, []).is_empty


def test_draw_convex_uses_polycollection() -> None:
    """The default drawing path keeps the convex-hull collection."""
    hypergraph = Hypergraph({"e1": {0, 1, 2}, "e2": {1, 2, 3}})
    positions = {
        0: (0.0, 0.0),
        1: (1.0, 0.2),
        2: (1.0, 1.0),
        3: (2.0, 0.4),
        "e1": (0.7, 0.4),
        "e2": (1.3, 0.5),
    }
    _, axis = plt.subplots()
    returned = draw(
        hypergraph,
        pos=positions,
        ax=axis,
        with_edge_labels=True,
        with_node_labels=True,
        return_pos=True,
    )
    assert returned == positions
    assert any(isinstance(artist, PolyCollection) for artist in axis.collections)
    assert not any(isinstance(artist, PathPatch) for artist in axis.patches)


def test_draw_concave_cuts_layout_nodes_and_labels() -> None:
    """Concave drawing adds one patch per edge and can place labels inside."""
    hypergraph = Hypergraph({"e1": {0, 1, 2, 3}, "e2": {2, 3, 4}})
    positions = {
        0: (0.0, 0.0),
        1: (1.0, 0.1),
        2: (1.1, 1.0),
        3: (0.1, 1.0),
        4: (2.0, 0.4),
        "e1": (0.5, 0.5),
        "e2": (1.4, 0.6),
    }
    _, axis = plt.subplots()
    draw(
        hypergraph,
        pos=positions,
        ax=axis,
        convex=False,
        concavity=0.5,
        with_edge_labels=True,
        with_node_labels=True,
        edge_labels_on_edge=False,
        fill_edges=True,
    )
    patches = [artist for artist in axis.patches if isinstance(artist, PathPatch)]
    assert len(patches) == len(hypergraph.edges)
    assert len(axis.texts) >= len(hypergraph.edges) + len(hypergraph.nodes)


def test_draw_concave_one_node_and_layout() -> None:
    """A one-node edge and a computed layout both draw without error."""
    single = Hypergraph({"e": [1]})
    _, axis = plt.subplots()
    draw(
        single,
        pos={1: (0.0, 0.0), "e": (0.2, 0.0)},
        ax=axis,
        convex=False,
        with_edge_labels=False,
        with_node_labels=False,
    )
    assert any(isinstance(artist, PathPatch) for artist in axis.patches)

    hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4)})
    _, layout_axis = plt.subplots()
    positions = draw(
        hypergraph,
        ax=layout_axis,
        convex=False,
        concavity=4.0,
        return_pos=True,
        with_edge_labels=False,
        with_node_labels=False,
    )
    assert set(hypergraph.nodes).issubset(positions)
    assert layout_axis.patches
