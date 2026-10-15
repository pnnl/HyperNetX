# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import EllipseCollection, PolyCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.spatial import ConvexHull, Delaunay, QhullError
from scipy.spatial.distance import pdist
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from hypernetx.drawing.util import (
    get_set_layering,
    inflate,
    inflate_kwargs,
    transpose_inflated_kwargs,
)

# increases the default figure size to 8in square.
plt.rcParams["figure.figsize"] = (8, 8)

N_CONTROL_POINTS = 24

theta = np.linspace(0, 2 * np.pi, N_CONTROL_POINTS + 1)[:-1]

cp = np.vstack((np.cos(theta), np.sin(theta))).T


def add_edge_defaults(H, edges_kwargs):
    edges_kwargs = edges_kwargs.copy()

    colors = plt.cm.tab10(np.arange(len(H.edges)) % 10)
    edges_kwargs.setdefault("edgecolors", colors)
    edges_kwargs.setdefault("facecolors", "none")

    edges_kwargs.setdefault("linewidth", 1)

    return edges_kwargs


def layout_node_link(H, G=None, layout=nx.spring_layout, **kwargs):
    """
    Helper function to use a NetwrokX-like graph layout algorithm on a Hypergraph

    The hypergraph is converted to a bipartite graph, allowing the usual graph layout
    techniques to be applied.

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    G: Graph
        an additional set of links to consider during the layout process
    layout: function
        the layout algorithm which accepts a NetworkX graph and keyword arguments
    kwargs: dict
        Keyword arguments are passed through to the layout algorithm

    Returns
    -------
    dict
        mapping of node and edge positions to R^2
    """

    B = H.bipartite()

    if G is not None:
        B.add_edges_from(G.edges())

    return layout(B, **kwargs)


def get_default_radius(H, pos):
    """
    Calculate a reasonable default node radius

    This function iterates over the hyper edges and finds the most distant
    pair of points given the positions provided. Then, the node radius is a fraction
    of the median of this distance take across all hyper-edges.

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    pos: dict
        mapping of node and edge positions to R^2

    Returns
    -------
    float
        the recommended radius

    """
    if len(H) > 1:
        return 0.0125 * np.median(
            [pdist(np.vstack(list(map(pos.get, H.nodes)))).max() for nodes in H.edges()]
        )
    return 1


def draw_hyper_edge_labels(
    H, pos, labels, polys, edge_labels_on_edge=True, ax=None, **kwargs
):
    """
    Draws a label on the hyper edge boundary.

    Should be passed Matplotlib PolyCollection representing the hyper-edges, see
    the return value of draw_hyper_edges.

    The label will be draw on the least curvy part of the polygon, and will be
    aligned parallel to the orientation of the polygon where it is drawn.

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    polys: PolyCollection
        collection of polygons returned by draw_hyper_edges
    labels: dict
        mapping of node id to string label
    ax: Axis
        matplotlib axis on which the plot is rendered
    kwargs: dict
        Keyword arguments are passed through to Matplotlib's annotate function.

    """
    ax = ax or plt.gca()

    params = transpose_inflated_kwargs(inflate_kwargs(H.edges, kwargs))

    for edge, s, path, edge_params in zip(H.edges, labels, polys.get_paths(), params):

        theta = 0
        xy = None

        if edge_labels_on_edge:
            # calculate the xy location of the annotation
            # this is the midpoint of the pair of adjacent points the most distant
            d = ((path.vertices[:-1] - path.vertices[1:]) ** 2).sum(axis=1)
            i = d.argmax()

            x1, x2 = path.vertices[i : i + 2]
            x, y = x2 - x1
            theta = 360 * np.arctan2(y, x) / (2 * np.pi)
            theta = (theta + 360) % 360

            while theta > 90:
                theta -= 180

            xy = (x1 + x2) / 2
        else:
            xy = pos[edge]

        # the string is a comma separated list of the edge uid
        ax.annotate(
            s, xy, **{"rotation": theta, "ha": "center", "va": "center", **edge_params}
        )


def layout_hyper_edges(H, pos, node_radius=None, dr=None, contain_hyper_edges=False):
    """
    Draws a convex hull for each edge in H.

    Position of the nodes in the graph is specified by the position dictionary,
    pos. Convex hulls are spaced out such that if one set contains another, the
    convex hull will surround the contained set. The amount of spacing added
    between hulls is specified by the parameter, dr.

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    pos: dict
        mapping of node and edge positions to R^2
    node_radius: dict
        mapping of node to R^1 (radius of each node)
    dr: float
        the spacing between concentric rings
    ax: Axis
        matplotlib axis on which the plot is rendered

    Returns
    -------
    dict
        A mapping from hyper edge ids to paths (Nx2 numpy matrices)
    """

    if node_radius is None:
        node_radius = {}

    if len(node_radius):
        r0 = min(node_radius.values())
    else:
        r0 = get_default_radius(H, pos)

    dr = dr or r0

    levels = get_set_layering(H)

    radii = {
        v: {v: i for i, v in enumerate(sorted(e, key=levels.get))}
        for v, e in H.nodes.memberships.items()
    }

    def get_padded_hull(uid, edge):
        # make sure the edge contains at least one node
        if len(edge):
            points = [
                cp * (node_radius.get(v, r0) + dr * (1 + radii[v][uid])) + pos[v]
                for v in edge
            ]

            if contain_hyper_edges:
                points.append(cp * r0 + pos[uid])

            points = np.vstack(points)

        # if not, draw an empty edge centered around the location of the edge node (in the bipartite graph)
        else:
            points = 4 * r0 * cp + pos[uid]

        hull = ConvexHull(points)

        return hull.points[hull.vertices]

    return [get_padded_hull(uid, list(H.edges[uid])) for uid in H.edges]


def draw_hyper_edges(
    H,
    pos,
    ax=None,
    node_radius=None,
    contain_hyper_edges=False,
    dr=None,
    fill_edges=False,
    fill_edge_alpha=-0.5,
    **kwargs,
):
    """
    Draws a convex hull around the nodes contained within each edge in H

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    pos: dict
        mapping of node and edge positions to R^2
    node_radius: dict
        mapping of node to R^1 (radius of each node)
    dr: float
        the spacing between concentric rings
    ax: Axis
        matplotlib axis on which the plot is rendered
    kwargs: dict
        keyword arguments, e.g., linewidth, facecolors, are passed through to the PolyCollection constructor

    Returns
    -------
    PolyCollection
        a Matplotlib PolyCollection that can be further styled
    """
    if node_radius is None:
        node_radius = {}

    points = layout_hyper_edges(
        H, pos, node_radius=node_radius, dr=dr, contain_hyper_edges=contain_hyper_edges
    )

    polys = PolyCollection(points, **inflate_kwargs(H.edges, kwargs))
    if fill_edges:
        color = polys.get_edgecolors() + np.array([0, 0, 0, fill_edge_alpha])
        polys.set_facecolors(color)

    (ax or plt.gca()).add_collection(polys)

    return polys


def draw_hyper_nodes(H, pos, node_radius=None, r0=None, ax=None, **kwargs):
    """
    Draws a circle for each node in H.

    The position of each node is specified by the a dictionary/list-like, pos,
    where pos[v] is the xy-coordinate for the vertex. The radius of each node
    can be specified as a dictionary where node_radius[v] is the radius. If a
    node is missing from this dictionary, or the node_radius is not specified at
    all, a sensible default radius is chosen based on distances between nodes
    given by pos.

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    pos: dict
        mapping of node and edge positions to R^2
    node_radius: dict
        mapping of node to R^1 (radius of each node)
    r0: float
        minimum distance that concentric rings start from the node position
    ax: Axis
        matplotlib axis on which the plot is rendered
    kwargs: dict
        keyword arguments, e.g., linewidth, facecolors, are passed through to the PolyCollection constructor

    Returns
    -------
    PolyCollection
        a Matplotlib PolyCollection that can be further styled
    """

    ax = ax or plt.gca()
    if node_radius is None:
        node_radius = {}
    kwargs.setdefault("facecolors", "black")

    r0 = r0 or get_default_radius(H, pos)
    offsets = [pos[v] for v in H.nodes]
    sizes = [2 * node_radius.get(v, r0) for v in H.nodes]

    circles = EllipseCollection(
        widths=sizes,
        heights=sizes,
        angles=0,
        units="xy",
        offsets=offsets,
        transOffset=ax.transData,
        **inflate_kwargs(H, kwargs),
    )
    ax.add_collection(circles)

    return circles


def draw_hyper_labels(H, pos, labels, node_radius=None, ax=None, **kwargs):
    """
    Draws text labels for the hypergraph nodes.

    The label is drawn to the right of the node. The node radius is needed (see
    draw_hyper_nodes) so the text can be offset appropriately as the node size
    changes.

    The text label can be customized by passing in a dictionary, labels, mapping
    a node to its custom label. By default, the label is the string
    representation of the node.

    Keyword arguments are passed through to Matplotlib's annotate function.

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    pos: dict
        mapping of node and edge positions to R^2
    node_radius: dict
        mapping of node to R^1 (radius of each node)
    ax: Axis
        matplotlib axis on which the plot is rendered
    labels: dict
        mapping of node to text label
    kwargs: dict
        keyword arguments passed to matplotlib.annotate

    """
    ax = ax or plt.gca()
    if node_radius is None:
        node_radius = {}

    params = transpose_inflated_kwargs(inflate_kwargs(H.nodes, kwargs))

    for v, s, v_kwargs in zip(H.nodes, labels, params):
        xy = np.array([node_radius.get(v, 0), 0]) + pos[v]
        ax.annotate(s, xy, **v_kwargs)


def _circumradius(triangle_points: np.ndarray) -> float:
    """Return the circumradius of a triangle, or infinity if it is degenerate.

    Parameters
    ----------
    triangle_points : np.ndarray
        Array of shape (3, 2).

    Returns
    -------
    float
        Circumradius of the triangle.
    """
    first, second, third = triangle_points
    edge_ab = second - first
    edge_ac = third - first
    twice_area = abs(float(edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0]))
    if twice_area <= 1e-12:
        return float("inf")
    length_a = float(np.linalg.norm(second - third))
    length_b = float(np.linalg.norm(first - third))
    length_c = float(np.linalg.norm(first - second))
    return (length_a * length_b * length_c) / (2.0 * twice_area)


def _convex_hull_geometry(points: np.ndarray) -> BaseGeometry:
    """Return the convex hull of a 2-d point set.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2). Duplicate rows are ignored.

    Returns
    -------
    BaseGeometry
        A point, line, or polygon, depending on how many distinct locations
        the input contains.
    """
    unique = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(unique) == 1:
        return Point(unique[0])
    if len(unique) == 2:
        return LineString(unique)
    try:
        hull = ConvexHull(unique)
    except QhullError:
        return LineString(unique)
    return Polygon(unique[hull.vertices])


def get_concave_hull(points: np.ndarray, alpha: float = 0.5) -> BaseGeometry:
    """Return an alpha shape of ``points``.

    The construction follows Edelsbrunner alpha shapes, the same rule used by
    the ``alphashape`` library. A Delaunay triangle is kept when its
    circumradius is strictly less than ``1 / alpha``. Larger ``alpha`` drops
    more triangles and produces a more concave outline. ``alpha <= 0`` returns
    the convex hull.

    ``alpha`` is absolute in the coordinate system of ``points``. Layouts that
    place nodes in a unit box need a larger value before concavities appear.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2).
    alpha : float
        Alpha-shape parameter. Values less than or equal to zero select the
        convex hull.

    Returns
    -------
    BaseGeometry
        The hull. Falls back to the convex hull when the triangulation is
        degenerate or no triangle satisfies the radius test.

    Raises
    ------
    ValueError
        If ``points`` does not have shape (n, 2).
    """
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("points must be an array of shape (n, 2)")
    unique = np.unique(coordinates, axis=0)
    if len(unique) < 4 or alpha <= 0:
        return _convex_hull_geometry(unique)
    try:
        triangulation = Delaunay(unique)
    except QhullError:
        return _convex_hull_geometry(unique)

    radius_limit = 1.0 / alpha
    kept: list[Polygon] = []
    for simplex in triangulation.simplices:
        triangle_points = unique[simplex]
        if _circumradius(triangle_points) < radius_limit:
            triangle = Polygon(triangle_points)
            if triangle.is_valid and triangle.area > 0:
                kept.append(triangle)
    if not kept:
        return _convex_hull_geometry(unique)
    hull = unary_union(kept)
    if not hull.is_valid:
        hull = hull.buffer(0)
    return hull


def _as_dicts(*values: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """Replace omitted mappings with fresh dictionaries.

    Parameters
    ----------
    values : Mapping or None
        Keyword-argument dictionaries supplied by a caller.

    Returns
    -------
    tuple of dict
        ``{}`` wherever a value was ``None``, otherwise the original mapping.
    """
    return tuple({} if value is None else dict(value) for value in values)


def _with_padding(geom: BaseGeometry, pad: float) -> BaseGeometry:
    """Expand ``geom`` by ``pad``.

    Points and lines become polygons. Invalid polygons are repaired.

    Parameters
    ----------
    geom : BaseGeometry
        Geometry to expand.
    pad : float
        Buffer distance in the geometry's coordinate system.

    Returns
    -------
    BaseGeometry
        Buffered geometry.
    """
    if geom.is_empty:
        return geom
    if geom.geom_type in {"Point", "MultiPoint", "LineString", "MultiLineString"}:
        return geom.buffer(pad)
    padded = geom.buffer(pad)
    if not padded.is_valid:
        padded = padded.buffer(0)
    return padded


def _outline_padding(
    edge_points: np.ndarray,
    member_radii: np.ndarray,
    buffer_distance: float,
) -> float:
    """Return the distance used to expand one edge outline.

    Parameters
    ----------
    edge_points : np.ndarray
        Member coordinates, shape (n, 2).
    member_radii : np.ndarray
        Marker radius of each member.
    buffer_distance : float
        Fraction of the mean pairwise distance, or the distance itself when
        the edge has one node.

    Returns
    -------
    float
        Positive padding distance.
    """
    if len(edge_points) > 1:
        adaptive = buffer_distance * float(np.mean(pdist(edge_points)))
    else:
        adaptive = buffer_distance
    radius_floor = float(np.max(member_radii)) if len(member_radii) else 0.0
    return max(adaptive, radius_floor, 1e-9)


def _member_hull(edge_points: np.ndarray, alpha: float) -> BaseGeometry:
    """Return the unpadded outline of an edge's member coordinates.

    Parameters
    ----------
    edge_points : np.ndarray
        Member coordinates, shape (n, 2).
    alpha : float
        Alpha-shape parameter used when there are at least three points.

    Returns
    -------
    BaseGeometry
        A point, line, or polygon.
    """
    if len(edge_points) == 1:
        return Point(edge_points[0])
    if len(edge_points) == 2:
        return LineString(edge_points)
    return get_concave_hull(edge_points, alpha)


def _cut_out_nodes(
    geom: BaseGeometry,
    outsiders: Iterable[Any],
    nodes_pos: Mapping[Any, Sequence[float]],
    radii: Mapping[Any, float],
    pad: float,
) -> BaseGeometry:
    """Remove disks around non-members that fall inside ``geom``.

    Parameters
    ----------
    geom : BaseGeometry
        Padded edge outline.
    outsiders : Iterable
        Node ids that must not be covered.
    nodes_pos : Mapping
        Node id to position.
    radii : Mapping
        Node id to marker radius.
    pad : float
        Minimum radius of each removed disk.

    Returns
    -------
    BaseGeometry
        Outline with those disks removed.
    """
    holes: list[BaseGeometry] = []
    for node in outsiders:
        if node not in nodes_pos:
            continue
        point = Point(nodes_pos[node])
        if not geom.contains(point):
            continue
        holes.append(point.buffer(max(float(radii.get(node, 0.0)), pad)))
    if not holes:
        return geom
    cut = geom.difference(unary_union(holes))
    if not cut.is_valid:
        cut = cut.buffer(0)
    return cut


def create_rubber_band(
    nodes_pos: Mapping[Any, Sequence[float]],
    edge_nodes: Iterable[Any],
    buffer_distance: float = 0.1,
    alpha: float = 0.5,
    other_nodes: Iterable[Any] | None = None,
    node_radius: Mapping[Any, float] | None = None,
) -> BaseGeometry:
    """Build a deterministic outline around the nodes of one hyperedge.

    The outline is an alpha shape of the member nodes, expanded so node
    markers sit inside it. Markers for ``other_nodes`` that land inside that
    outline are cut out. The same inputs always produce the same geometry.

    Parameters
    ----------
    nodes_pos : Mapping
        Node id to ``(x, y)`` position. Only ids that are drawn as hypergraph
        nodes should be included in ``other_nodes``; bipartite edge anchors
        stored in a layout dictionary are not nodes.
    edge_nodes : Iterable
        Nodes that belong to the hyperedge.
    buffer_distance : float
        For two or more nodes, fraction of their mean pairwise distance used
        as padding. For a single node, the padding distance itself.
    alpha : float
        Alpha-shape parameter forwarded to :func:`get_concave_hull`.
    other_nodes : Iterable or None
        Nodes that must not be covered. ``None`` uses every key of
        ``nodes_pos`` that is not a member of the edge.
    node_radius : Mapping or None
        Node id to marker radius. The padding is at least the largest member
        radius, and each excluded node is cut out by a disk of its own radius
        or the padding, whichever is larger.

    Returns
    -------
    BaseGeometry
        A polygon or multipolygon. Empty when ``edge_nodes`` is empty.
    """
    members = list(edge_nodes)
    if not members:
        return Polygon()

    radii = node_radius or {}
    edge_points = np.asarray([nodes_pos[node] for node in members], dtype=float)
    member_radii = np.asarray(
        [float(radii.get(node, 0.0)) for node in members], dtype=float
    )
    pad = _outline_padding(edge_points, member_radii, buffer_distance)
    geom = _with_padding(_member_hull(edge_points, alpha), pad)

    if other_nodes is None:
        member_ids = set(members)
        outsiders: Iterable[Any] = [
            node for node in nodes_pos if node not in member_ids
        ]
    else:
        outsiders = other_nodes
    geom = _cut_out_nodes(geom, outsiders, nodes_pos, radii, pad)
    if geom.is_empty:
        geom = unary_union(
            [
                Point(point).buffer(max(float(radius), pad))
                for point, radius in zip(edge_points, member_radii)
            ]
        )
    return geom


def _iter_polygons(geom: BaseGeometry) -> Iterable[Polygon]:
    """Yield polygons stored directly or inside a collection.

    Parameters
    ----------
    geom : BaseGeometry
        Geometry that may contain polygons.

    Yields
    ------
    Polygon
        Each polygon component.
    """
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for part in geom.geoms:
            yield from _iter_polygons(part)


def _geometry_to_path(geom: BaseGeometry) -> Path | None:
    """Convert a polygonal geometry, including holes, to a Matplotlib path.

    Parameters
    ----------
    geom : BaseGeometry
        Polygon, multipolygon, or collection.

    Returns
    -------
    Path or None
        ``None`` when ``geom`` contains no polygon.
    """

    def add_ring(
        ring: Any, vertices: list[tuple[float, float]], codes: list[int]
    ) -> None:
        coords = np.asarray(ring.coords, dtype=float)
        if len(coords) < 2:
            return
        vertices.extend((float(x), float(y)) for x, y in coords)
        codes.append(int(Path.MOVETO))
        codes.extend([int(Path.LINETO)] * (len(coords) - 2))
        codes.append(int(Path.CLOSEPOLY))

    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for polygon in _iter_polygons(geom):
        add_ring(polygon.exterior, vertices, codes)
        for interior in polygon.interiors:
            add_ring(interior, vertices, codes)
    if not vertices:
        return None
    return Path(vertices, codes)


def _with_fill_alpha(color: Any, fill_edge_alpha: float) -> Any:
    """Shift the alpha channel of an RGBA color.

    Parameters
    ----------
    color : Any
        Matplotlib color. Only length-4 numeric colors are changed.
    fill_edge_alpha : float
        Amount added to the alpha channel, then clipped to ``[0, 1]``.

    Returns
    -------
    Any
        Adjusted RGBA color, or ``color`` unchanged when it is not RGBA.
    """
    try:
        array = np.asarray(color, dtype=float)
    except (TypeError, ValueError):
        return color
    if array.shape != (4,):
        return color
    adjusted = array.copy()
    adjusted[3] = float(np.clip(adjusted[3] + fill_edge_alpha, 0.0, 1.0))
    return adjusted


def _patch_style(
    style: Mapping[str, Any],
    fill_edges: bool,
    fill_edge_alpha: float,
) -> dict[str, Any]:
    """Translate PolyCollection style keys into PathPatch keyword arguments.

    Parameters
    ----------
    style : Mapping
        Style for one edge, including ``edgecolors`` and ``facecolors``.
    fill_edges : bool
        When True and no face color was set, fill with the edge color.
    fill_edge_alpha : float
        Alpha added to that fill color. Negative values make the fill lighter.

    Returns
    -------
    dict
        Keyword arguments for :class:`matplotlib.patches.PathPatch`.
    """
    patch_kwargs = dict(style)
    edgecolor = patch_kwargs.pop("edgecolors", "black")
    facecolor = patch_kwargs.pop("facecolors", "none")
    if fill_edges and facecolor in {"none", None}:
        facecolor = _with_fill_alpha(edgecolor, fill_edge_alpha)
    patch_kwargs["edgecolor"] = edgecolor
    patch_kwargs["facecolor"] = facecolor
    return patch_kwargs


def _boundary_anchor(geom: BaseGeometry) -> tuple[np.ndarray, float]:
    """Return a label point on the longest exterior segment and its angle.

    Parameters
    ----------
    geom : BaseGeometry
        Edge outline.

    Returns
    -------
    tuple
        Midpoint of the longest exterior edge and a text angle in degrees,
        folded into ``[-90, 90]``.
    """
    polygons = list(_iter_polygons(geom))
    if not polygons:
        return np.zeros(2), 0.0
    polygon = max(polygons, key=lambda item: item.area)
    vertices = np.asarray(polygon.exterior.coords, dtype=float)
    if len(vertices) < 2:
        return vertices[0], 0.0
    lengths = ((vertices[:-1] - vertices[1:]) ** 2).sum(axis=1)
    index = int(lengths.argmax())
    start = vertices[index]
    end = vertices[index + 1]
    delta = end - start
    theta = 360.0 * float(np.arctan2(delta[1], delta[0])) / (2.0 * np.pi)
    theta = (theta + 360.0) % 360.0
    while theta > 90.0:
        theta -= 180.0
    return (start + end) / 2.0, theta


def draw_concave_hyper_edges(
    H: Any,
    pos: Mapping[Any, Sequence[float]],
    node_radius: Mapping[Any, float] | None = None,
    concavity: float = 0.5,
    buffer_distance: float = 0.1,
    ax: Any = None,
    fill_edges: bool = False,
    fill_edge_alpha: float = -0.5,
    **kwargs: Any,
) -> list[BaseGeometry]:
    """Draw one alpha-shape outline for each edge.

    Parameters
    ----------
    H : hnx.Hypergraph
        The entity to be drawn.
    pos : Mapping
        Node positions. Edge ids may also be present and are used only to
        place an empty edge.
    node_radius : Mapping or None
        Node id to marker radius.
    concavity : float
        Alpha-shape parameter. See :func:`get_concave_hull`.
    buffer_distance : float
        Padding forwarded to :func:`create_rubber_band`.
    ax : Axis
        Matplotlib axis. The current axis is used when this is omitted.
    fill_edges : bool
        Fill each outline with a more transparent copy of its edge color when
        no face color was given.
    fill_edge_alpha : float
        Amount added to the fill alpha. Typically negative.
    kwargs : dict
        Keyword arguments forwarded to the patch for each edge. Sequence
        values with one entry per edge, including the ``edgecolors`` array
        from :func:`add_edge_defaults`, are applied per edge.

    Returns
    -------
    list of BaseGeometry
        One geometry per edge, in iteration order. Missing empty edges are
        empty polygons.
    """
    ax = ax or plt.gca()
    styles = transpose_inflated_kwargs(inflate_kwargs(H.edges, kwargs))
    node_ids = list(H.nodes)
    geometries: list[BaseGeometry] = []
    for edge, style in zip(H.edges, styles):
        members = list(H.edges[edge])
        if members:
            outsiders = [node for node in node_ids if node not in members]
            geom = create_rubber_band(
                pos,
                members,
                buffer_distance=buffer_distance,
                alpha=concavity,
                other_nodes=outsiders,
                node_radius=node_radius,
            )
        elif edge in pos:
            radius = 4.0 * (min(node_radius.values()) if node_radius else 1.0)
            geom = Point(pos[edge]).buffer(radius)
        else:
            geometries.append(Polygon())
            continue
        geometries.append(geom)
        path = _geometry_to_path(geom)
        if path is not None:
            ax.add_patch(
                PathPatch(
                    path,
                    **_patch_style(style, fill_edges, fill_edge_alpha),
                )
            )
    return geometries


def draw_concave_edge_labels(
    H: Any,
    pos: Mapping[Any, Sequence[float]],
    labels: Iterable[Any],
    geometries: Sequence[BaseGeometry],
    edge_labels_on_edge: bool = True,
    ax: Any = None,
    **kwargs: Any,
) -> None:
    """Annotate concave edge outlines.

    Parameters
    ----------
    H : hnx.Hypergraph
        The entity to be drawn.
    pos : Mapping
        Node and edge positions. Used when labels are drawn inside an edge.
    labels : Iterable
        One label per edge, in the same order as ``H.edges``.
    geometries : Sequence
        Outlines returned by :func:`draw_concave_hyper_edges`.
    edge_labels_on_edge : bool
        When True, place the label on the longest boundary segment. Otherwise
        place it at the edge's layout position, or at the member centroid when
        that position is absent.
    ax : Axis
        Matplotlib axis. The current axis is used when this is omitted.
    kwargs : dict
        Keyword arguments forwarded to ``Axes.annotate``.
    """
    ax = ax or plt.gca()
    params = transpose_inflated_kwargs(inflate_kwargs(H.edges, kwargs))
    for edge, label, geom, edge_kwargs in zip(H.edges, labels, geometries, params):
        members = list(H.edges[edge])
        theta = 0.0
        if edge_labels_on_edge and not geom.is_empty:
            xy, theta = _boundary_anchor(geom)
        elif edge in pos:
            xy = np.asarray(pos[edge], dtype=float)
        elif members:
            xy = np.mean(
                [np.asarray(pos[node], dtype=float) for node in members], axis=0
            )
        else:
            continue
        ax.annotate(
            label,
            xy,
            **{"rotation": theta, "ha": "center", "va": "center", **edge_kwargs},
        )


def draw(
    H,
    pos=None,
    layout=nx.spring_layout,
    layout_kwargs=None,
    ax=None,
    node_radius=None,
    fill_edges=False,
    fill_edge_alpha=-0.5,
    edges_kwargs=None,
    nodes_kwargs=None,
    edge_labels_on_edge=True,
    edge_labels=None,
    edge_labels_kwargs=None,
    node_labels=None,
    node_labels_kwargs=None,
    with_edge_labels=True,
    with_node_labels=True,
    node_label_alpha=0.35,
    edge_label_alpha=0.35,
    with_additional_edges=None,
    contain_hyper_edges=False,
    additional_edges_kwargs=None,
    return_pos=False,
    convex: bool = True,
    concavity: float = 0.5,
    buffer_distance: float = 0.1,
):
    """
    Draw a hypergraph as a Matplotlib figure

    By default this will draw a colorful "rubber band" like hypergraph, where
    convex hulls represent edges and are drawn around the nodes they contain.
    Set ``convex`` to False to draw an alpha shape instead. The outline can
    then bend around, and cut out, nodes that are not members of the edge.

    This is a convenience function that wraps calls with sensible parameters to
    the following lower-level drawing functions:

    * draw_hyper_edges or draw_concave_hyper_edges,
    * draw_hyper_edge_labels or draw_concave_edge_labels,
    * draw_hyper_labels, and
    * draw_hyper_nodes

    The default layout algorithm is nx.spring_layout, but other layouts can be
    passed in. The Hypergraph is converted to a bipartite graph, and the layout
    algorithm is passed the bipartite graph.

    If you have a pre-determined layout, you can pass in a "pos" dictionary.
    This is a dictionary mapping from node id's to x-y coordinates. For example:

        >>> pos = {
        >>> 'A': (0, 0),
        >>> 'B': (1, 2),
        >>> 'C': (5, -3)
        >>> }

    will position the nodes {A, B, C} manually at the locations specified. The
    coordinate system is in Matplotlib "data coordinates", and the figure will
    be centered within the figure.

    By default, this will draw in a new figure, but the axis to render in can be
    specified using :code:`ax`.

    This approach works well for small hypergraphs, and does not guarantee
    a rigorously "correct" drawing. Overlapping of sets in the drawing generally
    implies that the sets intersect, but sometimes sets overlap if there is no
    intersection. It is not possible, in general, to draw a "correct" hypergraph
    this way for an arbitrary hypergraph, in the same way that not all graphs
    have planar drawings.

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    pos: dict
        mapping of node and edge positions to R^2
    layout: function
        layout algorithm to compute
    layout_kwargs: dict
        keyword arguments passed to layout function
    ax: Axis
        matplotlib axis on which the plot is rendered
    fill_edges: bool
        set to True to fill set the facecolor of edges to a lighter version of the edgecolor if no facecolor is otherwise specified
    fill_edge_alpha: float
        amount to add to the alpha channel when filling edges. Should be between -1 and 0, causing a decrease in alpha
    edges_kwargs: dict
        keyword arguments passed to matplotlib.collections.PolyCollection for edges
    node_radius: None, int, float, or dict
        radius of all nodes, or dictionary of node:value; the default (None) calculates radius based on number of collapsed nodes; reasonable values range between 1 and 3
    nodes_kwargs: dict
        keyword arguments passed to matplotlib.collections.PolyCollection for nodes
    edge_labels_on_edge: bool
        whether to draw edge labels on the edge (rubber band) or inside
    edge_labels_kwargs: dict
        keyword arguments passed to matplotlib.annotate for edge labels
    node_labels_kwargs: dict
        keyword argumetns passed to matplotlib.annotate for node labels
    with_edge_labels: bool
        set to False to make edge labels invisible
    with_node_labels: bool
        set to False to make node labels invisible
    node_label_alpha: float
        the transparency (alpha) of the box behind text drawn in the figure for node labels
    edge_label_alpha: float
        the transparency (alpha) of the box behind text drawn in the figure for edge labels
    with_additional_edges: networkx.Graph
        ...
    contain_hyper_edges: bool
        whether the rubber band shoudl be drawn around the location of the edge in the bipartite graph. This may be invisibile unless "with_additional_edges" contains this information.
    convex: bool
        When True, each edge is the convex hull of its nodes. When False, each
        edge is an alpha shape. Nodes that are not members of the edge are cut
        out of that shape when they fall inside it.
    concavity: float
        Alpha-shape parameter used when ``convex`` is False. Triangles whose
        circumradius is greater than or equal to ``1 / concavity`` are removed,
        so larger values are more concave. Values less than or equal to 0 keep
        the convex hull. The value is in coordinate units: a spring layout in a
        unit box often needs a larger value, such as 4, before bays appear.
    buffer_distance: float
        When ``convex`` is False, fraction of the mean pairwise distance among
        an edge's nodes used to expand the outline. A one-node edge is expanded
        by this distance directly.

    """

    ax = ax or plt.gca()
    (
        layout_kwargs,
        edges_kwargs,
        nodes_kwargs,
        edge_labels_kwargs,
        node_labels_kwargs,
        additional_edges_kwargs,
    ) = _as_dicts(
        layout_kwargs,
        edges_kwargs,
        nodes_kwargs,
        edge_labels_kwargs,
        node_labels_kwargs,
        additional_edges_kwargs,
    )

    if pos is None:
        pos = layout_node_link(H, with_additional_edges, layout=layout, **layout_kwargs)

    # guarantee that node radius is a dictionary mapping nodes to values
    r0 = get_default_radius(H, pos)
    node_radius = dict(
        zip(
            H.nodes,
            [
                r0 * r
                for r in inflate(H.nodes, 1 if node_radius is None else node_radius)
            ],
        )
    )

    # for convenience, we are using setdefault to mutate the argument
    # however, we need to copy this to prevent side-effects
    edges_kwargs = add_edge_defaults(H, edges_kwargs)

    geometries: list[BaseGeometry] = []
    if convex:
        polys = draw_hyper_edges(
            H,
            pos,
            node_radius=node_radius,
            ax=ax,
            contain_hyper_edges=contain_hyper_edges,
            fill_edges=fill_edges,
            fill_edge_alpha=fill_edge_alpha,
            **edges_kwargs,
        )
    else:
        polys = None
        geometries = draw_concave_hyper_edges(
            H,
            pos,
            node_radius=node_radius,
            concavity=concavity,
            buffer_distance=buffer_distance,
            ax=ax,
            fill_edges=fill_edges,
            fill_edge_alpha=fill_edge_alpha,
            **edges_kwargs,
        )

    if with_additional_edges:
        nx.draw_networkx_edges(
            with_additional_edges,
            pos=pos,
            ax=ax,
            **inflate_kwargs(with_additional_edges.edges(), additional_edges_kwargs),
        )

    if with_edge_labels:
        edge_label_values = inflate(
            H.edges, list(H.edges) if edge_labels is None else edge_labels
        )
        if convex:
            draw_hyper_edge_labels(
                H,
                pos,
                edge_label_values,
                polys,
                backgroundcolor=(1, 1, 1, edge_label_alpha),
                ax=ax,
                edge_labels_on_edge=edge_labels_on_edge,
                **{"color": polys.get_edgecolors(), **edge_labels_kwargs},
            )
        else:
            draw_concave_edge_labels(
                H,
                pos,
                edge_label_values,
                geometries,
                backgroundcolor=(1, 1, 1, edge_label_alpha),
                ax=ax,
                edge_labels_on_edge=edge_labels_on_edge,
                **{"color": edges_kwargs["edgecolors"], **edge_labels_kwargs},
            )

    if with_node_labels:
        draw_hyper_labels(
            H,
            pos,
            inflate(H.nodes, list(H.nodes) if node_labels is None else node_labels),
            node_radius=node_radius,
            ax=ax,
            **{
                "va": "center",
                "xytext": (5, 0),
                "textcoords": "offset points",
                "backgroundcolor": (1, 1, 1, node_label_alpha),
                **node_labels_kwargs,
            },
        )

    draw_hyper_nodes(H, pos, node_radius=node_radius, ax=ax, **nodes_kwargs)

    if len(H.nodes) == 1:
        x, y = pos[next(iter(H.nodes))]
        s = 20

        ax.axis([x - s, x + s, y - s, y + s])
    else:
        ax.axis("equal")

    ax.axis("off")
    if return_pos:
        return pos
