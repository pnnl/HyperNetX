import hypernetx as hnx

from collections import defaultdict
from functools import lru_cache
import re

import urllib

import networkx as nx

import numpy as np


def from_bipartite(G):
    """
    Construct an incidence dictionary from a bipartite graph

    The ordering of the nodes within edges is used to determine the partitions
    of the graph. In other words, all source nodes are in one part and all
    target nodes are in the other part.

    Parameters
    ----------
    G: nx.Graph or nx.DiGraph
        the bipartite graph to convert

    Returns
    -------
    dict
        an incidence dictionary

    """

    incidences = defaultdict(list)

    for v, e in G.edges():
        incidences[e].append(v)

    return incidences


def create_ordered_hypergraph_with_kwargs(incidence_dict, sort_key=None, **kwargs):
    """
    Convenience function to create a hypergraph and edge ordering

    Parameters
    ----------
    incidence_dict: dict
        the incidence dictionary defining the hypergraph
    sort_key: func
        function to sort the hyper edges by
    **kwargs: dict
        additional pass through arguments for convenience

    Returns
    -------
    hypernetx.Hypergraph
        the constructed hypergraph
    dict
        kwargs to be used by layout functions, for example

    """

    H = hnx.Hypergraph(incidence_dict)
    return H, dict(edge_order=sorted(H.edges, key=sort_key), **kwargs)


def create_lesmis_small():
    """
    Create canonical Les Miserables subgraph

    Hyper edges represent scenes from the play, so the edge order returned
    reflects this.

    """
    return create_ordered_hypergraph_with_kwargs(
        {
            0: ("FN", "TH"),
            1: ("TH", "JV"),
            2: ("BM", "FN", "JA"),
            3: ("JV", "JU", "CH", "BM"),
            4: ("JU", "CH", "BR", "CN", "CC", "JV", "BM"),
            5: ("TH", "GP"),
            6: ("GP", "MP"),
            7: ("MA", "GP"),
        },
        int,
    )


def create_davis_southern_women():
    """
    Returns the Davis Southern Women dataset as a hypergraph

    This dataset is available in networkx.davis_southern_women_graph as a
    bipartite graph. The events in the graph also define an ordering of the
    hyperedges.

    """

    return create_ordered_hypergraph_with_kwargs(
        from_bipartite(nx.davis_southern_women_graph()), lambda e: int(e[1:])
    )


def create_star_wars():
    """
    Returns a representation of the first few interactions in Star Wars based
    on a the XKCD comic's storyline visualization [1].

    References
    ----------
    [1] https://xkcd.com/657/

    """

    VADER = "Vader"
    LEIA = "Leia"
    R2 = "R2-D2"
    C3PO = "C-3PO"
    OBIWAN = "Obi-Wan"
    LUKE = "Luke"
    HAN = "Han"
    CHEWIE = "Chewie"
    JABBA = "Jabba"

    events = [
        {VADER},
        {LEIA, R2},
        {C3PO},
        {OBIWAN},
        {LUKE},
        {HAN, CHEWIE},
        {JABBA},
        {VADER, LEIA},
        {R2, C3PO},
        {R2, C3PO, LUKE},
        {OBIWAN, R2, C3PO, LUKE},
        {OBIWAN, R2, C3PO, LUKE, HAN, CHEWIE, JABBA},
        {LEIA, LUKE, HAN, CHEWIE},  # Leia rescued
        {R2, C3PO},
        {OBIWAN, VADER},  # Duel
        {VADER, LUKE, R2},  # Death star
        {C3PO, LEIA},
        {VADER, LUKE, R2, HAN, CHEWIE},
        {VADER},
        {LUKE, R2, HAN, CHEWIE, LEIA, C3PO},
        {LUKE},
        {JABBA},
    ]

    def rgb(*args):
        return np.array(args) / 255.0

    node_colors = defaultdict(
        lambda: rgb(158, 158, 158),
        **{
            VADER: rgb(0, 0, 0),
            C3PO: rgb(208, 196, 0),
            R2: rgb(46, 124, 230),
            CHEWIE: rgb(167, 100, 40),
            JABBA: rgb(130, 189, 119),
        },
    )

    labels = [
        ({LEIA, LUKE, HAN, CHEWIE}, "Leia\nrescued"),
        ({OBIWAN, VADER}, "Duel"),
        ({VADER, LUKE, R2}, "Death\nStar"),
    ]

    edge_labels = defaultdict(lambda: "")
    for k, v in labels:
        edge_labels[events.index(k)] = v

    return create_ordered_hypergraph_with_kwargs(
        dict(enumerate(events)),
        nodes_kwargs={"edgecolor": node_colors},
        edges_kwargs={"edgecolor": None, "facecolor": rgb(211, 211, 211)},
        edge_labels=edge_labels,
        edge_labels_on_axis=False,
        edge_labels_kwargs=dict(fontsize=6, backgroundcolor=(1, 1, 1, 0.5)),
    )


@lru_cache()
def load_file(path_or_url, encoding="utf-8"):
    """
    Cached retrieval of file from URL

    Parameters
    ----------
    path_or_url: string
        location of resource to load
    encoding: str
        character encoding of file

    Returns
    -------
    str
        string representation of file contents

    """
    print("Reading", path_or_url)

    with urllib.request.urlopen(path_or_url) as fp:
        return fp.read().decode(encoding)


def parse_play(
    path_or_url,
    encoding="utf-8",
    start=None,
    end=None,
    start_str=None,
    end_str=None,
    ignore=set(),
    replace=dict(),
    return_text=False,
    split_act="ACT [IXV]+",
    split_scene="SCENE [IXV]+",
):
    """
    Load and parse certain content from Project Gutenberg [1] into a hypergraph

    This is intended to load plays, e.g. Hamlet, that have a specific
    formatting allowing a simple parsing out of acts, scenes, and characters.
    The result is a hypergraph where the edges are "act.scene" strings and the
    nodes are the characters in the play.

    Parameters
    ----------
    path_or_url: str
        location of resource to load
    encoding: str
        string encoding of resource
    start: int
        starting string offset to start parsing, i.e., to ignore front matter
    end: int
        ending string offset to stop parsing, i.e., to ignore end matter
    start_str:
        pattern to match to start parsing, e.g. "ACT I."
    end_str:
        pattern to stop parsing, e.g. "*** END"
    ignore: set
        minor characters or potential parsing errors to ignore
    replace: dict
        mapping of names to replace, e.g., BARNARD -> BARNARDO
    split_act: str
        regular expression to split text into acts
    split_scene: str
        regular expression to split text into scenes

    Returns
    -------
    hypernetx.Hypergraph
        hypergraph representing interactions between characters in scenes
    dict
        contains the edge_order property which sortes edges by act and scene


    References
    ----------
    [1] https://www.gutenberg.org/

    """
    txt = load_file(path_or_url, encoding).replace("\r", "")

    if start_str:
        start = txt.index(start_str)

    if end_str:
        end = txt.index(end_str)

    txt = txt[start:end]

    edges = {}

    for i, act in enumerate(re.split(split_act, txt)[1:]):
        for j, scene in enumerate(re.split(split_scene, act)[1:]):
            nodes = set(map(str.strip, re.findall("([A-Z ]+).\n", scene))).difference(
                ignore
            )

            edges[f"{i + 1}.{j + 1}"] = set([replace.get(v, v) for v in nodes])

    if return_text:
        return edges, txt

    return edges


def create_macbeth(path=None):
    return create_ordered_hypergraph_with_kwargs(
        parse_play(
            path or "https://www.gutenberg.org/cache/epub/1533/pg1533.txt",
            start_str="ACT I\n\nSCENE I.",
            end_str="*** END OF THE PROJECT GUTENBERG",
            ignore={"", "I", "ALL", "BOTH MURDERERS"},
            replace={"MURDERER": "FIRST MURDERER", "LORDS": "LORD"},
        )
    )


def create_hamlet(path=None):
    return create_ordered_hypergraph_with_kwargs(
        parse_play(
            path or "https://www.gutenberg.org/cache/epub/1524/pg1524.txt",
            start_str="ACT I\n\nSCENE I.",
            end_str="*** END OF THE PROJECT GUTENBERG",
            ignore={"", "I", "T", "ALL", "BOTH"},
            replace={
                "BARNARD": "BARNARDO",
                "LORD": "LORDS",
                "FIRST CLOWN": "CLOWNS",
                "SECOND CLOWN": "CLOWNS",
                "PLAYER KING": "PLAYER KING & QUEEN",
                "PLAYER QUEEN": "PLAYER KING & QUEEN",
            },
        )
    )
