# Copyright © 2024 Battelle Memorial Institute
# All rights reserved.

import hypernetx as hnx
import pandas as pd
import json
import fastjsonschema
import requests
from copy import deepcopy
from .exception import HyperNetXError

schema_url = "https://raw.githubusercontent.com/pszufe/HIF_validators/main/schemas/hif_schema_v0.1.0.json"


def normalize_dataframe(df):
    """
    Moves common attributes into misc_properties for translating into HIF.

    Parameters
    ----------
    df : pd.DataFrame
        HypergraphView.dataframe

    Returns
    -------
    pd.DataFrame
        allowed columns are limited to HIF keys
    """
    default_cols = (
        ["weight"]
        + list(set(df.columns).intersection(["direction"]))
        + ["misc_properties"]
    )
    cols = list(set(df.columns).difference(default_cols))
    dfdict = df[cols].T.to_dict()
    newdf = df[default_cols]
    for uid in newdf.index:
        newdf.loc[uid]["misc_properties"].update(dfdict[uid])
    return newdf.fillna("nil")


def to_hif(hg, filename=None, network_type="undirected", metadata=None):
    """
    Returns a dictionary object valid for the HIF Json schema

    Parameters
    ----------
    hg : hnx.Hypergraph

    filename : str, optional
        filepath where json object is to be stored, by default None
    network_type : str, optional
        One of 'undirected','directed','asc', by default 'undirected'
    metadata : dict, optional
        Additional information to store, by default None

    Returns
    -------
    hif : dict
        format is defined by HIF schema
    """

    resp = requests.get(schema_url)
    schema = json.loads(resp.text)
    validator = fastjsonschema.compile(schema)

    hyp_objs = ["nodes", "edges", "incidences"]
    defaults = {
        part: dict(getattr(hg, part).property_store._defaults) for part in hyp_objs
    }
    for part in hyp_objs:
        misc_properties = defaults[part].pop("misc_properties", {})
        defaults[part]["attrs"] = dict(misc_properties)

    incj = deepcopy(hg.incidences.to_dataframe)
    incj.index.names = ["edge", "node"]
    incj = normalize_dataframe(incj)
    incj = incj.rename(columns={"misc_properties": "attrs"})
    incj = incj.reset_index().to_dict(orient="records")

    edgj = deepcopy(hg.edges.to_dataframe)
    edgj.index.names = ["edge"]
    edgj = normalize_dataframe(edgj)
    edgj = edgj.rename(columns={"misc_properties": "attrs"})
    edgj = edgj.reset_index().to_dict(orient="records")

    nodj = deepcopy(hg.nodes.to_dataframe)
    nodj.index.names = ["node"]
    nodj = normalize_dataframe(nodj)
    nodj = nodj.rename(columns={"misc_properties": "attrs"})
    nodj = nodj.reset_index().to_dict(orient="records")

    if isinstance(metadata, dict):
        metadata = metadata.update({"default_attrs": defaults})
    else:
        metadata = {"default_attrs": defaults}
    if hg.name is not None:
        metadata["name"] = hg.name

    hif = {
        "edges": edgj,
        "nodes": nodj,
        "incidences": incj,
        "network-type": network_type,
        "metadata": metadata,
    }
    try:
        validator(hif)
        if filename is not None:
            json.dump(hif, open(filename, "w"))
        return hif
    except Exception as ex:
        HyperNetXError(ex)


def from_hif(hif=None, filename=None):
    """
    Reads HIF formatted string or dictionary and returns corresponding
    hnx.Hypergraph

    Parameters
    ----------
    hif : dict, optional
        Useful if file is read by json and inspected before turning into a hypergraph,
        by default None
    filename : str, optional
        Full path to location of HIF formatted JSON in storage,
        by default None

    Returns
    -------
    hnx.Hypergraph

    """

    resp = requests.get(schema_url)
    schema = json.loads(resp.text)
    validator = fastjsonschema.compile(schema)

    if hif is not None:
        try:
            validator(hif)
        except Exception as ex:
            HyperNetXError(ex)
            return None
    elif filename is not None:
        hif = json.load(open(filename, "r"))
        try:
            validator(hif)
        except Exception as ex:
            HyperNetXError(ex)
            return None
    else:
        print("No data given")

    mkdd = lambda: {"weight": 1, "attrs": {}}
    hifex = deepcopy(hif)
    parts = {
        part: deepcopy(pd.DataFrame(hifex.get(part, {})))
        for part in ["nodes", "edges", "incidences"]
    }
    metadata = hifex.get("metadata", {})
    defaults = metadata.get("default_attrs", {})
    defaults = {part: defaults.get(part, mkdd()) for part in parts}
    # cols = dict()
    default_weights = {part: defaults[part].get("weight", 1) for part in parts}
    for part in parts:
        if len(part) == 0:
            continue
        thispart = parts[part]
        d = deepcopy(defaults[part])
        dkeys = [k for k in d.keys() if k not in ["weight", "attrs"]]
        # cols[part] = ['weight'] + dkeys + ['attrs']
        if len(dkeys) > 0:
            for attr in dkeys:
                thispart[attr] = [
                    row.attrs.pop(attr, d[attr]) for row in thispart.itertuples()
                ]
    hyp_objects = dict()
    for part in ["nodes", "edges"]:
        if len(parts[part]) > 0:
            uid = part[:-1]
            cols = [uid] + list(set(parts[part].columns).difference([uid]))
            hyp_objects[part] = parts[part][cols]
        else:
            hyp_objects[part] = None
    cols = ["edge", "node"] + list(
        set(parts["incidences"].columns).difference(["edge", "node"])
    )
    incidences = parts["incidences"][cols]
    name = metadata.get("name", None)
    return hnx.Hypergraph(
        incidences,
        default_cell_weight=default_weights["incidences"],
        misc_cell_properties_col="attrs",
        node_properties=hyp_objects["nodes"],
        default_edge_weight=default_weights["edges"],
        edge_properties=hyp_objects["edges"],
        default_node_weight=default_weights["nodes"],
        misc_properties_col="attrs",
        name=name,
    )
