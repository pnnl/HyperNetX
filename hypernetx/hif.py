# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

import json
import fastjsonschema
import requests
from .exception import HyperNetXError

schema_url = "https://raw.githubusercontent.com/pszufe/HIF_validators/main/schemas/hif_schema_v0.1.0.json"
resp = requests.get(url)
schema = json.loads(resp.text)
validator = fastjsonschema.compile(schema)


def from_hif(hif):
    assert validator(hif)
    incidences = pd.DataFrame(hif["incidences"])
    misc_cell_properties_col = 'attrs'  if 'attrs' in incidences.columns else None

    if "edges" in hif and len(hif["edges"])>0:
        edges = pd.DataFrame(hif['edges']).rename(columns={'edge':'id'})
        misc_edge_properties_col = 'attrs'  if 'attrs' in edges.columns else None
    else:
        edges = None
        misc_edge_properties_col = None

    if "nodes" in hif and len(hif["nodes"])>0:
        nodes = pd.DataFrame(hif['nodes']).rename(columns={'node':'id'})
        misc_node_properties_col = 'attrs'  if 'attrs' in nodes.columns else None
    else:
        nodes = None
        misc_node_properties_col = None
    
    h = hnx.Hypergraph(incidences, misc_cell_properties_col=misc_cell_properties_col,
                          node_properties=nodes, misc_node_properties_col=misc_node_properties_col,
                          edge_properties=edges, misc_edge_properties_col=misc_edge_properties_col,
                          )    
    ## store additional data for reuse later
    h.hif_data = {k:hif[k] for k in set(hif.keys()).difference(['edges','nodes','incidences'])}
    return h
                  

def to_hif(hg):
    edgj = hg.edges.to_dataframe.rename(columns={"misc_properties":"attrs"})
    edid = edgj.index._name or "index"
    nodj = hg.nodes.to_dataframe.rename(columns={"misc_properties":"attrs"})
    ndid = nodj.index._name or "index"
    edgj = edgj.reset_index().rename(columns={edid: "edge"}).to_dict(orient="records")
    nodj = nodj.reset_index().rename(columns={ndid: "node"}).to_dict(orient="records")
    incj = (
        hg.incidences.to_dataframe.reset_index()
        .rename(columns={"nodes": "node", "edges": "edge","misc_properties":"attrs"})
        .to_dict(orient="records")
    )
    hif = {"edges": edgj, "nodes": nodj, "incidences": incj}
    hif.update(getattr(hg,'hif_data',{}))
    assert validator(hif)
    return hif