import numpy as np
import pytest
from hypernetx import Entity, EntitySet
from hypernetx import HyperNetXError

# things to test
# EntitySet: add and remove methods, getting children, adding throws conflict, len, construct incidence dict and matrix, adding dict or EntitySet, collapse.
# Hypergraph: constructor


# Test entity
def test_entity_edge_constructor():
    ent = Entity("edge", ["A", "C", "K"], weight=2.0)
    assert ent.uid == "edge"
    assert len(ent) == 3
    assert ent.size == 3
    assert ent._elements == {"A", "C", "K"}
    assert str(ent) == "edge"
    assert ent.properties["weight"] == 2.0
    assert "A" in ent
    assert ent.clone("edgecopy").uid == "edgecopy"
    assert ent.clone("edgecopy")._elements == {"A", "C", "K"}
    assert ent.clone("edgecopy").properties["weight"] == 2.0
    ent.cellweights=(1,2,3)
    ent.directions={"A":-1, "C":1, "K":-1}
    assert ent.properties["cellweights"] == (1,2,3)
    assert ent.properties["directions"] == {"A":-1, "C":1, "K":-1}

# test updating property

def test_entity_construct_from_entity():
    e1 = Entity("e1", elements=[1, 2, 3])
    e2 = Entity("e2", entity=e1)
    e3 = Entity("e3", entity=e2, elements=[6, 7])
    assert e1 != e2
    assert e1._elements == e2._elements
    assert e1._elements == e3._elements
    assert e3._elements == {1, 2, 3}

def test_entityset_construction():
    edgeDict = {0:(1, 2, 3), 1: (1,3,5), 2: (2,3,5)}
    edgeList = [(1,2,3), (1,3,5), (2,3,5)]
    assert EntitySet("es1", edgeDict) == EntitySet("es1", edgeList)
    assert EntitySet("es1", edgeDict) == EntitySet("es1", entityset=EntitySet("es1", edgeDict))
    assert EntitySet("es1", edgeList) == EntitySet("es1", entityset=EntitySet("es1", edgeList))
    assert EntitySet("es1").uid == "es1"
    assert len(EntitySet("es1", edgeList)) == 3
    assert 2 in EntitySet("es1", edgeList)
    assert Entity(2, (2,3,5)) in EntitySet("es1", edgeList)

def test_entity_add():
    # add an element with property
    edgeDict = {0:(1, 2, 3), 1: (1,3,5), 2: (2,3,5,6)}
    es = EntitySet("es", edgeDict)
    ent = Entity("e", {"A", "C", "K"})
    es._add({"D":Entity("D", [1, 2, 3], color="red")})
    assert len(es) == 4
    assert 2 in es["D"]
    assert es["D"].color == "red"
    es._add({ent.uid: ent})
    assert "e" in es
    assert es.children == {1,2,3,5,6,"A","C","K"}

def test_add_no_overwrite():
    # adding and entity of the same name as an existing entity
    # only updates properties
    e1 = Entity("e1", {"A", "C", "K"})
    e2 = Entity("D", ["X"], w=5)
    assert "X" in e2
    e1._add(e2)
    assert "D" in e1
    assert e2.w == 5

def test_entity_remove():
    ent = Entity("e", {"A", "C", "K"})
    assert ent.size == 3
    ent._remove("A")
    assert ent.size == 2
