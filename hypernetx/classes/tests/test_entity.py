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
    assert ent.elements == ["A", "C", "K"]
    assert str(ent) == "edge"
    assert ent.properties["weight"] == 2.0
    assert "A" in ent
    assert ent.clone("edgecopy").uid == "edgecopy"
    assert ent.clone("edgecopy").elements == ["A", "C", "K"]
    assert ent.clone("edgecopy").properties["weight"] == 2.0
    ent.update_properties(cellweights=(1,2,3), directions={"A":-1, "C":1, "K":-1})
    assert ent.properties["cellweights"] == (1,2,3)
    assert ent.properties["directions"] == {"A":-1, "C":1, "K":-1}

# test updating property

def test_entity_construct_from_entity():
    e1 = Entity("e1", elements=[1, 2, 3])
    e2 = Entity("e2", entity=e1)
    e3 = Entity("e3", entity=e2, elements=[6, 7])
    assert e1 != e2
    assert e1.elements == e2.elements
    assert e1.elements == e3.elements
    assert e3.elements == [1, 2, 3]
    with pytest.raises(HyperNetXError):
        e4 = Entity("e1", entity=e1)

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
    es.add({"D":Entity("D", [1, 2, 3], color="red")})
    assert len(es) == 4
    assert 2 in es["D"].elements
    assert es["D"].color == "red"
    es.add({ent.uid: ent})
    assert "e" in es
    assert es.children == {1,2,3,5,6,"A","C","K"}

# def test_entity_no_self_reference():
#     # confirm entity may not add itself.
#     ent = Entity("e", {"A", "C", "K"})
#     with pytest.raises(HyperNetXError) as excinfo:
#         ent.add(Entity("e"))
#     assert "Self reference" in str(excinfo.value)
#     with pytest.raises(HyperNetXError) as excinfo2:
#         ent.add(ent)
#     assert "Self reference" in str(excinfo2.value)


# def test_add_and_merge_properties():
#     # assert that adding an existing entity will not overwrite
#     e1 = Entity("e1", {"A", "C", "K"})
#     e2 = Entity("C", w=5)
#     e1.add(e2)
#     assert e1["C"].w == 5
#     ent3 = Entity("C")
#     e1.add(ent3)
#     assert e1["C"].w == 5


def test_add_no_overwrite():
    # adding and entity of the same name as an existing entity
    # only updates properties
    e1 = Entity("e1", {"A", "C", "K"})
    e2 = Entity("D", ["X"], w=5)
    assert "X" in e2.elements
    e1.add(e2)
    assert "D" in e1
    assert e2.w == 5

def test_entity_remove():
    ent = Entity("e", {"A", "C", "K"})
    assert ent.size == 3
    ent.remove("A")
    assert ent.size == 2

# def test_entity_set():
#     eset = EntitySet("eset1", {"A", "C", "K"})
#     eset2 = eset.clone("eset2")
#     assert eset.children == eset2.children
#     assert len(eset) == 1
#     assert eset.uid == "eset1"
#     assert len(eset) == 1

# def test_entity_set_from_dict(seven_by_six):
#     sbs = seven_by_six
#     eset = EntitySet("sbs", sbs.edgedict)
#     M, rowdict, coldict = eset.incidence_matrix(index=True)
#     assert len(rowdict) == 7
#     assert len(coldict) == 6
#     x = np.ones(6).transpose()
#     assert np.max(M.dot(x)) == 3


# def test_entity_from_dict_mixed_types():
#     e1 = Entity("e1", [2])
#     d = {"e1": e1, "e2": [5]}
#     e3 = Entity("e3", d)
#     assert 2 in e3.children


# def test_equality():
#     # Different uids, elements generated differently
#     e1 = Entity("e1", [1, 2, 3])
#     e2 = Entity("e2", [1, 2, 3])
#     assert not e1 == e2
#     assert not e1[1] == e2[1]
#     # Different uids, elements generated the same
#     elts = [Entity(uid) for uid in [1, 2, 3]]
#     e1 = Entity("e1", elts)
#     e2 = Entity("e2", elts)
#     assert not e1 == e2
#     assert e1[1] == e2[1]
#     # Different properties only
#     e1 = Entity("e1", elts)
#     e2 = Entity("e1", elts, weight=1)
#     assert not e1 == e2


# def test_merge_entities():
#     x = Entity("x", [1, 2, 3], weight=1, color="r")
#     y = Entity("y", [2, 3, 4], weight=3)
#     z = Entity.merge_entities("z", x, y)
#     assert z.uidset == set([1, 2, 3, 4])
#     assert z.weight == 3
#     assert z.color == "r"
