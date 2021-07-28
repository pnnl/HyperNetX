import numpy as np
import pytest
from hypernetx import Entity, EntitySet
from hypernetx import HyperNetXError


def test_entity_constructor():
    ent = Entity("edge", {"A", "C", "K"})
    assert ent.uid == "edge"
    assert ent.size() == 3
    assert len(ent.uidset) == 3
    assert len(ent.children) == 0
    assert isinstance(ent.incidence_dict["A"], set)
    assert "A" in ent


def test_entity_construct_from_entity(triloop):
    e1 = Entity("e1", elements=triloop.edgedict)
    e2 = Entity("e2", entity=e1)
    e3 = Entity("e3", entity=e2, elements={"Z": "A"})
    assert e1 != e2
    assert e1.elements == e2.elements
    assert e1.children == e3.children
    assert e1.elements != e3.elements
    with pytest.raises(HyperNetXError):
        e4 = Entity("e1", entity=e1)


def test_entity_add():
    # add an element with property
    ent = Entity("e", {"A", "C", "K"})
    ent.add(Entity("D", [1, 2, 3], color="red"))
    assert ent.size() == 4
    assert len(ent.children) == 3
    assert 2 in ent["D"].elements
    assert ent["D"].color == "red"


def test_entity_no_self_reference():
    # confirm entity may not add itself.
    ent = Entity("e", {"A", "C", "K"})
    with pytest.raises(HyperNetXError) as excinfo:
        ent.add(Entity("e"))
    assert "Self reference" in str(excinfo.value)
    with pytest.raises(HyperNetXError) as excinfo2:
        ent.add(ent)
    assert "Self reference" in str(excinfo2.value)


def test_add_and_merge_properties():
    # assert that adding an existing entity will not overwrite
    e1 = Entity("e1", {"A", "C", "K"})
    e2 = Entity("C", w=5)
    e1.add(e2)
    assert e1["C"].w == 5
    ent3 = Entity("C")
    e1.add(ent3)
    assert e1["C"].w == 5


def test_add_no_overwrite():
    # adding and entity of the same name as an existing entity
    # only updates properties
    e1 = Entity("e1", {"A", "C", "K"})
    e2 = Entity("C", ["X"], w=5)
    assert "X" in e2.elements
    e1.add(e2)
    assert "X" not in e1["C"]
    assert e1["C"].w == 5


def test_entity_remove():
    ent = Entity("e", {"A", "C", "K"})
    assert ent.size() == 3
    ent.remove("A")
    assert ent.size() == 2


def test_entity_depth():
    e1 = Entity("e1")
    e2 = Entity("e2", [e1])
    e3 = Entity("e3", [e2])
    e4 = Entity("e4", [e3, e1])
    assert e4.depth() == 3
    e3.remove(e2)
    assert e4.depth() == 1


def test_entity_set():
    eset = EntitySet("eset1", {"A", "C", "K"})
    eset2 = eset.clone("eset2")
    assert eset.elements == eset2.elements
    assert len(eset) == 3
    assert eset.uid == "eset1"
    assert len(eset.children) == 0
    with pytest.raises(HyperNetXError) as excinfo:
        eset2.add(Entity("Z", ["A", "C", "B"]))
    assert "Fails the bipartite condition" in str(excinfo.value)


def test_entity_set_from_dict(seven_by_six):
    sbs = seven_by_six
    eset = EntitySet("sbs", sbs.edgedict)
    M, rowdict, coldict = eset.incidence_matrix(index=True)
    assert len(rowdict) == 7
    assert len(coldict) == 6
    x = np.ones(6).transpose()
    assert np.max(M.dot(x)) == 3


def test_entity_from_dict_mixed_types():
    e1 = Entity("e1", [2])
    d = {"e1": e1, "e2": [5]}
    e3 = Entity("e3", d)
    assert 2 in e3.children


def test_equality():
    # Different uids, elements generated differently
    e1 = Entity("e1", [1, 2, 3])
    e2 = Entity("e2", [1, 2, 3])
    assert not e1 == e2
    assert not e1[1] == e2[1]
    # Different uids, elements generated the same
    elts = [Entity(uid) for uid in [1, 2, 3]]
    e1 = Entity("e1", elts)
    e2 = Entity("e2", elts)
    assert not e1 == e2
    assert e1[1] == e2[1]
    # Different properties only
    e1 = Entity("e1", elts)
    e2 = Entity("e1", elts, weight=2)
    assert not e1 == e2


def test_merge_entities():
    x = Entity("x", [1, 2, 3], weight=1, color="r")
    y = Entity("y", [2, 3, 4], weight=3)
    z = Entity.merge_entities("z", x, y)
    assert z.uidset == set([1, 2, 3, 4])
    assert z.weight == 3
    assert z.color == "r"
