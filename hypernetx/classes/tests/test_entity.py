import numpy as np

from collections.abc import Iterable
from collections import UserList
from hypernetx.classes import Entity


def test_Entity_constructor(ent_sbs):
    assert ent_sbs.size() == 6
    assert len(ent_sbs.uidset) == 6
    assert len(ent_sbs.children) == 7
    assert isinstance(ent_sbs.incidence_dict["I"], UserList)
    assert "I" in ent_sbs
    assert "K" in ent_sbs


def test_Entity_property(ent_hp):
    assert len(ent_hp.uidset) == 7
    assert len(ent_hp.elements) == 7
    assert isinstance(ent_hp.elements["Hufflepuff"], UserList)
    assert ent_hp.is_empty(2) is False
    assert len(ent_hp.incidence_dict["Gryffindor"]) == 6


def test_Entity_attributes(ent_hp):
    assert isinstance(ent_hp.data, np.ndarray)
    assert ent_hp.data.shape == ent_hp.dataframe[ent_hp._data_cols].shape
    assert isinstance(ent_hp.labels, dict)
    assert ent_hp.dimensions == (7, 11, 10, 36, 26)
    assert ent_hp.dimsize == 5
    df = ent_hp.dataframe[ent_hp._data_cols]
    assert list(df.columns) == [
        "House",
        "Blood status",
        "Species",
        "Hair colour",
        "Eye colour",
    ]
    assert ent_hp.dimensions == tuple(df.nunique())
    assert set(ent_hp.labels["House"]) == set(df["House"].unique())


def test_Entity_custom_attributes(ent_hp):
    assert ent_hp.__len__() == 7
    assert isinstance(ent_hp.__str__(), str)
    assert isinstance(ent_hp.__repr__(), str)
    assert isinstance(ent_hp.__contains__("Muggle"), bool)
    assert ent_hp.__contains__("Muggle") is True
    assert ent_hp.__getitem__("Slytherin") == [
        "Half-blood",
        "Pure-blood",
        "Pure-blood or half-blood",
    ]
    assert isinstance(ent_hp.__iter__(), Iterable)
    assert isinstance(ent_hp.__call__(), Iterable)
    assert ent_hp.__call__().__next__() == "Unknown House"


def test_Entity_level(ent_sbs):
    assert ent_sbs.level("I") == (0, 5)
    assert ent_sbs.level("K") == (1, 3)
    assert ent_sbs.level("K", max_level=0) is None


def test_Entity_uidset_by_level(ent_sbs):
    assert ent_sbs.uidset_by_level(0) == {"I", "L", "O", "P", "R", "S"}
    assert ent_sbs.uidset_by_level(1) == {"A", "C", "E", "K", "T1", "T2", "V"}


def test_Entity_elements_by_level(ent_sbs):
    assert ent_sbs.elements_by_level(0, 1)


def test_Entity_incidence_matrix(ent_sbs):
    assert ent_sbs.incidence_matrix(1, 0).todense().shape == (6, 7)


def test_Entity_indices(ent_sbs):
    assert ent_sbs.indices("nodes", "K") == [3]
    assert ent_sbs.indices("nodes", ["K", "T1"]) == [3, 4]


def test_Entity_translate(ent_sbs):
    assert ent_sbs.translate(0, 0) == "P"
    assert ent_sbs.translate(1, [3, 4]) == ["K", "T1"]


def test_Entity_translate_arr(ent_sbs):
    assert ent_sbs.translate_arr((0, 0)) == ["P", "A"]


def test_Entity_index(ent_sbs):
    assert ent_sbs.index("nodes") == 1
    assert ent_sbs.index("nodes", "K") == (1, 3)


def test_restrict_to_levels(ent_hp):
    assert len(ent_hp.restrict_to_levels([0]).uidset) == 7


def test_restrict_to_indices(ent_hp):
    assert ent_hp.restrict_to_indices([1, 2]).uidset == {
        "Gryffindor",
        "Ravenclaw",
    }


def test_Entity_construct_from_entity(sbs):
    ent = Entity(entity=sbs.edgedict)
    assert len(ent.elements) == 6


def test_construct_empty_entity():
    ent = Entity()
    assert ent.empty
    assert ent.is_empty()
    assert len(ent.elements) == 0
    assert ent.dimsize == 0
