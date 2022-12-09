import numpy as np

from collections.abc import Iterable
from collections import UserList

from hypernetx import Entity, EntitySet


def test_Entity_constructor(ent7x6):
    assert ent7x6.size() == 6
    assert len(ent7x6.uidset) == 6
    assert len(ent7x6.children) == 7
    assert isinstance(ent7x6.incidence_dict["I"], UserList)
    assert "I" in ent7x6
    assert "K" in ent7x6


def test_Entity_property(harry_potter_ent):
    assert len(harry_potter_ent.uidset) == 7
    assert len(harry_potter_ent.elements) == 7
    assert isinstance(harry_potter_ent.elements["Hufflepuff"], UserList)
    assert harry_potter_ent.is_empty(2) is False
    assert len(harry_potter_ent.incidence_dict["Gryffindor"]) == 6


def test_Entity_attributes(harry_potter_ent):
    assert isinstance(harry_potter_ent.data, np.ndarray)
    assert (
        harry_potter_ent.data.shape
        == harry_potter_ent.dataframe[harry_potter_ent._data_cols].shape
    )
    assert isinstance(harry_potter_ent.labels, dict)
    assert harry_potter_ent.dimensions == (7, 11, 10, 36, 26)
    assert harry_potter_ent.dimsize == 5
    df = harry_potter_ent.dataframe[harry_potter_ent._data_cols]
    assert list(df.columns) == [
        "House",
        "Blood status",
        "Species",
        "Hair colour",
        "Eye colour",
    ]
    assert harry_potter_ent.dimensions == tuple(df.nunique())
    assert set(harry_potter_ent.labels["House"]) == set(df["House"].unique())


def test_Entity_custom_attributes(harry_potter_ent):
    assert harry_potter_ent.__len__() == 7
    assert isinstance(harry_potter_ent.__str__(), str)
    assert isinstance(harry_potter_ent.__repr__(), str)
    assert isinstance(harry_potter_ent.__contains__("Muggle"), bool)
    assert harry_potter_ent.__contains__("Muggle") is True
    assert harry_potter_ent.__getitem__("Slytherin") == [
        "Half-blood",
        "Pure-blood",
        "Pure-blood or half-blood",
    ]
    assert isinstance(harry_potter_ent.__iter__(), Iterable)
    assert isinstance(harry_potter_ent.__call__(), Iterable)
    assert harry_potter_ent.__call__().__next__() == "Unknown House"


def test_Entity_level(ent7x6):
    assert ent7x6.level("I") == (0, 5)
    assert ent7x6.level("K") == (1, 3)
    assert ent7x6.level("K", max_level=0) is None


def test_Entity_uidset_by_level(ent7x6):
    assert ent7x6.uidset_by_level(0) == {"I", "L", "O", "P", "R", "S"}
    assert ent7x6.uidset_by_level(1) == {"A", "C", "E", "K", "T1", "T2", "V"}


def test_Entity_elements_by_level(ent7x6):
    assert ent7x6.elements_by_level(0, 1)


def test_Entity_incidence_matrix(ent7x6):
    assert ent7x6.incidence_matrix(1, 0).todense().shape == (6, 7)


def test_Entity_indices(ent7x6):
    assert ent7x6.indices("nodes", "K") == [3]
    assert ent7x6.indices("nodes", ["K", "T1"]) == [3, 4]


def test_Entity_translate(ent7x6):
    assert ent7x6.translate(0, 0) == "P"
    assert ent7x6.translate(1, [3, 4]) == ["K", "T1"]


def test_Entity_translate_arr(ent7x6):
    assert ent7x6.translate_arr((0, 0)) == ["P", "A"]


def test_Entity_index(ent7x6):
    assert ent7x6.index("nodes") == 1
    assert ent7x6.index("nodes", "K") == (1, 3)


def test_restrict_to_levels(harry_potter_ent):
    assert len(harry_potter_ent.restrict_to_levels([0]).uidset) == 7


def test_restrict_to_indices(harry_potter_ent):
    assert harry_potter_ent.restrict_to_indices([1, 2]).uidset == {
        "Gryffindor",
        "Ravenclaw",
    }


def test_Entityset(harry_potter):
    data = np.asarray(harry_potter.data)
    labels = harry_potter.labels
    ent = EntitySet(data=data, labels=labels, level1=1, level2=3)
    assert ent.indices("Blood status", ["Pure-blood", "Half-blood"]) == [2, 1]
    assert ent.incidence_matrix().shape == (36, 11)
    assert len(ent.collapse_identical_elements()) == 11


def test_Entity_construct_from_entity(seven_by_six):
    sbs = seven_by_six
    ent = Entity(entity=sbs.edgedict)
    assert len(ent.elements) == 6


def test_construct_empty_entity():
    ent = Entity()
    assert ent.empty
    assert ent.is_empty()
    assert len(ent.elements) == 0
    assert ent.dimsize == 0


def test_construct_empty_entityset():
    ent = EntitySet()
    assert ent.empty
    assert len(ent.elements) == 0
    assert ent.dimsize == 0
