import numpy as np
import pytest

from collections.abc import Iterable
from collections import UserList
from hypernetx.classes import EntitySet

from pandas import DataFrame, Series


def test_empty_entityset():
    es = EntitySet()
    assert es.empty
    assert len(es.elements) == 0
    assert es.elements == {}
    assert es.dimsize == 0
    assert es.uid is None


def test_entityset_from_dataframe():
    data_dict = {
        1: ["A", "D"],
        2: ["A", "C", "D"],
        3: ["D"],
        4: ["A", "B"],
        5: ["B", "C"],
    }

    all_edge_pairs = Series(data_dict).explode()

    entity = DataFrame(
        {"edges": all_edge_pairs.index.to_list(), "nodes": all_edge_pairs.values}
    )

    es = EntitySet(entity=entity)

    assert not es.empty
    assert len(es.elements) == 5
    assert es.dimsize == 2
    assert es.uid is None


## Tests using Seven By Six hypergraphs
def test_entityset_from_dictionary(sbs):
    ent = EntitySet(entity=sbs.edgedict)
    assert len(ent.elements) == 6


def test_entityset_from_ndarray_sbs(sbs):
    ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)

    assert ent_sbs.size() == 6
    assert len(ent_sbs.uidset) == 6
    assert len(ent_sbs.children) == 7
    assert isinstance(ent_sbs.incidence_dict["I"], list)
    assert "I" in ent_sbs
    assert "K" in ent_sbs


def test_uidset_by_level(sbs):
    ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)

    assert ent_sbs.uidset_by_level(0) == {"I", "L", "O", "P", "R", "S"}
    assert ent_sbs.uidset_by_level(1) == {"A", "C", "E", "K", "T1", "T2", "V"}


def test_elements_by_level(sbs):
    ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
    assert ent_sbs.elements_by_level(0, 1)


def test_incidence_matrix(sbs):
    ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
    assert ent_sbs.incidence_matrix(1, 0).todense().shape == (6, 7)


def test_indices(sbs):
    ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
    assert ent_sbs.indices("nodes", "K") == [3]
    assert ent_sbs.indices("nodes", ["K", "T1"]) == [3, 4]


def test_translate(sbs):
    ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
    assert ent_sbs.translate(0, 0) == "P"
    assert ent_sbs.translate(1, [3, 4]) == ["K", "T1"]


def test_translate_arr(sbs):
    ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
    assert ent_sbs.translate_arr((0, 0)) == ["P", "A"]


def test_index(sbs):
    ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
    assert ent_sbs.index("nodes") == 1
    assert ent_sbs.index("nodes", "K") == (1, 3)


@pytest.mark.xfail(
    reason="at some point we are casting out and back to categorical dtype without preserving categories ordering from `labels` provided to constructor"
)
def test_level(sbs):
    # TODO: at some point we are casting out and back to categorical dtype without
    #  preserving categories ordering from `labels` provided to constructor
    ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
    assert ent_sbs.level("I") == (0, 5)  # fails
    assert ent_sbs.level("K") == (1, 3)
    assert ent_sbs.level("K", max_level=0) is None


## Tests using Harry Potter hypergraph
def test_entityset_from_ndarray_harry_potter(harry_potter):
    ent_hp = EntitySet(data=np.asarray(harry_potter.data), labels=harry_potter.labels)
    assert len(ent_hp.uidset) == 7
    assert len(ent_hp.elements) == 7
    assert isinstance(ent_hp.elements["Hufflepuff"], UserList)
    assert not ent_hp.is_empty()
    assert len(ent_hp.incidence_dict["Gryffindor"]) == 6


def test_custom_attributes(harry_potter):
    ent_hp = EntitySet(data=np.asarray(harry_potter.data), labels=harry_potter.labels)
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


def test_restrict_to_levels(harry_potter):
    ent_hp = EntitySet(data=np.asarray(harry_potter.data), labels=harry_potter.labels)
    assert len(ent_hp.restrict_to_levels([0]).uidset) == 7


def test_restrict_to_indices(harry_potter):
    ent_hp = EntitySet(data=np.asarray(harry_potter.data), labels=harry_potter.labels)
    assert ent_hp.restrict_to_indices([1, 2]).uidset == {
        "Gryffindor",
        "Ravenclaw",
    }


@pytest.mark.xfail(
    reason="Entity does not remove row duplicates from self._data if constructed from np.ndarray, defaults to first two cols as data cols"
)
def test_attributes(ent_hp):
    assert isinstance(ent_hp.data, np.ndarray)
    # TODO: Entity does not remove row duplicates from self._data if constructed from np.ndarray
    assert ent_hp.data.shape == ent_hp.dataframe[ent_hp._data_cols].shape  # fails
    assert isinstance(ent_hp.labels, dict)
    # TODO: Entity defaults to first two cols as data cols
    assert ent_hp.dimensions == (7, 11, 10, 36, 26)  # fails
    assert ent_hp.dimsize == 5  # fails
    df = ent_hp.dataframe[ent_hp._data_cols]
    assert list(df.columns) == [  # fails
        "House",
        "Blood status",
        "Species",
        "Hair colour",
        "Eye colour",
    ]
    assert ent_hp.dimensions == tuple(df.nunique())
    assert set(ent_hp.labels["House"]) == set(df["House"].unique())
