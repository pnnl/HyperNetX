import numpy as np
import pandas as pd
import pytest
from collections.abc import Iterable
from hypernetx import Entity, EntitySet
from hypernetx import StaticEntity, StaticEntitySet
from hypernetx import HyperNetXError


def test_staticentity_constructor(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.size() == 6
    assert len(ent.uidset) == 6
    assert len(ent.children) == 7
    assert isinstance(ent.incidence_dict["I"], list)
    assert "I" in ent
    assert "K" in ent


def test_staticentity_attributes(harry_potter):
    arr = harry_potter.arr
    labels = harry_potter.labels
    ent = StaticEntity(arr=arr, labels=labels)
    assert isinstance(ent.arr, np.ndarray)
    assert isinstance(ent.data, np.ndarray)
    assert ent.data.shape == ent.dataframe.shape
    assert isinstance(ent.labels, dict)
    assert isinstance(ent.array_with_counts, np.ndarray)
    # uid??
    assert len(ent.uidset) == 7
    assert isinstance(ent.uidset, frozenset)
    assert len(ent.elements) == 7
    assert isinstance(ent.elements, dict)
    assert isinstance(ent.elements['Hufflepuff'], list)
    assert len(ent.incidence_dict['Gryffindor']) == 6
    assert ent.dimensions == (7, 11, 10, 36, 26)
    assert ent.dimsize == 5
    assert len(ent.keys) == 5
    assert ent.keyindex('House') == 0
    assert ent.is_empty(2) == False
    assert len(ent.labs(0)) == 7
    assert isinstance(ent.children, set)
    assert isinstance(ent.dataframe, pd.DataFrame)


def test_staticentity_custom_attributes(harry_potter):
    arr = harry_potter.arr
    labels = harry_potter.labels
    ent = StaticEntity(arr=arr, labels=labels)
    assert ent.__len__() == 7
    assert isinstance(ent.__str__(), str)
    assert isinstance(ent.__repr__(), str)
    assert isinstance(ent.__contains__('Muggle'), bool)
    assert ent.__contains__('Muggle') == True
    assert ent.__getitem__('Slytherin') == ['Half-blood', 'Pure-blood', 'Pure-blood or half-blood']
    assert isinstance(ent.__iter__(), Iterable)
    assert isinstance(ent.__call__(), Iterable)


def test_staticentity_level(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.level("I") == (0, 0)
    assert ent.level("K") == (1, 3)
    assert ent.level("K", max_level=0) == None


def test_staticentity_uidset_by_level(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    ent.uidset_by_level(0) == {'I', 'L', 'O', 'P', 'R', 'S'}
    ent.uidset_by_level(1) == {'A', 'C', 'E', 'K', 'T1', 'T2', 'V'}


def test_staticentity_elements_by_level(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.elements_by_level(0)


def test_staticentity_incidence_matrix(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.incidence_matrix(1, 0).todense().shape == (6, 7)


def test_staticentity_indices(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.indices("nodes", "K") == [3]
    assert ent.indices("nodes", ["K", "T1"]) == [3, 4]


def test_staticentity_translate(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.translate(0, 0) == "I"
    assert ent.translate(1, [3, 4]) == ["K", "T1"]


def test_staticentity_translate_arr(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.translate_arr((0, 0)) == ["I", "A"]


def test_staticentity_index(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.index("nodes") == 1
    assert ent.index("nodes", "K") == (1, 3)


def test_staticentity_turn_entity_data_into_dataframe(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    subset = ent.data[0:5]
    assert ent.turn_entity_data_into_dataframe(subset).shape == (5, 2)


def test_restrict_to_levels(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.restrict_to_levels([0, 1])


def test_restrict_to_indices(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.restrict_to_indices([0, 1])


def test_staticentityset(harry_potter):
    arr = harry_potter.arr
    labels = harry_potter.labels
    ent = StaticEntitySet(arr=arr, labels=labels, level1=1, level2=3)
    assert ent.keys[0] == "Blood status"
    ent.indices("Blood status", ["Pure-blood", "Half-blood"]) == [2, 1]
    assert ent.restrict_to([2, 1]).keys[1] == "Hair colour"
    assert ent.incidence_matrix().shape == (36, 11)
    assert isinstance(ent.convert_to_entityset('Hair colour'), EntitySet)


def test_staticentity_construct_from_entity(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(entity=sbs.edgedict)
    assert len(ent.elements) == 6


