import numpy as np
import pytest
from hypernetx import Entity, EntitySet
from hypernetx import StaticEntity, StaticEntitySet
from hypernetx import HyperNetXError


def test_staticentity_constructor(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.size() == 6
    assert len(ent.uidset) == 6
    assert len(ent.children) == 7
    assert isinstance(ent.incidence_dict['I'], list)
    assert 'I' in ent
    assert 'K' in ent


def test_staticentity_attributes(harry_potter):
    arr = harry_potter.imat
    labels = harry_potter.labels
    ent = StaticEntity(arr=arr, labels=labels)
    assert ent.dimensions == (7, 11, 10, 36, 26)
    assert len(ent.elements) == 7


def test_staticentity_level(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.level('I') == (0, 0)
    assert ent.level('K') == (1, 3)
    assert ent.level('K', max_level=0) == None


def test_staticentity_indices(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.indices('nodes', 'K') == [3]
    assert ent.indices('nodes', ['K', 'T1']) == [3, 4]


def test_staticentity_translate(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.translate(0, 0) == 'I'
    assert ent.translate(1, [3, 4]) == ['K', 'T1']


def test_staticentity_translate_arr(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.translate_arr((0, 0)) == ['I', 'A']


def test_staticentity_index(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.index('nodes') == 1
    assert ent.index('nodes', 'K') == (1, 3)


def test_restrict_to_indices(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
    assert ent.restrict_to_indices([0, 1])


def test_staticentityset(harry_potter):
    arr = harry_potter.imat
    labels = harry_potter.labels
    ent = StaticEntitySet(arr=arr, labels=labels, level1=1, level2=3)
    ent.keys[0] == 'Blood status'
    ent.indices('Blood status', ['Pure-blood', 'Half-blood']) == [2, 1]
    ent.restrict_to([2, 1]).keys[1] == 'Hair colour'


def test_staticentity_construct_from_entity(seven_by_six):
    sbs = seven_by_six
    ent = StaticEntity(arr=sbs.arr, labels=sbs.labels)
