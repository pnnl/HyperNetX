import numpy as np
import pytest

from hypernetx import Entity, EntitySet


def test_construct_entityset_from_empty_dict():
    es = EntitySet({})
    assert len(es.elements) == 0
    assert es.dimsize == 1


@pytest.mark.xfail(reason="default arguments fail for empty Entity")
def test_construct_empty_entityset():
    es = EntitySet()
    assert es.empty
    assert len(es.elements) == 0
    assert es.dimsize == 0


@pytest.mark.xfail(
    reason="at some point we are casting out and back to categorical dtype without preserving categories ordering from `labels` provided to constructor"
)
def test_construct_entityset_from_data(harry_potter):
    es = EntitySet(
        data=np.asarray(harry_potter.data),
        labels=harry_potter.labels,
        level1=1,
        level2=3,
    )
    # TODO: at some point we are casting out and back to categorical dtype without
    #  preserving categories ordering from `labels` provided to constructor
    assert es.indices("Blood status", ["Pure-blood", "Half-blood"]) == [2, 1]  # fails
    assert es.incidence_matrix().shape == (36, 11)
    assert len(es.collapse_identical_elements()) == 11


@pytest.mark.skip(reason="EntitySet from Entity no longer supported")
def test_construct_entityset_from_entity_hp(harry_potter):
    es = EntitySet(
        entity=Entity(data=np.asarray(harry_potter.data), labels=harry_potter.labels),
        level1="Blood status",
        level2="House",
    )
    assert es.indices("Blood status", ["Pure-blood", "Half-blood"]) == [2, 1]
    assert es.incidence_matrix().shape == (7, 11)
    assert len(es.collapse_identical_elements()) == 9


@pytest.mark.skip(reason="EntitySet from Entity no longer supported")
def test_construct_entityset_from_entity(sbs):
    es = EntitySet(entity=Entity(entity=sbs.edgedict))

    assert not es.empty
    assert es.dimsize == 2
    assert es.incidence_matrix().shape == (7, 6)
