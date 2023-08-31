import numpy as np
import pytest

from collections.abc import Iterable
from collections import UserList
from hypernetx.classes import EntitySet
from hypernetx.classes.entityset import restrict_to_two_columns

from pandas import DataFrame, Series
import pandas as pd


def test_empty_entityset():
    es = EntitySet()
    assert es.empty
    assert len(es.elements) == 0
    assert es.elements == {}
    assert es.dimsize == 0

    assert isinstance(es.data, np.ndarray)
    assert es.data.shape == (0, 0)

    assert es.labels == {}
    assert es.cell_weights == {}
    assert es.isstatic
    assert es.incidence_dict == {}
    assert "foo" not in es
    assert es.incidence_matrix() is None

    # TODO: results in bound method issue
    # assert es.size == 0

    with (pytest.raises(AttributeError)):
        es.get_cell_property("foo", "bar", "roma")
    with (pytest.raises(AttributeError)):
        es.get_cell_properties("foo", "bar")
    with (pytest.raises(KeyError)):
        es.set_cell_property("foo", "bar", "roma", "ff")
    with (pytest.raises(KeyError)):
        es.get_properties("foo")
    # with(pytest.raises(KeyError)):
    #     es.get_property("foo", "bar")
    with (pytest.raises(ValueError)):
        es.set_property("foo", "bar", "roma")


class TestEntitySetOnDataframe:
    def test_cell_properties(self, dataframe_example):
        es = EntitySet(entity=dataframe_example)

        assert es.cell_properties.shape == (3, 1)

    def test_data(self, dataframe_example):
        es = EntitySet(entity=dataframe_example)

        data = es.data

        assert isinstance(data, np.ndarray)
        assert data.shape == (3, 2)
        assert not es.empty
        assert len(es.elements) == 2
        assert es.dimsize == 2
        assert es.uid is None


class TestEntitySetOnSevenBySixDataset:
    # Tests on different inputs for entity and data
    def test_entityset_with_dict(self, sbs):
        ent = EntitySet(entity=sbs.edgedict)
        assert len(ent.elements) == 6

    def test_entityset_with_dict_data_cols(self, sbs):
        ent = EntitySet(entity=sbs.edgedict,  data_cols=["edges", "nodes"])
        assert len(ent.elements) == 6

    def test_entityset_with_ndarray(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)

        assert ent_sbs.size() == 6
        assert len(ent_sbs.uidset) == 6
        assert len(ent_sbs.children) == 7
        assert isinstance(ent_sbs.incidence_dict["I"], list)
        assert "I" in ent_sbs
        assert "K" in ent_sbs

    def test_entityset_with_ndarray_fail_on_labels(self, sbs):
        with (pytest.raises(ValueError, match="Labels must be of type Dictionary.")):
            EntitySet(data=np.asarray(sbs.data), labels=[])

    def test_entityset_with_ndarray_fail_on_length_labels(self, sbs):
        with (pytest.raises(ValueError, match="The length of labels must equal the length of columns in the dataframe.")):
            EntitySet(data=np.asarray(sbs.data), labels=dict())


    # Tests for properties

    @pytest.mark.skip(reason="TODO: implement")
    def test_cell_weights(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_children(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_dataframe(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_dimensions(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_dimsize(self):
        pass

    def test_dimensions_equal_dimsize(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.dimsize == len(ent_sbs.dimensions)

    @pytest.mark.skip(reason="TODO: implement")
    def test_elements(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_empty(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_incidence_dict(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_isstatic(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_labels(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_memberships(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_properties(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_uid(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_uidset(self):
        pass

    # Tests for methods
    @pytest.mark.skip(reason="TODO: implement")
    def test_add(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_add_element(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_add_elements_from(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_assign_properties(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_collapse_identitical_elements(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_elements_by_column(self):
        pass

    def test_elements_by_level(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.elements_by_level(0, 1)

    @pytest.mark.skip(reason="TODO: implement")
    def test_encode(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_get_cell_properties(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_get_cell_property(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_get_properties(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_get_property(self):
        pass

    def test_incidence_matrix(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.incidence_matrix(1, 0).todense().shape == (6, 7)

    def test_index(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.index("nodes") == 1
        assert ent_sbs.index("nodes", "K") == (1, 3)

    def test_indices(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.indices("nodes", "K") == [3]
        assert ent_sbs.indices("nodes", ["K", "T1"]) == [3, 4]

    @pytest.mark.skip(reason="TODO: implement")
    def test_is_empty(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_level(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_remove(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_remove_elements(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_restrict_to(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_restrict_to_indices(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_restrict_to_levels(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_set_cell_property(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_set_property(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_size(self):
        pass

    def test_translate(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.translate(0, 0) == "P"
        assert ent_sbs.translate(1, [3, 4]) == ["K", "T1"]

    def test_translate_arr(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.translate_arr((0, 0)) == ["P", "A"]

    @pytest.mark.skip(reason="TODO: implement")
    def test_uidset_by_column(self):
        pass

    def test_uidset_by_level(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)

        assert ent_sbs.uidset_by_level(0) == {"I", "L", "O", "P", "R", "S"}
        assert ent_sbs.uidset_by_level(1) == {"A", "C", "E", "K", "T1", "T2", "V"}


class TestEntitySetOnHarryPotterDataSet:
    def test_entityset_from_ndarray(self, harry_potter):
        ent_hp = EntitySet(
            data=np.asarray(harry_potter.data), labels=harry_potter.labels
        )
        assert len(ent_hp.uidset) == 7
        assert len(ent_hp.elements) == 7
        assert isinstance(ent_hp.elements["Hufflepuff"], UserList)
        assert not ent_hp.is_empty()
        assert len(ent_hp.incidence_dict["Gryffindor"]) == 6

    def test_custom_attributes(self, harry_potter):
        ent_hp = EntitySet(
            data=np.asarray(harry_potter.data), labels=harry_potter.labels
        )
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

    def test_restrict_to_levels(self, harry_potter):
        ent_hp = EntitySet(
            data=np.asarray(harry_potter.data), labels=harry_potter.labels
        )
        assert len(ent_hp.restrict_to_levels([0]).uidset) == 7

    def test_restrict_to_indices(self, harry_potter):
        ent_hp = EntitySet(
            data=np.asarray(harry_potter.data), labels=harry_potter.labels
        )
        assert ent_hp.restrict_to_indices([1, 2]).uidset == {
            "Gryffindor",
            "Ravenclaw",
        }


# testing entityset helpers


def test_restrict_to_two_columns_on_ndarray(harry_potter):
    data = np.asarray(harry_potter.data)
    labels = harry_potter.labels
    expected_num_cols = 2
    expected_ndarray_first_row = np.array([1, 1])

    entity, data, labels = restrict_to_two_columns(
        entity=None,
        data=data,
        labels=labels,
        cell_properties=None,
        weight_col="cell_weights",
        weights=1,
        level1=0,
        level2=1,
        misc_cell_props_col="properties",
    )

    assert entity is None
    assert len(labels) == 2
    assert 0 in labels
    assert 1 in labels

    print(data)
    print(type(data[0]))

    assert data.shape[1] == expected_num_cols
    assert np.array_equal(data[0], expected_ndarray_first_row)


@pytest.mark.skip(reason="TODO: implement")
def test_restrict_to_two_columns_on_dataframe(sbs):
    pass


@pytest.mark.skip(reason="TODO: implement")
def build_dataframe_from_entity_on_dataframe(sbs):
    pass


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
