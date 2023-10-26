import pytest
import numpy as np

from collections.abc import Iterable
from collections import UserList

from hypernetx import EntitySet


class TestEntitySetOnSBSasNDArray:
    def test_ndarray_fail_on_labels(self, sbs_data):
        with pytest.raises(ValueError, match="Labels must be of type Dictionary."):
            EntitySet(data=np.asarray(sbs_data), labels=[])

    def test_ndarray_fail_on_length_labels(self, sbs_data):
        with pytest.raises(
            ValueError,
            match="The length of labels must equal the length of columns in the dataframe.",
        ):
            EntitySet(data=np.asarray(sbs_data), labels=dict())

    def test_dimensions_equal_dimsize(self, sbs_data, sbs_labels):
        ent_sbs = EntitySet(data=np.asarray(sbs_data), labels=sbs_labels)
        assert ent_sbs.dimsize == len(ent_sbs.dimensions)

    def test_translate(self, sbs_data, sbs_labels):
        ent_sbs = EntitySet(data=np.asarray(sbs_data), labels=sbs_labels)
        assert ent_sbs.translate(0, 0) == "P"
        assert ent_sbs.translate(1, [3, 4]) == ["K", "T1"]

    def test_translate_arr(self, sbs_data, sbs_labels):
        ent_sbs = EntitySet(data=np.asarray(sbs_data), labels=sbs_labels)
        assert ent_sbs.translate_arr((0, 0)) == ["P", "A"]

    def test_uidset_by_level(self, sbs_data, sbs_labels):
        ent_sbs = EntitySet(data=np.asarray(sbs_data), labels=sbs_labels)

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


@pytest.mark.xfail(
    reason="Entity does not remove row duplicates from self._data if constructed from np.ndarray, defaults to first two cols as data cols"
)
def test_attributes(harry_potter):
    assert isinstance(harry_potter.data, np.ndarray)
    ent_hp = EntitySet(data=np.asarray(harry_potter.data), labels=harry_potter.labels)
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
