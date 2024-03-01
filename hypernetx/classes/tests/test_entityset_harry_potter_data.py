import numpy as np
import pytest

from collections.abc import Iterable
from collections import UserList
from hypernetx.classes import EntitySet


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
