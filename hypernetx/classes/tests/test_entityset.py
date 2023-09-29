import numpy as np
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from collections.abc import Iterable
from collections import UserList
from hypernetx.classes import EntitySet


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

    assert es.size() == 0

    with pytest.raises(AttributeError):
        es.get_cell_property("foo", "bar", "roma")
    with pytest.raises(AttributeError):
        es.get_cell_properties("foo", "bar")
    with pytest.raises(KeyError):
        es.set_cell_property("foo", "bar", "roma", "ff")
    with pytest.raises(KeyError):
        es.get_properties("foo")
    with pytest.raises(KeyError):
        es.get_property("foo", "bar")
    with pytest.raises(ValueError):
        es.set_property("foo", "bar", "roma")


class TestEntitySetOnSevenBySixDataset:
    # Tests on different use cases for combination of the following params: entity, data, data_cols, labels

    @pytest.mark.parametrize(
        "entity, data, data_cols, labels",
        [
            (lazy_fixture("sbs_dataframe"), None, (0, 1), None),
            (lazy_fixture("sbs_dict"), None, (0, 1), None),
            (lazy_fixture("sbs_dict"), None, ["edges", "nodes"], None),
            # (None, lazy_fixture("sbs_data"), (0, 1), lazy_fixture("sbs_labels")),
        ],
    )
    def test_all_attribute_properties_on_common_entityset_instances(
        self, entity, data, data_cols, labels, sbs
    ):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)

        assert es.isstatic
        assert es.uid is None
        assert not es.empty

        assert es.uidset == {"I", "R", "S", "P", "O", "L"}
        assert es.size() == len(sbs.edgedict)
        assert es.dimsize == 2
        assert es.dimensions == (6, 7)
        assert es.data.shape == (15, 2)
        assert es.data.ndim == 2

        assert len(es.elements) == 6
        expected_elements = {
            "I": ["K", "T2"],
            "L": ["E", "C"],
            "O": ["T1", "T2"],
            "P": ["C", "K", "A"],
            "R": ["E", "A"],
            "S": ["K", "V", "A", "T2"],
        }
        for expected_edge, expected_nodes in expected_elements.items():
            assert expected_edge in es.elements
            assert es.elements[expected_edge].sort() == expected_nodes.sort()

        expected_incident_dict = {
            "I": ["K", "T2"],
            "L": ["E", "C"],
            "O": ["T1", "T2"],
            "P": ["C", "K", "A"],
            "R": ["E", "A"],
            "S": ["K", "V", "A", "T2"],
        }
        for expected_edge, expected_nodes in expected_incident_dict.items():
            assert expected_edge in es.incidence_dict
            assert es.incidence_dict[expected_edge].sort() == expected_nodes.sort()

        # check dunder methods
        assert isinstance(es.incidence_dict["I"], list)
        assert "I" in es
        assert "K" in es

        assert es.children == {"C", "T1", "A", "K", "T2", "V", "E"}
        assert es.memberships == {
            "A": ["P", "R", "S"],
            "C": ["P", "L"],
            "E": ["R", "L"],
            "K": ["P", "S", "I"],
            "T1": ["O"],
            "T2": ["S", "O", "I"],
            "V": ["S"],
        }

        assert es.cell_properties.shape == (
            15,
            1,
        )  # cell properties: a pandas dataframe of one column of all the cells. A cell is an edge-node pair. And we are saving the weight of each pair
        assert es.cell_weights == {
            ("P", "C"): 1,
            ("P", "K"): 1,
            ("P", "A"): 1,
            ("R", "E"): 1,
            ("R", "A"): 1,
            ("S", "K"): 1,
            ("S", "V"): 1,
            ("S", "A"): 1,
            ("S", "T2"): 1,
            ("L", "E"): 1,
            ("L", "C"): 1,
            ("O", "T1"): 1,
            ("O", "T2"): 1,
            ("I", "K"): 1,
            ("I", "T2"): 1,
        }

        # check labeling based on given attributes for EntitySet
        if data_cols == [
            "edges",
            "nodes",
        ]:  # labels should use the data_cols as keys for labels
            assert es.labels == {
                "edges": ["I", "L", "O", "P", "R", "S"],
                "nodes": ["A", "C", "E", "K", "T1", "T2", "V"],
            }
        elif labels is not None:  # labels should match the labels explicity given
            assert es.labels == labels
        else:  # if data_cols or labels not given, labels should conform to default format
            assert es.labels == {
                0: ["I", "L", "O", "P", "R", "S"],
                1: ["A", "C", "E", "K", "T1", "T2", "V"],
            }

        # check dataframe
        # size should be the number of rows times the number of columns, i.e 15 x 3
        assert es.dataframe.size == 45

        actual_edge_row0 = es.dataframe.iloc[0, 0]
        actual_node_row0 = es.dataframe.iloc[0, 1]
        actual_cell_weight_row0 = es.dataframe.loc[0, "cell_weights"]

        assert actual_edge_row0 == "P"
        assert actual_node_row0 in ["A", "C", "K"]
        assert actual_cell_weight_row0 == 1

        # print(es.data)
        # print(es.properties)
        assert len(es.data) == 15  # TODO: validate state of 'data'

        assert (
            es.properties.size == 39
        )  # Properties has three columns and 13 rows of data (i.e. edges + nodes)
        assert list(es.properties.columns) == ["uid", "weight", "properties"]

    def test_ndarray_fail_on_labels(self, sbs):
        with pytest.raises(ValueError, match="Labels must be of type Dictionary."):
            EntitySet(data=np.asarray(sbs.data), labels=[])

    def test_ndarray_fail_on_length_labels(self, sbs):
        with pytest.raises(
            ValueError,
            match="The length of labels must equal the length of columns in the dataframe.",
        ):
            EntitySet(data=np.asarray(sbs.data), labels=dict())

    def test_dimensions_equal_dimsize(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.dimsize == len(ent_sbs.dimensions)

    @pytest.mark.parametrize(
        "data",
        [
            pd.DataFrame({0: ["P"], 1: ["E"]}),
            {0: ["P"], 1: ["E"]},
            EntitySet(entity={"P": ["E"]}),
        ],
    )
    def test_add(self, sbs_dataframe, data):
        es = EntitySet(entity=sbs_dataframe)

        assert es.data.shape == (15, 2)
        assert es.dataframe.size == 45

        es.add(data)

        assert es.data.shape == (16, 2)
        assert es.dataframe.size == 48

    def test_remove(self, sbs_dataframe):
        es = EntitySet(entity=sbs_dataframe)
        assert es.data.shape == (15, 2)
        assert es.dataframe.size == 45

        es.remove("P")

        assert es.data.shape == (12, 2)
        assert es.dataframe.size == 36
        assert "P" not in es.elements

    @pytest.mark.parametrize(
        "props, multidx, expected_props",
        [
            (
                lazy_fixture("props_dataframe"),
                (0, "P"),
                {"prop1": "propval1", "prop2": "propval2"},
            ),
            (
                {0: {"P": {"prop1": "propval1", "prop2": "propval2"}}},
                (0, "P"),
                {"prop1": "propval1", "prop2": "propval2"},
            ),
            (
                {1: {"A": {"prop1": "propval1", "prop2": "propval2"}}},
                (1, "A"),
                {"prop1": "propval1", "prop2": "propval2"},
            ),
        ],
    )
    def test_assign_properties(self, sbs_dataframe, props, multidx, expected_props):
        es = EntitySet(entity=sbs_dataframe)

        original_prop = es.properties.loc[multidx]
        assert original_prop.properties == {}

        es.assign_properties(props)

        updated_prop = es.properties.loc[multidx]
        assert updated_prop.properties == expected_props

    @pytest.mark.parametrize(
        "cell_props, multidx, expected_cell_properties",
        [
            (
                lazy_fixture("cell_props_dataframe"),
                ("P", "A"),
                {"prop1": "propval1", "prop2": "propval2"},
            ),
            (
                lazy_fixture("cell_props_dataframe_multidx"),
                ("P", "A"),
                {"prop1": "propval1", "prop2": "propval2"},
            ),
            (
                {"P": {"A": {"prop1": "propval1", "prop2": "propval2"}}},
                ("P", "A"),
                {"prop1": "propval1", "prop2": "propval2"},
            ),
        ],
    )
    def test_assign_cell_properties_on_default_cell_properties(
        self, sbs_dataframe, cell_props, multidx, expected_cell_properties
    ):
        es = EntitySet(entity=sbs_dataframe)

        es.assign_cell_properties(cell_props=cell_props)

        updated_cell_prop = es.cell_properties.loc[multidx]

        assert updated_cell_prop.cell_properties == expected_cell_properties

    def test_assign_cell_properties_on_multiple_properties(self, sbs_dataframe):
        es = EntitySet(entity=sbs_dataframe)
        multidx = ("P", "A")

        es.assign_cell_properties(
            cell_props={"P": {"A": {"prop1": "propval1", "prop2": "propval2"}}}
        )

        updated_cell_prop = es.cell_properties.loc[multidx]
        assert updated_cell_prop.cell_properties == {
            "prop1": "propval1",
            "prop2": "propval2",
        }

        es.assign_cell_properties(
            cell_props={
                "P": {
                    "A": {"prop1": "propval1", "prop2": "propval2", "prop3": "propval3"}
                }
            }
        )

        updated_cell_prop = es.cell_properties.loc[multidx]
        assert updated_cell_prop.cell_properties == {
            "prop1": "propval1",
            "prop2": "propval2",
            "prop3": "propval3",
        }

    @pytest.mark.skip(reason="TODO: implement")
    def test_collapse_identitical_elements(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_elements_by_column(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_level(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_index(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_indices(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_translate(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_translate_arr(self):
        pass

    @pytest.mark.skip(reason="TODO: implement")
    def test_incidence_matrix(self):
        pass

    def test_elements_by_level(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.elements_by_level(0, 1)

    def test_encode(self, sbs_dataframe):
        es = EntitySet()

        df = pd.DataFrame({"Category": ["A", "B", "A", "C", "B"]})
        # Convert 'Category' column to categorical
        df["Category"] = df["Category"].astype("category")

        expected_arr = np.array([[0], [1], [0], [2], [1]])
        actual_arr = es.encode(df)

        assert np.array_equal(actual_arr, expected_arr)

    def test_get_cell_properties(self, sbs_dataframe):
        es = EntitySet(entity=sbs_dataframe)

        props = es.get_cell_properties("P", "A")

        assert props == {"cell_weights": 1}

    def test_get_cell_properties_raises_keyerror(self, sbs_dataframe):
        es = EntitySet(entity=sbs_dataframe)

        with pytest.raises(KeyError, match="cell_properties:"):
            es.get_cell_properties("P", "FOOBAR")

    def test_get_cell_property(self, sbs_dataframe):
        es = EntitySet(entity=sbs_dataframe)
        props = es.get_cell_property("P", "A", "cell_weights")
        assert props == 1

    @pytest.mark.parametrize(
        "item1, item2, prop_name, err_msg",
        [
            ("P", "FOO", "cell_weights", "Item not exists. cell_properties:"),
            (
                "P",
                "A",
                "Not a real property",
                "Item exists but property does not exist. cell_properties:",
            ),
        ],
    )
    def test_get_cell_property_raises_keyerror(
        self, sbs_dataframe, item1, item2, prop_name, err_msg
    ):
        es = EntitySet(entity=sbs_dataframe)

        with pytest.raises(KeyError, match=err_msg):
            es.get_cell_property(item1, item2, prop_name)

    @pytest.mark.parametrize("item, level", [("P", 0), ("P", None), ("A", 1)])
    def test_get_properties(self, sbs_dataframe, item, level):
        es = EntitySet(entity=sbs_dataframe)

        # to avoid duplicate test code, reuse 'level' to get the item_uid
        # but if level is None, assume it to be 0 and that the item exists at level 0
        if level is None:
            item_uid = es.properties.loc[(0, item), "uid"]
        else:
            item_uid = es.properties.loc[(level, item), "uid"]

        props = es.get_properties(item, level=level)

        assert props == {"uid": item_uid, "weight": 1, "properties": {}}

    @pytest.mark.parametrize(
        "item, level, err_msg",
        [
            ("Not a valid item", None, ""),
            ("Not a valid item", 0, "no properties initialized for"),
        ],
    )
    def test_get_properties_raises_keyerror(self, sbs_dataframe, item, level, err_msg):
        es = EntitySet(entity=sbs_dataframe)

        with pytest.raises(KeyError, match=err_msg):
            es.get_properties(item, level=level)

    @pytest.mark.parametrize(
        "item, prop_name, level, expected_prop",
        [
            ("P", "weight", 0, 1),
            ("P", "properties", 0, {}),
            ("P", "uid", 0, 3),
            ("A", "weight", 1, 1),
            ("A", "properties", 1, {}),
            ("A", "uid", 1, 6),
        ],
    )
    def test_get_property(self, sbs_dataframe, item, prop_name, level, expected_prop):
        es = EntitySet(entity=sbs_dataframe)

        prop = es.get_property(item, prop_name, level)

        assert prop == expected_prop

    @pytest.mark.parametrize(
        "item, prop_name, err_msg",
        [
            ("XXX", "weight", "item does not exist:"),
            ("P", "not a real prop name", "no properties initialized for"),
        ],
    )
    def test_get_property_raises_keyerror(
        self, sbs_dataframe, item, prop_name, err_msg
    ):
        es = EntitySet(entity=sbs_dataframe)

        with pytest.raises(KeyError, match=err_msg):
            es.get_property(item, prop_name)

    @pytest.mark.parametrize(
        "item, prop_name, prop_val, level",
        [
            ("P", "weight", 42, 0),
        ],
    )
    def test_set_property(self, sbs_dataframe, item, prop_name, prop_val, level):
        es = EntitySet(entity=sbs_dataframe)

        orig_prop_val = es.get_property(item, prop_name, level)

        es.set_property(item, prop_name, prop_val, level)

        new_prop_val = es.get_property(item, prop_name, level)

        assert new_prop_val != orig_prop_val
        assert new_prop_val == prop_val

    @pytest.mark.parametrize(
        "item, prop_name, prop_val, level, misc_props_col",
        [
            ("P", "new_prop", "foobar", 0, "properties"),
            ("P", "new_prop", "foobar", 0, "some_new_miscellaneaus_col"),
        ],
    )
    def test_set_property_on_non_existing_property(
        self, sbs_dataframe, item, prop_name, prop_val, level, misc_props_col
    ):
        es = EntitySet(entity=sbs_dataframe, misc_props_col=misc_props_col)

        es.set_property(item, prop_name, prop_val, level)

        new_prop_val = es.get_property(item, prop_name, level)

        assert new_prop_val == prop_val

    def test_set_property_raises_keyerror(self, sbs_dataframe):
        es = EntitySet(entity=sbs_dataframe)

        with pytest.raises(
            ValueError, match="cannot infer 'level' when initializing 'item' properties"
        ):
            es.set_property("XXXX", "weight", 42)

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

    @pytest.mark.parametrize("level", [0, 1])
    def test_is_empty(self, sbs_dataframe, level):
        es = EntitySet(entity=sbs_dataframe)
        assert not es.is_empty(level)

    @pytest.mark.skip(reason="TODO: implement")
    def test_level(self):
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
