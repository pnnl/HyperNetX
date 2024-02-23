import pytest

import pandas as pd
import numpy as np

from hypernetx import EntitySet


class TestEntitySetOnSBSDataframe:
    @pytest.fixture
    def es_from_df(self, sbs):
        return EntitySet(entity=sbs.dataframe)

    @pytest.fixture
    def es_from_dupe_df(self, sbsd):
        return EntitySet(entity=sbsd.dataframe)

    # check all methods
    @pytest.mark.parametrize(
        "data",
        [
            pd.DataFrame({0: ["P"], 1: ["E"]}),
            {0: ["P"], 1: ["E"]},
            EntitySet(entity={"P": ["E"]}),
        ],
    )
    def test_add(self, es_from_df, data):
        assert es_from_df.data.shape == (15, 2)
        assert es_from_df.dataframe.size == 45

        es_from_df.add(data)

        assert es_from_df.data.shape == (16, 2)
        assert es_from_df.dataframe.size == 48

    def test_remove(self, es_from_df):
        assert es_from_df.data.shape == (15, 2)
        assert es_from_df.dataframe.size == 45

        es_from_df.remove("P")

        assert es_from_df.data.shape == (12, 2)
        assert es_from_df.dataframe.size == 36
        assert "P" not in es_from_df.elements

    @pytest.mark.parametrize(
        "props, multidx, expected_props",
        [
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
    def test_assign_properties(self, es_from_df, props, multidx, expected_props):
        original_prop = es_from_df.properties.loc[multidx]
        assert original_prop.properties == {}

        es_from_df.assign_properties(props)

        updated_prop = es_from_df.properties.loc[multidx]
        assert updated_prop.properties == expected_props

    @pytest.mark.parametrize(
        "cell_props, multidx, expected_cell_properties",
        [
            (
                {"P": {"A": {"prop1": "propval1", "prop2": "propval2"}}},
                ("P", "A"),
                {"prop1": "propval1", "prop2": "propval2"},
            ),
        ],
    )
    def test_assign_cell_properties_on_default_cell_properties(
        self, es_from_df, cell_props, multidx, expected_cell_properties
    ):
        es_from_df.assign_cell_properties(cell_props=cell_props)

        updated_cell_prop = es_from_df.cell_properties.loc[multidx]

        assert updated_cell_prop.cell_properties == expected_cell_properties

    def test_assign_cell_properties_on_multiple_properties(self, es_from_df):
        multidx = ("P", "A")

        es_from_df.assign_cell_properties(
            cell_props={"P": {"A": {"prop1": "propval1", "prop2": "propval2"}}}
        )

        updated_cell_prop = es_from_df.cell_properties.loc[multidx]
        assert updated_cell_prop.cell_properties == {
            "prop1": "propval1",
            "prop2": "propval2",
        }

        es_from_df.assign_cell_properties(
            cell_props={
                "P": {
                    "A": {"prop1": "propval1", "prop2": "propval2", "prop3": "propval3"}
                }
            }
        )

        updated_cell_prop = es_from_df.cell_properties.loc[multidx]
        assert updated_cell_prop.cell_properties == {
            "prop1": "propval1",
            "prop2": "propval2",
            "prop3": "propval3",
        }

    def test_set_cell_property_on_cell_weights(self, es_from_df):
        item1 = "P"
        item2 = "A"
        prop_name = "cell_weights"
        prop_val = 42

        es_from_df.set_cell_property(item1, item2, prop_name, prop_val)

        assert es_from_df.cell_properties.loc[(item1, item2), prop_name] == 42.0

        # Check that the other cell_weights were not changed and retained the default value of 1
        for row in es_from_df.cell_properties.itertuples():
            if row.Index != (item1, item2):
                assert row.cell_weights == 1

    def test_set_cell_property_on_non_exisiting_cell_property(self, es_from_df):
        item1 = "P"
        item2 = "A"
        prop_name = "non_existing_cell_property"
        prop_val = {"foo": "bar"}
        es_from_df.set_cell_property(item1, item2, prop_name, prop_val)

        assert es_from_df.cell_properties.loc[(item1, item2), "cell_properties"] == {
            prop_name: prop_val
        }

        # Check that the other rows received the default empty dictionary
        for row in es_from_df.cell_properties.itertuples():
            if row.Index != (item1, item2):
                assert row.cell_properties == {}

        item2 = "K"
        es_from_df.set_cell_property(item1, item2, prop_name, prop_val)

        assert es_from_df.cell_properties.loc[(item1, item2), "cell_properties"] == {
            prop_name: prop_val
        }

    @pytest.mark.parametrize("ret_ec", [True, False])
    def test_collapse_identical_elements_on_duplicates(self, es_from_dupe_df, ret_ec):
        # There are two edges that share the same set of 3 (three) nodes
        new_es = es_from_dupe_df.collapse_identical_elements(
            return_equivalence_classes=ret_ec
        )

        es_temp = new_es
        if isinstance(new_es, tuple):
            # reset variable for actual EntitySet
            es_temp = new_es[0]

            # check equiv classes
            collapsed_edge_key = "L: 2"
            assert "M: 2" not in es_temp.elements
            assert collapsed_edge_key in es_temp.elements
            assert set(es_temp.elements.get(collapsed_edge_key)) == {"F", "C", "E"}

            equiv_classes = new_es[1]
            assert equiv_classes == {
                "I: 1": ["I"],
                "L: 2": ["L", "M"],
                "O: 1": ["O"],
                "P: 1": ["P"],
                "R: 1": ["R"],
                "S: 1": ["S"],
            }

        # check dataframe
        assert len(es_temp.dataframe) != len(es_from_dupe_df.dataframe)
        assert len(es_temp.dataframe) == len(es_from_dupe_df.dataframe) - 3

    @pytest.mark.parametrize(
        "col1, col2, expected_elements",
        [
            (
                0,
                1,
                {
                    "I": {"K", "T2"},
                    "L": {"C", "E"},
                    "O": {"T1", "T2"},
                    "P": {"K", "A", "C"},
                    "R": {"A", "E"},
                    "S": {"K", "A", "V", "T2"},
                },
            ),
            (
                1,
                0,
                {
                    "A": {"P", "R", "S"},
                    "C": {"P", "L"},
                    "E": {"R", "L"},
                    "K": {"P", "S", "I"},
                    "T1": {"O"},
                    "T2": {"S", "O", "I"},
                    "V": {"S"},
                },
            ),
        ],
    )
    def test_elements_by_column(self, es_from_df, col1, col2, expected_elements):
        elements_temps = es_from_df.elements_by_column(col1, col2)
        actual_elements = {
            elements_temps[k]._key[1]: set(v) for k, v in elements_temps.items()
        }

        assert actual_elements == expected_elements

    def test_elements_by_level(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.elements_by_level(0, 1)

    def test_encode(self, es_from_df):
        df = pd.DataFrame({"Category": ["A", "B", "A", "C", "B"]})
        # Convert 'Category' column to categorical
        df["Category"] = df["Category"].astype("category")

        expected_arr = np.array([[0], [1], [0], [2], [1]])
        actual_arr = es_from_df.encode(df)

        assert np.array_equal(actual_arr, expected_arr)

    def test_get_cell_properties(self, es_from_df):
        props = es_from_df.get_cell_properties("P", "A")

        assert props == {"cell_weights": 1}

    def test_get_cell_properties_raises_keyerror(self, es_from_df):
        assert es_from_df.get_cell_properties("P", "FOOBAR") is None

    def test_get_cell_property(self, es_from_df):
        props = es_from_df.get_cell_property("P", "A", "cell_weights")
        assert props == 1

    @pytest.mark.parametrize(
        "item1, item2, prop_name, err_msg",
        [
            ("P", "FOO", "cell_weights", "Item not exists. cell_properties:"),
        ],
    )
    def test_get_cell_property_raises_keyerror(
        self, es_from_df, item1, item2, prop_name, err_msg
    ):
        with pytest.raises(KeyError, match=err_msg):
            es_from_df.get_cell_property(item1, item2, prop_name)

    def test_get_cell_property_returns_none_on_prop(self, es_from_df):
        assert es_from_df.get_cell_property("P", "A", "Not a real property") is None

    @pytest.mark.parametrize("item, level", [("P", 0), ("P", None), ("A", 1)])
    def test_get_properties(self, es_from_df, item, level):
        # to avoid duplicate test code, reuse 'level' to get the item_uid
        # but if level is None, assume it to be 0 and that the item exists at level 0
        if level is None:
            item_uid = es_from_df.properties.loc[(0, item), "uid"]
        else:
            item_uid = es_from_df.properties.loc[(level, item), "uid"]

        props = es_from_df.get_properties(item, level=level)

        assert props == {"uid": item_uid, "weight": 1, "properties": {}}

    @pytest.mark.parametrize(
        "item, level, err_msg",
        [
            ("Not a valid item", None, ""),
            ("Not a valid item", 0, "no properties initialized for"),
        ],
    )
    def test_get_properties_raises_keyerror(self, es_from_df, item, level, err_msg):
        with pytest.raises(KeyError, match=err_msg):
            es_from_df.get_properties(item, level=level)

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
    def test_get_property(self, es_from_df, item, prop_name, level, expected_prop):
        prop = es_from_df.get_property(item, prop_name, level)

        assert prop == expected_prop

    @pytest.mark.parametrize(
        "item, prop_name, err_msg",
        [
            ("XXX", "weight", "item does not exist:"),
        ],
    )
    def test_get_property_raises_keyerror(self, es_from_df, item, prop_name, err_msg):
        with pytest.raises(KeyError, match=err_msg):
            es_from_df.get_property(item, prop_name)

    def test_get_property_returns_none_on_no_property(self, es_from_df):
        assert es_from_df.get_property("P", "non-existing property") is None

    @pytest.mark.parametrize(
        "item, prop_name, prop_val, level",
        [
            ("P", "weight", 42, 0),
        ],
    )
    def test_set_property(self, es_from_df, item, prop_name, prop_val, level):
        orig_prop_val = es_from_df.get_property(item, prop_name, level)

        es_from_df.set_property(item, prop_name, prop_val, level)

        new_prop_val = es_from_df.get_property(item, prop_name, level)

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
        self, es_from_df, item, prop_name, prop_val, level, misc_props_col
    ):
        es_from_df.set_property(item, prop_name, prop_val, level)

        new_prop_val = es_from_df.get_property(item, prop_name, level)

        assert new_prop_val == prop_val

    def test_set_property_raises_keyerror(self, es_from_df):
        with pytest.raises(
            ValueError, match="cannot infer 'level' when initializing 'item' properties"
        ):
            es_from_df.set_property("XXXX", "weight", 42)

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
    def test_is_empty(self, es_from_df, level):
        assert not es_from_df.is_empty(level)

    @pytest.mark.parametrize(
        "item_level, item, min_level, max_level, expected_lidx",
        [
            (0, "P", 0, None, (0, 3)),
            (0, "P", 0, 0, (0, 3)),
            (0, "P", 1, 1, None),
            (1, "A", 0, None, (1, 0)),
            (1, "A", 0, 0, None),
            (1, "K", 0, None, (1, 3)),
        ],
    )
    def test_level(
        self, es_from_df, item_level, item, min_level, max_level, expected_lidx
    ):
        actual_lidx = es_from_df.level(item, min_level=min_level, max_level=max_level)

        assert actual_lidx == expected_lidx

        if isinstance(actual_lidx, tuple):
            index_item_in_labels = actual_lidx[1]
            assert index_item_in_labels == es_from_df.labels[item_level].index(item)
