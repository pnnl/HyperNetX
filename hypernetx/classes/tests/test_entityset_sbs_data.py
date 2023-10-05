import numpy as np
import pandas as pd
import pytest

from pytest_lazyfixture import lazy_fixture

from hypernetx.classes import EntitySet


@pytest.mark.parametrize(
    "entity, data, data_cols, labels",
    [
        (lazy_fixture("sbs_dataframe"), None, (0, 1), None),
        (lazy_fixture("sbs_dict"), None, (0, 1), None),
        (lazy_fixture("sbs_dict"), None, ["edges", "nodes"], None),
        # (None, lazy_fixture("sbs_data"), (0, 1), lazy_fixture("sbs_labels")),
    ],
)
class TestEntitySetUseCasesOnSBS:
    # Tests on different use cases for combination of the following params: entity, data, data_cols, labels

    def test_size(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert es.size() == len(sbs.edgedict)

    # check all the EntitySet properties
    def test_isstatic(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert es.isstatic

    def test_uid(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert es.uid is None

    def test_empty(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert not es.empty

    def test_uidset(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert es.uidset == {"I", "R", "S", "P", "O", "L"}

    def test_dimsize(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert es.dimsize == 2

    def test_elements(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
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

    def test_incident_dict(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
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
        assert isinstance(es.incidence_dict["I"], list)
        assert "I" in es
        assert "K" in es

    def test_children(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert es.children == {"C", "T1", "A", "K", "T2", "V", "E"}

    def test_memberships(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert es.memberships == {
            "A": ["P", "R", "S"],
            "C": ["P", "L"],
            "E": ["R", "L"],
            "K": ["P", "S", "I"],
            "T1": ["O"],
            "T2": ["S", "O", "I"],
            "V": ["S"],
        }

    def test_cell_properties(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert es.cell_properties.shape == (
            15,
            1,
        )

    def test_cell_weights(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
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

    def test_labels(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
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

    def test_dataframe(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        # check dataframe
        # size should be the number of rows times the number of columns, i.e 15 x 3
        assert es.dataframe.size == 45

        actual_edge_row0 = es.dataframe.iloc[0, 0]
        actual_node_row0 = es.dataframe.iloc[0, 1]
        actual_cell_weight_row0 = es.dataframe.loc[0, "cell_weights"]

        assert actual_edge_row0 == "P"
        assert actual_node_row0 in ["A", "C", "K"]
        assert actual_cell_weight_row0 == 1

    def test_data(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert len(es.data) == 15  # TODO: validate state of 'data'

    def test_properties(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert (
            es.properties.size == 39
        )  # Properties has three columns and 13 rows of data (i.e. edges + nodes)
        assert list(es.properties.columns) == ["uid", "weight", "properties"]


class TestEntitySetOnSBSasNDArray:
    # Check all methods
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

    def test_translate(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.translate(0, 0) == "P"
        assert ent_sbs.translate(1, [3, 4]) == ["K", "T1"]

    def test_translate_arr(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.translate_arr((0, 0)) == ["P", "A"]

    def test_uidset_by_level(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)

        assert ent_sbs.uidset_by_level(0) == {"I", "L", "O", "P", "R", "S"}
        assert ent_sbs.uidset_by_level(1) == {"A", "C", "E", "K", "T1", "T2", "V"}


class TestEntitySetOnSBSDataframe:
    @pytest.fixture
    def es_from_sbsdf(self, sbs):
        return EntitySet(entity=sbs.dataframe)

    @pytest.fixture
    def es_from_sbs_dupe_df(self, sbsd):
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
    def test_add(self, es_from_sbsdf, data):
        assert es_from_sbsdf.data.shape == (15, 2)
        assert es_from_sbsdf.dataframe.size == 45

        es_from_sbsdf.add(data)

        assert es_from_sbsdf.data.shape == (16, 2)
        assert es_from_sbsdf.dataframe.size == 48

    def test_remove(self, es_from_sbsdf):
        assert es_from_sbsdf.data.shape == (15, 2)
        assert es_from_sbsdf.dataframe.size == 45

        es_from_sbsdf.remove("P")

        assert es_from_sbsdf.data.shape == (12, 2)
        assert es_from_sbsdf.dataframe.size == 36
        assert "P" not in es_from_sbsdf.elements

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
    def test_assign_properties(self, es_from_sbsdf, props, multidx, expected_props):
        original_prop = es_from_sbsdf.properties.loc[multidx]
        assert original_prop.properties == {}

        es_from_sbsdf.assign_properties(props)

        updated_prop = es_from_sbsdf.properties.loc[multidx]
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
        self, es_from_sbsdf, cell_props, multidx, expected_cell_properties
    ):
        es_from_sbsdf.assign_cell_properties(cell_props=cell_props)

        updated_cell_prop = es_from_sbsdf.cell_properties.loc[multidx]

        assert updated_cell_prop.cell_properties == expected_cell_properties

    def test_assign_cell_properties_on_multiple_properties(self, es_from_sbsdf):
        multidx = ("P", "A")

        es_from_sbsdf.assign_cell_properties(
            cell_props={"P": {"A": {"prop1": "propval1", "prop2": "propval2"}}}
        )

        updated_cell_prop = es_from_sbsdf.cell_properties.loc[multidx]
        assert updated_cell_prop.cell_properties == {
            "prop1": "propval1",
            "prop2": "propval2",
        }

        es_from_sbsdf.assign_cell_properties(
            cell_props={
                "P": {
                    "A": {"prop1": "propval1", "prop2": "propval2", "prop3": "propval3"}
                }
            }
        )

        updated_cell_prop = es_from_sbsdf.cell_properties.loc[multidx]
        assert updated_cell_prop.cell_properties == {
            "prop1": "propval1",
            "prop2": "propval2",
            "prop3": "propval3",
        }

    def test_set_cell_property_on_cell_weights(self, es_from_sbsdf):
        item1 = "P"
        item2 = "A"
        prop_name = "cell_weights"
        prop_val = 42

        es_from_sbsdf.set_cell_property(item1, item2, prop_name, prop_val)

        assert es_from_sbsdf.cell_properties.loc[(item1, item2), prop_name] == 42.0

        # Check that the other cell_weights were not changed and retained the default value of 1
        for row in es_from_sbsdf.cell_properties.itertuples():
            if row.Index != (item1, item2):
                assert row.cell_weights == 1

    def test_set_cell_property_on_non_exisiting_cell_property(self, es_from_sbsdf):
        item1 = "P"
        item2 = "A"
        prop_name = "non_existing_cell_property"
        prop_val = {"foo": "bar"}
        es_from_sbsdf.set_cell_property(item1, item2, prop_name, prop_val)

        assert es_from_sbsdf.cell_properties.loc[(item1, item2), "cell_properties"] == {
            prop_name: prop_val
        }

        # Check that the other rows received the default empty dictionary
        for row in es_from_sbsdf.cell_properties.itertuples():
            if row.Index != (item1, item2):
                assert row.cell_properties == {}

        item2 = "K"
        es_from_sbsdf.set_cell_property(item1, item2, prop_name, prop_val)

        assert es_from_sbsdf.cell_properties.loc[(item1, item2), "cell_properties"] == {
            prop_name: prop_val
        }

    @pytest.mark.parametrize("ret_ec", [True, False])
    def test_collapse_identical_elements_on_duplicates(
        self, es_from_sbs_dupe_df, ret_ec
    ):
        # There are two edges that share the same set of 3 (three) nodes
        new_es = es_from_sbs_dupe_df.collapse_identical_elements(
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
        assert len(es_temp.dataframe) != len(es_from_sbs_dupe_df.dataframe)
        assert len(es_temp.dataframe) == len(es_from_sbs_dupe_df.dataframe) - 3

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
    def test_elements_by_column(self, es_from_sbsdf, col1, col2, expected_elements):
        elements_temps = es_from_sbsdf.elements_by_column(col1, col2)
        actual_elements = {
            elements_temps[k]._key[1]: set(v) for k, v in elements_temps.items()
        }

        assert actual_elements == expected_elements

    def test_elements_by_level(self, sbs):
        ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
        assert ent_sbs.elements_by_level(0, 1)

    def test_encode(self, es_from_sbsdf):
        df = pd.DataFrame({"Category": ["A", "B", "A", "C", "B"]})
        # Convert 'Category' column to categorical
        df["Category"] = df["Category"].astype("category")

        expected_arr = np.array([[0], [1], [0], [2], [1]])
        actual_arr = es_from_sbsdf.encode(df)

        assert np.array_equal(actual_arr, expected_arr)

    def test_get_cell_properties(self, es_from_sbsdf):
        props = es_from_sbsdf.get_cell_properties("P", "A")

        assert props == {"cell_weights": 1}

    def test_get_cell_properties_raises_keyerror(self, es_from_sbsdf):
        with pytest.raises(KeyError, match="cell_properties:"):
            es_from_sbsdf.get_cell_properties("P", "FOOBAR")

    def test_get_cell_property(self, es_from_sbsdf):
        props = es_from_sbsdf.get_cell_property("P", "A", "cell_weights")
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
        self, es_from_sbsdf, item1, item2, prop_name, err_msg
    ):
        with pytest.raises(KeyError, match=err_msg):
            es_from_sbsdf.get_cell_property(item1, item2, prop_name)

    @pytest.mark.parametrize("item, level", [("P", 0), ("P", None), ("A", 1)])
    def test_get_properties(self, es_from_sbsdf, item, level):
        # to avoid duplicate test code, reuse 'level' to get the item_uid
        # but if level is None, assume it to be 0 and that the item exists at level 0
        if level is None:
            item_uid = es_from_sbsdf.properties.loc[(0, item), "uid"]
        else:
            item_uid = es_from_sbsdf.properties.loc[(level, item), "uid"]

        props = es_from_sbsdf.get_properties(item, level=level)

        assert props == {"uid": item_uid, "weight": 1, "properties": {}}

    @pytest.mark.parametrize(
        "item, level, err_msg",
        [
            ("Not a valid item", None, ""),
            ("Not a valid item", 0, "no properties initialized for"),
        ],
    )
    def test_get_properties_raises_keyerror(self, es_from_sbsdf, item, level, err_msg):
        with pytest.raises(KeyError, match=err_msg):
            es_from_sbsdf.get_properties(item, level=level)

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
    def test_get_property(self, es_from_sbsdf, item, prop_name, level, expected_prop):
        prop = es_from_sbsdf.get_property(item, prop_name, level)

        assert prop == expected_prop

    @pytest.mark.parametrize(
        "item, prop_name, err_msg",
        [
            ("XXX", "weight", "item does not exist:"),
            ("P", "not a real prop name", "no properties initialized for"),
        ],
    )
    def test_get_property_raises_keyerror(
        self, es_from_sbsdf, item, prop_name, err_msg
    ):
        with pytest.raises(KeyError, match=err_msg):
            es_from_sbsdf.get_property(item, prop_name)

    @pytest.mark.parametrize(
        "item, prop_name, prop_val, level",
        [
            ("P", "weight", 42, 0),
        ],
    )
    def test_set_property(self, es_from_sbsdf, item, prop_name, prop_val, level):
        orig_prop_val = es_from_sbsdf.get_property(item, prop_name, level)

        es_from_sbsdf.set_property(item, prop_name, prop_val, level)

        new_prop_val = es_from_sbsdf.get_property(item, prop_name, level)

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
        self, es_from_sbsdf, item, prop_name, prop_val, level, misc_props_col
    ):
        es_from_sbsdf.set_property(item, prop_name, prop_val, level)

        new_prop_val = es_from_sbsdf.get_property(item, prop_name, level)

        assert new_prop_val == prop_val

    def test_set_property_raises_keyerror(self, es_from_sbsdf):
        with pytest.raises(
            ValueError, match="cannot infer 'level' when initializing 'item' properties"
        ):
            es_from_sbsdf.set_property("XXXX", "weight", 42)

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
    def test_is_empty(self, es_from_sbsdf, level):
        assert not es_from_sbsdf.is_empty(level)

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
        self, es_from_sbsdf, item_level, item, min_level, max_level, expected_lidx
    ):
        actual_lidx = es_from_sbsdf.level(
            item, min_level=min_level, max_level=max_level
        )

        assert actual_lidx == expected_lidx

        if isinstance(actual_lidx, tuple):
            index_item_in_labels = actual_lidx[1]
            assert index_item_in_labels == es_from_sbsdf.labels[item_level].index(item)


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
