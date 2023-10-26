import numpy as np
import pytest

from pytest_lazyfixture import lazy_fixture

from hypernetx.classes import EntitySet


@pytest.mark.parametrize(
    "entity, data, data_cols, labels",
    [
        (lazy_fixture("sbs_dict"), None, (0, 1), None),
        (lazy_fixture("sbs_dict"), None, (0, 1), lazy_fixture("sbs_labels")),
        (lazy_fixture("sbs_dict"), None, ["edges", "nodes"], None),
        (lazy_fixture("sbs_dict"), lazy_fixture("sbs_data"), (0, 1), None),
        (None, lazy_fixture("sbs_data"), (0, 1), lazy_fixture("sbs_labels")),
    ],
)
class TestEntitySBSDict:
    """Tests on different use cases for combination of the following params: entity, data, data_cols, labels"""

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
        elif (labels is not None and not entity) or (
            labels is not None and data
        ):  # labels should match the labels explicitly given
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

    # TODO: validate state of 'data'
    def test_data(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert len(es.data) == 15

    def test_properties(self, entity, data, data_cols, labels, sbs):
        es = EntitySet(entity=entity, data=data, data_cols=data_cols, labels=labels)
        assert (
            es.properties.size == 39
        )  # Properties has three columns and 13 rows of data (i.e. edges + nodes)
        assert list(es.properties.columns) == ["uid", "weight", "properties"]


@pytest.mark.xfail(reason="Deprecated; to be removed in next released")
def test_level(sbs):
    # at some point we are casting out and back to categorical dtype without
    #  preserving categories ordering from `labels` provided to constructor
    ent_sbs = EntitySet(data=np.asarray(sbs.data), labels=sbs.labels)
    assert ent_sbs.level("I") == (0, 5)  # fails
    assert ent_sbs.level("K") == (1, 3)
    assert ent_sbs.level("K", max_level=0) is None
