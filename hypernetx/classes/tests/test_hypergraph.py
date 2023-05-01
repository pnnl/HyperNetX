import pytest
from hypernetx import Hypergraph

"""
    This test suite runs all the tests on each hypergraph defined by `hyp`
    Write a test based on the following pattern:

    @pytest.mark.parametrize(
        "hyp, expected",
        [
            (pytest.lazy_fixture("hyp_no_props"), <some expected result for this hypergraph>),
            (pytest.lazy_fixture("hyp_df_with_props"), <some expected result for this hypergraph>),
            (pytest.lazy_fixture("hyp_dict_with_props"), <some expected result for this hypergraph>),
            (pytest.lazy_fixture("hyp_props_on_edges_nodes"), <some expected result for this hypergraph>)
        ],
    )
    def test_<some hypergraph method>(hyp: Hypergraph, expected):
        actual = hyp.<some hypergraph method>()
        assert actual == expected
"""


@pytest.mark.parametrize(
    "hyp, expected",
    [
        (pytest.lazy_fixture("hyp_no_props"), None),
        (pytest.lazy_fixture("hyp_df_with_props"), None),
        (pytest.lazy_fixture("hyp_dict_with_props"), None),
        (pytest.lazy_fixture("hyp_props_on_edges_nodes"), None),
    ],
)
def test_dual(hyp: Hypergraph, expected):
    actual = hyp.dual()
    # assertions on the hypergraph
    assert isinstance(actual, Hypergraph)

    # assertions on the actual result compared to the expected result that was defined in the parameterize decorator
    assert actual != expected
