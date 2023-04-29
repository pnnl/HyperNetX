import re

import pytest

from hypernetx import Hypergraph
from hypernetx.exception import NWHY_WARNING


@pytest.mark.skip(reason="Deprecated")
def test_get_linegraph_warn_nwhy(sbs):
    H = Hypergraph(sbs.edgedict)
    lg = H.get_linegraph(s=1, use_nwhy=False)
    with pytest.warns(FutureWarning, match=re.escape(NWHY_WARNING)):
        lg_nwhy = H.get_linegraph(s=1, use_nwhy=True)
    assert lg == lg_nwhy


def test_recover_from_state_warn_nwhy():
    with pytest.warns(FutureWarning, match=re.escape(NWHY_WARNING)):
        with pytest.raises(FileNotFoundError):
            Hypergraph.recover_from_state(use_nwhy=True)


@pytest.mark.skip(reason="Deprecated attribute and/or method")
def test_convert_to_static_warn_nwhy(sbs):
    H = Hypergraph(sbs.edgedict, static=False)
    H_static = H.convert_to_static(use_nwhy=False)
    with pytest.warns(FutureWarning, match=re.escape(NWHY_WARNING)):
        H_static_nwhy = H.convert_to_static(use_nwhy=True)

    assert not H_static_nwhy.nwhy
    assert H_static_nwhy.isstatic
    assert H_static.incidence_dict == H_static_nwhy.incidence_dict


@pytest.mark.skip(reason="Deprecated")
@pytest.mark.parametrize(
    "constructor, example",
    [
        (Hypergraph, "sbs_edgedict"),
        (Hypergraph.from_bipartite, "complete_bipartite_example"),
        (Hypergraph.from_numpy_array, "array_example"),
        #  (Hypergraph.from_dataframe, "dataframe_example"),
    ],
)
def test_constructors_warn_nwhy(constructor, example, request):
    example = request.getfixturevalue(example)
    H = constructor(example, use_nwhy=False)
    with pytest.warns(FutureWarning, match=re.escape(NWHY_WARNING)):
        H_nwhy = constructor(example, use_nwhy=True)
    assert not H_nwhy.nwhy
    assert H.incidence_dict == H_nwhy.incidence_dict


@pytest.mark.skip(reason="Deprecated attribute.")
def test_add_nwhy_deprecated(sbs_hypergraph):
    with pytest.deprecated_call():
        Hypergraph.add_nwhy(sbs_hypergraph)
