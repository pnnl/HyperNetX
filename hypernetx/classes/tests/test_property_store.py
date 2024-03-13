import pytest
import pandas as pd
from pandas import DataFrame
from hypernetx.classes.property_store import PropertyStore, WEIGHT, PROPERTIES, ID, UID


EDGES = 'edges'
NODES = 'nodes'
PROPERTIES_COLUMNS = [WEIGHT, PROPERTIES]
STRENGTH = 'strength'
HAIR_COLOR = 'hair_color'
INCIDENCES_PROPERTIES_COLUMNS = [WEIGHT, PROPERTIES, STRENGTH, HAIR_COLOR]


@pytest.fixture
def edges():
    return ['I', 'L', 'O', 'P', 'R', 'S']


@pytest.fixture
def nodes():
    return ['A', 'C', 'E', 'K', 'T1', 'T2', 'V']


@pytest.fixture
def incidences():
    return [('I', 'K'), ('I', 'T2'), ('L', 'C'), ('L', 'E'), ('O', 'T1'), ('O', 'T2'), ('P', 'A'), ('P', 'C'),
                  ('P', 'K'), ('R', 'A'), ('R', 'E'), ('S', 'A'), ('S', 'K'), ('S', 'T2'), ('S', 'V')]


@pytest.fixture
def edges_df(edges) -> DataFrame:
    data = [(1, {}) for _ in edges]
    index = pd.Index(edges, name=EDGES)
    return DataFrame(data=data, index=index, columns=PROPERTIES_COLUMNS)


@pytest.fixture
def nodes_df(nodes) -> DataFrame:
    data = [(1, {}) for _ in nodes]
    index = pd.Index(nodes, name=NODES)
    return DataFrame(data=data, index=index, columns=PROPERTIES_COLUMNS)


@pytest.fixture
def incidences_df(incidences) -> DataFrame:
    index = pd.MultiIndex.from_tuples(incidences, names=[EDGES, NODES])
    data = [(1, {}, 42, "red") for _ in range(len(index))]
    return DataFrame(data=data, index=index, columns=INCIDENCES_PROPERTIES_COLUMNS)


@pytest.fixture
def edges_ps(edges_df) -> PropertyStore:
    return PropertyStore(edges_df)


@pytest.fixture
def nodes_ps(nodes_df) -> PropertyStore:
    # Uses the index dataframe as the 'uid'
    return PropertyStore(nodes_df, index=True)


@pytest.fixture
def incidences_ps(incidences_df) -> PropertyStore:
    return PropertyStore(incidences_df)


def test_empty_property_store():
    ps = PropertyStore()
    assert len(ps) == 0
    assert ps.properties.columns.tolist() == [ID, WEIGHT, PROPERTIES, UID]


def test_properties_on_edges_ps(edges_ps):
    props = edges_ps.properties
    assert all(weight == 1.0 for weight in props.get(WEIGHT).tolist())
    assert all(prop == dict() for prop in props.get(PROPERTIES).tolist())


def test_properties_on_nodes_ps(nodes_ps, nodes):
    props = nodes_ps.properties
    assert all(weight == 1.0 for weight in props.get(WEIGHT).tolist())
    assert all(prop == dict() for prop in props.get(PROPERTIES).tolist())
    assert all(prop in nodes for prop in props.get(UID))


def test_properties_on_incidences_ps(incidences_ps):
    props = incidences_ps.properties
    assert all(weight == 1.0 for weight in props.get(WEIGHT).tolist())
    assert all(prop == dict() for prop in props.get(PROPERTIES).tolist())
    assert all(prop == 'red' for prop in props.get(HAIR_COLOR).tolist())
    assert all(prop == 42 for prop in props.get(STRENGTH).tolist())


@pytest.mark.parametrize(
    "fixture, uid, expected",
    [
        ("edges_ps", "P", {"weight": 1.0, PROPERTIES: dict()}),
        (
            "incidences_ps",
            ("S", "A"),
            {"weight": 1.0, PROPERTIES: dict(), "hair_color": "red", "strength": 42},
        ),
    ],
)
def test_get_properties(fixture, uid, expected, request):
    ps = request.getfixturevalue(fixture)
    props = ps.get_properties(uid)
    assert expected.items() <= props.items()



@pytest.mark.parametrize(
    "fixture, uid",
    [("edges_ps", "NEMO"), ("incidences_ps", ("NE", "MO"))],
)
def test_get_properties_raises_key_error(fixture, uid, request):
    with pytest.raises(KeyError) as exc_info:
        ps = request.getfixturevalue(fixture)
        ps.get_properties(uid)

    assert f"uid, {uid}, not found in PropertyStore" in str(exc_info.value)


@pytest.mark.parametrize(
    "fixture, uid, prop_name, expected",
    [
        ("edges_ps", "P", WEIGHT, 1.0),
        ("edges_ps", "P", PROPERTIES, dict()),
        ("incidences_ps", ("S", "A"), WEIGHT, 1.0),
        ("incidences_ps", ("S", "A"), PROPERTIES, dict()),
        ("incidences_ps", ("S", "A"), "strength", 42),
        ("incidences_ps", ("S", "A"), "hair_color", "red"),
        ("edges_ps", "P", "NOT A PROPERTY", None),
        ("incidences_ps", ("S", "A"), "not a property", None),
    ],
)
def test_get_property(fixture, uid, prop_name, expected, request):
    ps = request.getfixturevalue(fixture)
    props = ps.get_property(uid, prop_name)
    assert props == expected


@pytest.mark.parametrize(
    "fixture, uid, prop_name",
    [
        ("edges_ps", "NEMO", "weight"),
        ("incidences_ps", ("NE", "MO"), "weight"),
    ],
)
def test_get_property_raises_key_error(fixture, uid, prop_name, request):
    with pytest.raises(KeyError) as exc_info:
        ps = request.getfixturevalue(fixture)
        ps.get_property(uid, prop_name)

    assert f"uid, {uid}, not found in PropertyStore" in str(exc_info.value)


@pytest.mark.parametrize(
    "fixture, uid, prop_name, prop_val, current_props",
    [
        ("edges_ps", "P", "weight", 123.0, 1.0),
        ("edges_ps", "P", "cost", 42.42, None),
        ("incidences_ps", ("S", "A"), "weight", 123.0, 1.0),
        ("incidences_ps", ("S", "A"), "cost", 42.42, None),
        ("incidences_ps", ("S", "A"), "strength", 999, 42),
        ("incidences_ps", ("S", "A"), "hair_color", "blue", "red"),
    ],
)
def test_set_property(fixture, uid, prop_name, prop_val, current_props, request):
    ps = request.getfixturevalue(fixture)

    props = ps.get_property(uid, prop_name)
    assert props == current_props

    ps.set_property(uid, prop_name, prop_val)

    props = ps.get_property(uid, prop_name)

    assert props == prop_val


@pytest.mark.parametrize(
    "fixture, uid, prop_name, prop_val",
    [
        ("edges_ps", "NEMO", "cost", 42.42),
        ("incidences_ps", ("NE", "MO"), "hair_color", "red"),
    ],
)
def test_set_property_raises_key_error(fixture, uid, prop_name, prop_val, request):
    with pytest.raises(KeyError) as exc_info:
        ps = request.getfixturevalue(fixture)
        ps.set_property(uid, prop_name, prop_val)

    assert f"uid, {uid}, not found in PropertyStore" in str(exc_info.value)


@pytest.mark.parametrize(
    "fixture, uids",
    [
        ("edges_ps", "edges"),
        ("incidences_ps", "incidences"),
    ],
)
def test_iter(fixture, uids, request):
    ps = request.getfixturevalue(fixture)
    entities = request.getfixturevalue(uids)
    assert all([uid in entities for uid in ps])


@pytest.mark.parametrize(
    "fixture, expected",
    [
        ("edges_ps", 6),
        ("incidences_ps", 15),
    ],
)
def test_len(fixture, expected, request):
    ps = request.getfixturevalue(fixture)
    assert len(ps) == expected


@pytest.mark.parametrize(
    "fixture, uid, expected",
    [
        ("edges_ps", "P", {"weight": 1.0, PROPERTIES: dict()}),
        (
            "incidences_ps",
            ("S", "A"),
            {"weight": 1.0, PROPERTIES: dict(), "hair_color": "red", "strength": 42},
        ),
    ],
)
def test_getitem(fixture, uid, expected, request):
    ps = request.getfixturevalue(fixture)
    props = ps[uid]
    assert expected.items() <= props.items()


@pytest.mark.parametrize(
    "fixture, uid, expected",
    [
        ("edges_ps", "P", True),
        ("edges_ps", ("NEMO",), False),
        ("incidences_ps", ("S", "A"), True),
        ("incidences_ps", ("NE", "MO"), False),
    ],
)
def test_contains(fixture, uid, expected, request):
    ps = request.getfixturevalue(fixture)
    assert (uid in ps) == expected
