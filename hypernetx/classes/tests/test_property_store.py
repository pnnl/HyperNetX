import pytest
import pandas as pd
from hypernetx.classes.property_store import PropertyStore, DataFramePropertyStore


EDGE_LEVEL = 'e'
NODE_LEVEL = 'n'
LEVEL = 'level'
ID = 'id'
UUID = 'uuid'
WEIGHT = 'weight'
PROPERTIES = 'properties'
PROPERTIES_COLUMNS = [WEIGHT, PROPERTIES]
STRENGTH = 'strength'
HAIR_COLOR = 'hair_color'
INCIDENCES_PROPERTIES_COLUMNS = [WEIGHT, PROPERTIES, STRENGTH, HAIR_COLOR]


@pytest.fixture
def edges() -> list[tuple[str | int, str | int]]:
    edges = ['I', 'L', 'O', 'P', 'R', 'S']
    return [(EDGE_LEVEL, e) for e in edges]


@pytest.fixture
def nodes():
    nodes = ['A', 'C', 'E', 'K', 'T1', 'T2', 'V']
    return [(NODE_LEVEL, n) for n in nodes]


@pytest.fixture
def incidences():
    return [('I', 'K'), ('I', 'T2'), ('L', 'C'), ('L', 'E'), ('O', 'T1'), ('O', 'T2'), ('P', 'A'), ('P', 'C'),
                  ('P', 'K'), ('R', 'A'), ('R', 'E'), ('S', 'A'), ('S', 'K'), ('S', 'T2'), ('S', 'V')]


@pytest.fixture
def edges_df(edges) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(edges, names=[LEVEL, ID])
    data = [(1, {}) for _ in range(len(index))]
    return pd.DataFrame(data=data, index=index, columns=PROPERTIES_COLUMNS)


@pytest.fixture
def nodes_df(nodes) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(nodes, names=[LEVEL, ID])
    data = [(1, {}) for _ in range(len(index))]
    return pd.DataFrame(data=data, index=index, columns=PROPERTIES_COLUMNS)


@pytest.fixture
def incidences_df(incidences) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(incidences, names=[LEVEL, ID])
    data = [(1, {}, 42, "red") for _ in range(len(index))]
    return pd.DataFrame(data=data, index=index, columns=INCIDENCES_PROPERTIES_COLUMNS)


@pytest.fixture
def edges_dfps(edges_df) -> PropertyStore:
    return DataFramePropertyStore(edges_df)


@pytest.fixture
def nodes_dfps(nodes_df) -> PropertyStore:
    return DataFramePropertyStore(nodes_df)


@pytest.fixture
def incidences_dfps(incidences_df) -> PropertyStore:
    return DataFramePropertyStore(incidences_df)


def test_properties(edges_dfps, edges):
    props = edges_dfps.properties
    assert all(weight == 1.0 for weight in props.get(WEIGHT).tolist())
    assert all(prop == dict() for prop in props.get(PROPERTIES).tolist())


@pytest.mark.parametrize(
    "fixture, uid, expected",
    [
        ("edges_dfps", ("e", "P"), {"weight": 1.0, "properties": dict()}),
        (
            "incidences_dfps",
            ("S", "A"),
            {"weight": 1.0, "properties": dict(), "hair_color": "red", "strength": 42},
        ),
    ],
)
def test_get_properties(fixture, uid, expected, request):
    ps = request.getfixturevalue(fixture)
    props = ps.get_properties(uid)
    assert props == expected


@pytest.mark.parametrize(
    "fixture, uid",
    [("edges_dfps", ("e", "NEMO")), ("incidences_dfps", ("NE", "MO"))],
)
def test_get_properties_raises_key_error(fixture, uid, request):
    with pytest.raises(KeyError) as exc_info:
        ps = request.getfixturevalue(fixture)
        ps.get_properties(uid)

    assert (
        str(exc_info.value)
        == f'"uid, {uid}, not found in PropertyStore"'
    )


@pytest.mark.parametrize(
    "fixture, uid, prop_name, expected",
    [
        ("edges_dfps", ("e", "P",), "weight", 1.0),
        ("edges_dfps", ("e", "P",), "properties", dict()),
        ("incidences_dfps", ("S", "A"), "weight", 1.0),
        ("incidences_dfps", ("S", "A"), "properties", dict()),
        ("incidences_dfps", ("S", "A"), "strength", 42),
        ("incidences_dfps", ("S", "A"), "hair_color", "red"),
        ("edges_dfps", ("e", "P",), "NOT A PROPERTY", None),
        ("incidences_dfps", ("S", "A"), "not a property", None),
    ],
)
def test_get_property(fixture, uid, prop_name, expected, request):
    ps = request.getfixturevalue(fixture)
    props = ps.get_property(uid, prop_name)
    assert props == expected


@pytest.mark.parametrize(
    "fixture, uid, prop_name",
    [
        ("edges_dfps", ("NEMO",), "weight"),
        ("incidences_dfps", ("NE", "MO"), "weight"),
    ],
)
def test_get_property_raises_key_error(fixture, uid, prop_name, request):
    with pytest.raises(KeyError) as exc_info:
        ps = request.getfixturevalue(fixture)
        ps.get_property(uid, prop_name)

    assert (
        str(exc_info.value)
        == f'"uid, {uid}, not found in PropertyStore"'
    )


@pytest.mark.parametrize(
    "fixture, uid, prop_name, prop_val, current_props",
    [
        ("edges_dfps", ("e", "P",), "weight", 123.0, 1.0),
        ("edges_dfps", ("e", "P",), "cost", 42.42, None),
        ("incidences_dfps", ("S", "A"), "weight", 123.0, 1.0),
        ("incidences_dfps", ("S", "A"), "cost", 42.42, None),
        ("incidences_dfps", ("S", "A"), "strength", 999, 42),
        ("incidences_dfps", ("S", "A"), "hair_color", "blue", "red"),
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
        ("edges_dfps", ("NEMO",), "cost", 42.42),
        ("incidences_dfps", ("NE", "MO"), "hair_color", "red"),
    ],
)
def test_set_property_raises_key_error(fixture, uid, prop_name, prop_val, request):
    with pytest.raises(KeyError) as exc_info:
        ps = request.getfixturevalue(fixture)
        ps.set_property(uid, prop_name, prop_val)

    assert (
        str(exc_info.value)
        == f'"uid, {uid}, not found in PropertyStore"'
    )


@pytest.mark.parametrize(
    "fixture, uids",
    [
        ("edges_dfps", "edges"),
        ("incidences_dfps", "incidences"),
    ],
)
def test_iter(fixture, uids, request):
    ps = request.getfixturevalue(fixture)
    entities = request.getfixturevalue(uids)
    assert all([uid in entities for uid in ps])


@pytest.mark.parametrize(
    "fixture, expected",
    [
        ("edges_dfps", 6),
        ("incidences_dfps", 15),
    ],
)
def test_len(fixture, expected, request):
    ps = request.getfixturevalue(fixture)
    assert len(ps) == expected


@pytest.mark.parametrize(
    "fixture, uid, expected",
    [
        ("edges_dfps", ("e", "P",), {"weight": 1.0, "properties": dict()}),
        (
            "incidences_dfps",
            ("S", "A"),
            {"weight": 1.0, "properties": dict(), "hair_color": "red", "strength": 42},
        ),
    ],
)
def test_getitem(fixture, uid, expected, request):
    ps = request.getfixturevalue(fixture)
    props = ps[uid]
    assert props == expected


@pytest.mark.parametrize(
    "fixture, uid, expected",
    [
        ("edges_dfps", ("e", "P",), True),
        ("edges_dfps", ("NEMO",), False),
        ("incidences_dfps", ("S", "A"), True),
        ("incidences_dfps", ("NE", "MO"), False),
    ],
)
def test_contains(fixture, uid, expected, request):
    ps = request.getfixturevalue(fixture)
    assert (uid in ps) == expected
