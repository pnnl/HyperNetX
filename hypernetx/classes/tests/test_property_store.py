import pytest
import pandas as pd
from hypernetx.classes.property_store import PropertyStore, DataFramePropertyStore


@pytest.fixture
def edges() -> list[tuple[str]]:
    return [("P",), ("R",), ("S",), ("L",), ("M",), ("N",)]


@pytest.fixture
def df_edges(edges) -> pd.DataFrame:
    uid = edges
    data = {
        "uid": uid,
        "weight": [1.0 for _ in range(len(uid))],
        "properties": [{} for _ in range(len(uid))],
    }
    return pd.DataFrame.from_dict(data)


@pytest.fixture
def dfps_edges(df_edges) -> PropertyStore:
    return DataFramePropertyStore(df_edges)


@pytest.fixture
def incident_pairs() -> list[tuple[str, str]]:
    return [
        ("P", "K"),
        ("P", "A"),
        ("P", "C"),
        ("R", "A"),
        ("R", "E"),
        ("S", "K"),
        ("S", "A"),
        ("S", "V"),
        ("S", "T2"),
        ("L", "E"),
        ("L", "C"),
        ("M", "T2"),
        ("M", "T1"),
        ("N", "K"),
        ("N", "T2"),
    ]


@pytest.fixture
def df_incident_pairs(incident_pairs) -> pd.DataFrame:
    uid = incident_pairs
    data = {
        "uid": uid,
        "weight": [1.0 for _ in range(len(uid))],
        "properties": [{} for _ in range(len(uid))],
        "strength": [42 for _ in range(len(uid))],
        "hair_color": ["red" for _ in range(len(uid))],
    }
    return pd.DataFrame.from_dict(data)


@pytest.fixture
def dfps_incident_pairs(df_incident_pairs) -> PropertyStore:
    return DataFramePropertyStore(df_incident_pairs)


def test_properties(dfps_edges, edges):
    props = dfps_edges.properties
    assert isinstance(props, pd.DataFrame)
    # check that all edges are in 'properties'
    assert all(uid in props.get("uid").tolist() for uid in edges)
    assert all(weight == 1.0 for weight in props.get("weight").tolist())
    assert all(prop == dict() for prop in props.get("properties").tolist())


@pytest.mark.parametrize(
    "fixture, uid, expected",
    [
        ("dfps_edges", ("P",), {"weight": 1.0, "properties": dict()}),
        (
            "dfps_incident_pairs",
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
    [("dfps_edges", ("NEMO",)), ("dfps_incident_pairs", ("NE", "MO"))],
)
def test_get_properties_raises_key_error(fixture, uid, request):
    with pytest.raises(KeyError) as exc_info:
        ps = request.getfixturevalue(fixture)
        ps.get_properties(uid)

    assert (
        str(exc_info.value)
        == f"""'uid, ({",".join(uid)}), not found in PropertyStore'"""
    )


@pytest.mark.parametrize(
    "fixture, uid, prop_name, expected",
    [
        ("dfps_edges", ("P",), "weight", 1.0),
        ("dfps_edges", ("P",), "properties", dict()),
        ("dfps_incident_pairs", ("S", "A"), "weight", 1.0),
        ("dfps_incident_pairs", ("S", "A"), "properties", dict()),
        ("dfps_incident_pairs", ("S", "A"), "strength", 42),
        ("dfps_incident_pairs", ("S", "A"), "hair_color", "red"),
        ("dfps_edges", ("P",), "NOT A PROPERTY", None),
        ("dfps_incident_pairs", ("S", "A"), "not a property", None),
    ],
)
def test_get_property(fixture, uid, prop_name, expected, request):
    ps = request.getfixturevalue(fixture)
    props = ps.get_property(uid, prop_name)
    assert props == expected


@pytest.mark.parametrize(
    "fixture, uid, prop_name",
    [
        ("dfps_edges", ("NEMO",), "weight"),
        ("dfps_incident_pairs", ("NE", "MO"), "weight"),
    ],
)
def test_get_property_raises_key_error(fixture, uid, prop_name, request):
    with pytest.raises(KeyError) as exc_info:
        ps = request.getfixturevalue(fixture)
        ps.get_property(uid, prop_name)

    assert (
        str(exc_info.value)
        == f"""'uid, ({",".join(uid)}), not found in PropertyStore'"""
    )


@pytest.mark.parametrize(
    "fixture, uid, prop_name, prop_val, current_props",
    [
        ("dfps_edges", ("P",), "weight", 123.0, 1.0),
        ("dfps_edges", ("P",), "cost", 42.42, None),
        ("dfps_incident_pairs", ("S", "A"), "weight", 123.0, 1.0),
        ("dfps_incident_pairs", ("S", "A"), "cost", 42.42, None),
        ("dfps_incident_pairs", ("S", "A"), "strength", 999, 42),
        ("dfps_incident_pairs", ("S", "A"), "hair_color", "blue", "red"),
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
        ("dfps_edges", ("NEMO",), "cost", 42.42),
        ("dfps_incident_pairs", ("NE", "MO"), "hair_color", "red"),
    ],
)
def test_set_property_raises_key_error(fixture, uid, prop_name, prop_val, request):
    with pytest.raises(KeyError) as exc_info:
        ps = request.getfixturevalue(fixture)
        ps.set_property(uid, prop_name, prop_val)

    assert (
        str(exc_info.value)
        == f"""'uid, ({",".join(uid)}), not found in PropertyStore'"""
    )


@pytest.mark.parametrize(
    "fixture, uids",
    [
        ("dfps_edges", "edges"),
        ("dfps_incident_pairs", "incident_pairs"),
    ],
)
def test_iter(fixture, uids, request):
    ps = request.getfixturevalue(fixture)
    entities = request.getfixturevalue(uids)
    assert all([i.uid in entities for i in ps])


@pytest.mark.parametrize(
    "fixture, expected",
    [
        ("dfps_edges", 6),
        ("dfps_incident_pairs", 15),
    ],
)
def test_len(fixture, expected, request):
    ps = request.getfixturevalue(fixture)
    assert len(ps) == expected


@pytest.mark.parametrize(
    "fixture, uid, expected",
    [
        ("dfps_edges", ("P",), {"weight": 1.0, "properties": dict()}),
        (
            "dfps_incident_pairs",
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
        ("dfps_edges", ("P",), True),
        ("dfps_edges", ("NEMO",), False),
        ("dfps_incident_pairs", ("S", "A"), True),
        ("dfps_incident_pairs", ("NE", "MO"), False),
    ],
)
def test_contains(fixture, uid, expected, request):
    ps = request.getfixturevalue(fixture)
    assert (uid in ps) == expected
