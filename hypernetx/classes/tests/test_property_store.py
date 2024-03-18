import pytest
import pandas as pd
from pandas import DataFrame
from hypernetx.classes.property_store import PropertyStore, WEIGHT, MISC_PROPERTIES


EDGES = "edges"
NODES = "nodes"
PROPERTIES_COLUMNS = [WEIGHT, MISC_PROPERTIES]
STRENGTH = "strength"
HAIR_COLOR = "hair_color"
INCIDENCES_PROPERTIES_COLUMNS = [WEIGHT, MISC_PROPERTIES, STRENGTH, HAIR_COLOR]
MISC_PROPERTIES_NESTED = {
    "status": "married",
    "education": {
        "undergraduate": "University of Washington",
        "graduate": "Northeastern University",
        "vocational": "ITT Tech",
    },
}
MISC_PROPERTIES_FLATTENED = {
    "graduate": "Northeastern University",
    "status": "married",
    "undergraduate": "University of Washington",
    "vocational": "ITT Tech",
}
INCIDENCE_PROPERTIES_FLATTENED = {
    "graduate": "Northeastern University",
    "hair_color": "red",
    "status": "married",
    "strength": 42.0,
    "undergraduate": "University of Washington",
    "vocational": "ITT Tech",
    "weight": 1.23,
}


@pytest.fixture
def edges():
    return ["I", "L", "O", "P", "R", "S"]


@pytest.fixture
def nodes():
    return ["A", "C", "E", "K", "T1", "T2", "V"]


@pytest.fixture
def incidences():
    return [
        ("I", "K"),
        ("I", "T2"),
        ("L", "C"),
        ("L", "E"),
        ("O", "T1"),
        ("O", "T2"),
        ("P", "A"),
        ("P", "C"),
        ("P", "K"),
        ("R", "A"),
        ("R", "E"),
        ("S", "A"),
        ("S", "K"),
        ("S", "T2"),
        ("S", "V"),
    ]


@pytest.fixture
def edges_df(edges) -> DataFrame:
    # get a subset of edges that will have properties
    data = [(edge, 43.0, {}) for edge in edges[:4]]
    df = DataFrame(data=data, columns=[EDGES, WEIGHT, MISC_PROPERTIES])
    df.set_index(EDGES, inplace=True)
    return df


@pytest.fixture
def nodes_df(nodes) -> DataFrame:
    data = [(node, float(idx), {}) for idx, node in enumerate(nodes)]
    df = DataFrame(data=data, columns=[NODES, WEIGHT, MISC_PROPERTIES])
    df.set_index(NODES, inplace=True)
    return df


@pytest.fixture
def incidences_df(incidences) -> DataFrame:
    misc_properties = {
        "status": "married",
        "education": {
            "undergraduate": "University of Washington",
            "graduate": "Northeastern University",
            "vocational": "ITT Tech",
        },
    }
    index = pd.MultiIndex.from_tuples(incidences[:9], names=[EDGES, NODES])
    data = [(1.23, misc_properties, 42.0, "red") for _ in range(len(index))]
    return DataFrame(
        data=data, index=index, columns=[WEIGHT, MISC_PROPERTIES, STRENGTH, HAIR_COLOR]
    )


@pytest.fixture
def edges_ps(edges_df) -> PropertyStore:
    return PropertyStore(edges_df)


@pytest.fixture
def nodes_ps(nodes_df) -> PropertyStore:
    return PropertyStore(nodes_df, default_weight=42.0)


@pytest.fixture
def incidences_ps(incidences_df) -> PropertyStore:
    return PropertyStore(incidences_df, default_weight=3.33)


def test_empty_property_store():
    ps = PropertyStore()
    assert ps.properties.columns.tolist() == [WEIGHT, MISC_PROPERTIES]


def test_properties_on_edges_ps(edges_ps):
    props: DataFrame = edges_ps.properties

    assert all(weight == 43.0 for weight in props.get(WEIGHT).tolist())
    assert all(prop == dict() for prop in props.get(MISC_PROPERTIES).tolist())


def test_properties_on_incidences_ps(incidences_ps):
    props: DataFrame = incidences_ps.properties

    assert all(weight == 1.23 for weight in props.get(WEIGHT).tolist())
    assert all(
        prop == MISC_PROPERTIES_NESTED for prop in props.get(MISC_PROPERTIES).tolist()
    )
    assert all(prop == "red" for prop in props.get(HAIR_COLOR).tolist())
    assert all(prop == 42 for prop in props.get(STRENGTH).tolist())


@pytest.mark.parametrize(
    "property_store, uid, expected",
    [
        # edge has properties
        ("edges_ps", "P", {"weight": 43.0, "misc_properties": dict()}),
        # edge does not currently have any properties, uses defaults
        ("edges_ps", "R", {"weight": 1.0, "misc_properties": dict()}),
        # incidence has properties
        (
            "incidences_ps",
            ("I", "K"),
            {
                "weight": 1.23,
                "hair_color": "red",
                "strength": 42.0,
                **INCIDENCE_PROPERTIES_FLATTENED,
            },
        ),
        # incidence does not currently have any properties
        ("incidences_ps", ("R", "A"), {"weight": 3.33, "misc_properties": dict()}),
    ],
)
def test_get_properties(property_store, uid, expected, request):
    ps = request.getfixturevalue(property_store)

    props = ps.get_properties(uid)

    assert props == expected


@pytest.mark.parametrize(
    "property_store, uid, expected",
    [
        # edge has properties
        ("edges_ps", "P", {"weight": 43.0, "misc_properties": dict()}),
        # edge does not currently have any properties, uses defaults
        ("edges_ps", "R", {"weight": 1.0, "misc_properties": dict()}),
        # incidence has properties
        (
            "incidences_ps",
            ("I", "K"),
            {
                "weight": 1.23,
                "hair_color": "red",
                "strength": 42.0,
                **INCIDENCE_PROPERTIES_FLATTENED,
            },
        ),
        # incidence does not currently have any properties
        ("incidences_ps", ("R", "A"), {"weight": 3.33, "misc_properties": dict()}),
    ],
)
def test_getitem(property_store, uid, expected, request):
    ps = request.getfixturevalue(property_store)

    props = ps[uid]

    assert props == expected


@pytest.mark.parametrize(
    "property_store, uid, prop_name, expected",
    [
        # edge does have properties
        ("edges_ps", "P", WEIGHT, 43.0),
        ("edges_ps", "P", MISC_PROPERTIES, dict()),
        ("edges_ps", "P", "NOT A PROPERTY", None),
        # edge does not currently have any properties, uses defaults
        ("edges_ps", "S", WEIGHT, 1.0),
        ("edges_ps", "S", MISC_PROPERTIES, dict()),
        ("edges_ps", "S", "NOT A PROPERTY", None),
        # incidence does have properties
        ("incidences_ps", ("I", "K"), WEIGHT, 1.23),
        ("incidences_ps", ("I", "K"), MISC_PROPERTIES, MISC_PROPERTIES_FLATTENED),
        ("incidences_ps", ("I", "K"), "strength", 42),
        ("incidences_ps", ("I", "K"), "hair_color", "red"),
        ("incidences_ps", ("I", "K"), "not a property", None),
        # incidence does not have any properties, uses defaults
        ("incidences_ps", ("S", "A"), WEIGHT, 3.33),
        ("incidences_ps", ("S", "A"), MISC_PROPERTIES, dict()),
        ("incidences_ps", ("S", "A"), "not a property", None),
    ],
)
def test_get_property(property_store, uid, prop_name, expected, request):
    ps = request.getfixturevalue(property_store)
    props = ps.get_property(uid, prop_name)
    assert props == expected


@pytest.mark.parametrize(
    "fixture, uid, prop_name, prop_val, current_props",
    [
        # edge has property and updates it
        ("edges_ps", "P", "weight", 3.14, 43.0),
        # edge does not have property and adds it
        ("edges_ps", "P", "cost", 2.99, None),
        # edge has property (default value) and updates it
        ("edges_ps", "R", "weight", 78.9, 1.0),
        # edge does not have property and adds it
        ("edges_ps", "R", "cost", 2.99, None),
        # incidence has property and updates it
        ("incidences_ps", ("I", "K"), "weight", 93.3, 1.23),
        ("incidences_ps", ("I", "K"), "strength", 999, 42),
        ("incidences_ps", ("I", "K"), "hair_color", "blue", "red"),
        # incidence has property (default) and updates it
        ("incidences_ps", ("S", "A"), "weight", 0.46, 3.33),
        # incidence has does not have property (default) and adds it
        ("incidences_ps", ("S", "A"), "cost", 42.42, None),
        ("incidences_ps", ("S", "A"), "strength", 999, None),
        ("incidences_ps", ("S", "A"), "hair_color", "blue", None),
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
    "fixture, uid, expected",
    [
        ("edges_ps", "P", True),
        ("edges_ps", ("NEMO",), False),
        ("incidences_ps", ("I", "K"), True),
        ("incidences_ps", ("NE", "MO"), False),
    ],
)
def test_contains(fixture, uid, expected, request):
    ps = request.getfixturevalue(fixture)
    assert (uid in ps) == expected
