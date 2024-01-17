from hypernetx.classes.property_store import (
    PropertyStore,
    PropertyStoreFactory,
    DataFramePropertyStore,
    DictPropertyStore,
)


def test_property_store_factory(harry_potter):
    level = 0
    harry_potter.set_random_wandweights()
    data = harry_potter.dataframe

    ps = PropertyStoreFactory.from_dataframe(level, data)

    assert isinstance(ps, DataFramePropertyStore)
