import numpy as np
import pytest

from hypernetx.classes import EntitySet


def test_empty_entityset():
    es = EntitySet()
    assert es.empty
    assert len(es.elements) == 0
    assert es.elements == {}
    assert es.dimsize == 0

    assert isinstance(es.data, np.ndarray)
    assert es.data.shape == (0, 0)

    assert es.labels == {}
    assert es.cell_weights == {}
    assert es.isstatic
    assert es.incidence_dict == {}
    assert "foo" not in es
    assert es.incidence_matrix() is None

    assert es.size() == 0

    with pytest.raises(AttributeError):
        es.get_cell_property("foo", "bar", "roma")
    with pytest.raises(AttributeError):
        es.get_cell_properties("foo", "bar")
    with pytest.raises(KeyError):
        es.set_cell_property("foo", "bar", "roma", "ff")
    with pytest.raises(KeyError):
        es.get_properties("foo")
    with pytest.raises(KeyError):
        es.get_property("foo", "bar")
    with pytest.raises(ValueError):
        es.set_property("foo", "bar", "roma")
