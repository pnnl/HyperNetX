import pandas as pd

from typing import Any
from collections.abc import Hashable

from pandas import DataFrame


UID = "uid"
WEIGHT = "weight"
MISC_PROPERTIES = "misc_properties"


class PropertyStore:
    """Class for storing properties of a collection of edges, nodes, or incidences.

    Properties will be stored in a pandas dataframe;

    """

    def __init__(self, data=None, default_weight=1.0):
        """
        Parameters
        ----------
        data: DataFrame
            optional parameter that holds the properties data in a Dataframe or MultiIndex Dataframe of the following shape

            DataFrame index is uid of objects.

            Example of dataframe (uid is set as the index):

            uid        | weight | misc_properties | <additional property> | ...
            <edge1uid> | 1.0    | {}              | <property value>      | ...

            uid         | weight | misc_properties | <additional property> | ...
            <node1uid>  | 1.0    | {}              | <property value>      | ...

            Example of multiIndex dataframe (edgeid and nodeid are set as the multiIndex):

             level     |  id         | weight | misc_properties | <additional property> | ...
            <edge1uid> | <node1uid>  | 1.0    | {}              | <property value>      | ...
                       | <node2uid>  | 1.5    | {}              | <property value>      | ...
            <edge2uid> | <node1uid>  | 1.0    | {}              | <property value>      | ...
                       | <node3uid>  | 1.0    | {}              | <property value>      | ...

        """
        # If no dataframe is provided, create an empty dataframe
        if data is None:
            self._data: DataFrame = DataFrame(columns=[UID, WEIGHT, MISC_PROPERTIES])
            self._data.set_index(UID, inplace=True)
        else:
            self._data: DataFrame = data

        self._default_weight = default_weight

    @property
    def properties(self) -> DataFrame:
        """Properties assigned to all items in the underlying data table

        Returns
        -------
        out: pandas.DataFrame
            a dataframe with the following columns: level, id, uid, weight, properties, <optional props>
        """
        return self._data

    def get_properties(self, uid) -> dict:
        """Get all properties of an item

        Parameters
        ----------
        uid: Hashable
            uid is the index used to fetch all its properties

        Returns
        -------
        out : dict
            Output dictionary containing all properties of the uid.
            ``{named property: property value, ...,
            properties: {property name: property value}}``

        See Also
        --------
        get_property, set_property
        """
        # if the item is not in the data table, return defaults for properties
        if uid not in self._data.index:
            return {"weight": self._default_weight, MISC_PROPERTIES: {}}

        properties = self._data.loc[uid].to_dict()
        return flatten(properties)

    def get_property(self, uid, prop_name) -> Any:
        """Get a property of an item

        Parameters
        ----------
        uid: Hashable
            uid is the index used to fetch its property

        prop_name : str | int
            name of the property to get

        Returns
        -------
        out : Any
            value of the property

        None
            if property not found

        See Also
        --------
        get_properties, set_property
        """
        # if the item is in the data table and the property is 'misc_properties'
        # return a flattened dictionary of 'misc_properties'
        if uid in self._data.index and prop_name == MISC_PROPERTIES:
            return flatten(self._data.loc[uid, prop_name])

        properties: dict = self.get_properties(uid)
        return properties.get(prop_name, None)

    def set_property(self, uid, prop_name, prop_val) -> None:
        """Set a property of an item in the 'properties' collection

        Parameters
        ----------
        uid : Hashable
            uid is the index used to set its property
        prop_name : str | int
            name of the property to set
        prop_val : any
            value of the property to set

        Raises
        ------
        KeyError
            If (`uid`) is not in :attr:`properties`

        See Also
        --------
        get_property, get_properties
        """
        if uid in self._data.index:
            self._update_row(uid, prop_name, prop_val)
        else:
            self._add_row(uid, prop_name, prop_val)

    def _update_row(self, uid, prop_name, prop_val):
        properties = self.get_properties(uid)
        if prop_name in properties:
            self._data.loc[uid, prop_name] = prop_val
        else:
            # add the unique property to 'misc_properties'
            self._data.loc[uid, MISC_PROPERTIES].update({prop_name: prop_val})

    def _add_row(self, uid, prop_name, prop_val):
        if prop_name in self._data.columns.tolist():
            new_row = DataFrame(
                {
                    "weight": [self._default_weight],
                    MISC_PROPERTIES: [{}],
                    prop_name: [prop_val],
                },
                index=[uid],
            )
        else:
            # add the unique property to 'misc_properties'
            new_row = DataFrame(
                {
                    "weight": [self._default_weight],
                    MISC_PROPERTIES: [{prop_name: prop_val}],
                },
                index=[uid],
            )

        self._data = pd.concat([self._data, new_row])

    def __getitem__(self, uid) -> dict:
        """Gets all the properties of an item

        This magic method has the same behavior as get_properties; in fact, it calls the method
        get_properties.

        This magic method
        allows the use of brackets to get an item from an instance of PropertyStore

        For example:

        ps = PropertyStore(data=data)

        node = ps["493hg9"]
        same_node = ps.get_properties("493hg9")

        assert node = same_node

        """
        return self.get_properties(uid)

    def __contains__(self, uid) -> bool:
        """Checks if the item is present in the data table"""
        return uid in self._data.index


def flatten(my_dict):
    """
    Recursive method to flatten dictionary for returning properties as
    a dictionary instead of a Series, from [StackOverflow](https://stackoverflow.com/a/71952620)
    """
    result = {}
    for key, value in my_dict.items():
        if isinstance(value, dict):
            temp = flatten(value)
            # if temp is an empty dictionary, we still want to include the empty dictionary in the
            # flattened dictionary
            # example: { 'foo': 'bar', 'snafu': {} }
            if temp == dict():
                temp = {key: temp}
            temp.update(result)
            result = temp
        else:
            result[key] = value
    return result
