from typing import Any
from collections.abc import Hashable
from pandas import DataFrame


UID = "uid"
WEIGHT = "weight"
MISC_PROPERTIES = "misc_properties"
DEFAULT_PROPERTIES = [WEIGHT, MISC_PROPERTIES]


class PropertyStore:
    """Class for storing properties of a collection of edges, nodes, or incidences.

    Properties will be stored in a pandas dataframe.

    """

    def __init__(self, data=None, default_weight=1):
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

             edges     |  node         | weight | misc_properties | <additional property> | ...
            <edge1uid> | <node1uid>  | 1.0    | {}              | <property value>      | ...
                       | <node2uid>  | 1.5    | {}              | <property value>      | ...
            <edge2uid> | <node1uid>  | 1.0    | {}              | <property value>      | ...
                       | <node3uid>  | 1.0    | {}              | <property value>      | ...

        default_weight: int

            optional parameter that holds the specified default weight of the weight property
        """
        # If no dataframe is provided, create an empty dataframe
        if data is None:
            self._data: DataFrame = DataFrame(columns=[UID, WEIGHT, MISC_PROPERTIES])
            self._data.set_index(UID, inplace=True)
        else:
            self._data: DataFrame = data

        self._default_weight: int = default_weight
        self._columns = self._data.columns.tolist()

    @property
    def properties(self) -> DataFrame:
        """Properties assigned to all items in the underlying data table

        Returns
        -------
        out: pandas.DataFrame
            a dataframe with the following columns:
                uid, weight, properties, <optional props>
                or
                level, id, weight, properties, <optional props>
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
            return {WEIGHT: self._default_weight}
        return flatten(self._data.loc[uid].to_dict())

    def set_properties(self, uid, props) -> None:
        """
        Parameters
        ----------
        uid : Hashable
            uid is the index used to set its property
        props : a dictionary containing user-defined properties

        See Also
        --------
        get_property, get_properties, set_property
        """
        if uid not in self._data.index:
            self._data.loc[uid, :] = self._default_properties()

        for prop_name, prop_val in props.items():
            self._set_property(uid, prop_name, prop_val)

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
        # return 'misc_properties'
        if uid in self._data.index and prop_name == MISC_PROPERTIES:
            return self._data.loc[uid][MISC_PROPERTIES]
        return self.get_properties(uid).get(prop_name, None)

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

        See Also
        --------
        get_property, get_properties, set_properties
        """
        # if the uid is not present, add the uid with default properties to the dataframe
        if uid not in self._data.index:
            self._data.loc[uid, :] = self._default_properties()

        self._set_property(uid, prop_name, prop_val)

    def _set_property(self, uid, prop_name, prop_val):
        """Updates a property of an item in the underlying data table

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
            If (`uid`) is not in the underlying data table
        """
        # Holds the logic on how new properties are added to the dataframe for existing items
        # Currently supports updating existing properties and adding a property to the misc_properties
        # A potential feature is adding a common property to a subset of items
        if prop_name in self._columns:
            # overwrite the current property with the updated property
            self._data.at[uid, prop_name] = prop_val
        else:
            # if the property to be added is not one of existing properties,
            # add the unique property to 'misc_properties'
            self._data.at[uid, MISC_PROPERTIES].update({prop_name: prop_val})

    def _default_properties(self) -> dict:
        """Create a default properties dictionary; if custom properties are present, add them and set to None"""
        # Get the required property fields
        properties = self._required_properties()
        # add user-defined properties
        user_defined_properties = [
            prop
            for prop in self._data.columns.tolist()
            if prop not in DEFAULT_PROPERTIES
        ]

        if not user_defined_properties:
            return properties

        # Add user-defined properties to data
        for prop in user_defined_properties:
            properties[prop] = None
        return properties

    def default_properties_df(self) -> dict:
        """Returns a dictionary of only default properties with default values; this data is used as the value for
        the parameter 'data' to create a Dataframe

        MISC_PROPERTIES is a list of exactly one dictionary
        The dictionary must be put into a list so that Pandas uses the entire value in the entire location

        See HypergraphView.to_dataframe
        """
        return {
            WEIGHT: self._default_weight,
            MISC_PROPERTIES: [{}],
        }

    def _required_properties(self) -> dict:
        return {
            WEIGHT: self._default_weight,
            MISC_PROPERTIES: {},
        }

    def __getitem__(self, uid) -> dict:
        """Gets all the properties of an item

        This magic method has the same behavior as get_properties; in fact, it calls the method
        get_properties.

        This magic method allows the use of brackets to get an item from an instance of PropertyStore

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

    def copy(self, deep=False):
        data = self._data.copy(deep=deep)
        w = self._default_weight
        return PropertyStore(data, default_weight=w)


def flatten(my_dict):
    """
    Recursive method to flatten dictionary for returning properties as
    a dictionary instead of a Series, from [StackOverflow](https://stackoverflow.com/a/71952620)
    """
    result = {}
    for key, value in my_dict.items():
        if value == {}:
            continue
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
