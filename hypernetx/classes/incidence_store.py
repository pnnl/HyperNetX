
from copy import copy,deepcopy

class IncidenceStore(ABC):
    """
    Incidence store object that stores and accesses (multi) incidences with standard methods.

    Parameters
    ----------
    data : Iterable mapping of ordered pairs (edge,node) to data properties
    aggregate_by : method for handling properties associated with duplicate pairs,
        may be given as a dictionary keyed on properties
        
    Methods
    --------
    __iter__
    __len__
    __contains__
    __getitem__
    incidence_matrix
    collapseitem
    restrict_to
    dimensions
    membership
    elements
    dual
    """
    
    def __init__(self, data, edge_level, node_level):
        """
        Multi index for (Incidence) Property Store with additional
        functionality

        Parameters
        ----------
        data : _type_
            collection of ordered pairs
        """
        self._data = incidence_data  # Data with incidence pairs and attributes

        ### class does not allow duplicate pairs so either the first pair
        # and its data will be used or the aggregate_by method will be
        # used. these should be documented for behavior 
        pass

    
    def __iter__(self):
        """
        Iterator over the incidence pairs of the hypergraph

        """
        
        return iter(self._data)

    def __len__(self):
        """
        Total number of incidences

        Returns
        -------
        int

        """
        
        # 
        return len(self._data)
    
    def __contains__(self, incidence_pair):
        """
        First, check if the incidence pair is of length two.        
        Then, checks if it exists in incidence pairs.

        """
                
        pass
    
    # #### Do we need both of these??? Or can we have a single getitem??
    # def __getitem__(self, incidence_key):
    #     """
    #     key is incidence key (e.g., index) and returns incidence pair and attributes

    #     """
    #     ### let getitem retrieve single line or multiple lines depending on if 
    #     ### incidence_key is a hashable key for the instance, or an ordered pair of an
    #     ### incidence. Should replace get_incidence_attributes
    #     return self._data[incidence_key]
    # def get_incidence_attributes(self, incidence_pair):
    #     '''
    #     Given an incidence pair return all instances of that incidence with incidence keys and attributes.

    #     Parameters
    #     ----------
    #     incidence_pair : tuple
    #         (edge, node) pair.

    #     Returns
    #     -------
    #     dictionary of incidence pairs of that incidence with incidence key (e.g., index) as dictionary keys and 
    #     attributes as dictionary values.

    #     '''
    #     pass

    def neighbors(self,level,key):
        """
        Returns elements or memberships depending on level
        level 0, key is edge, returns nodes in edge
        level 1, key is node, returns edges containing node

        Parameters
        ----------
        level : _type_
            _description_
        key : _type_
            _description_
        """
    
    # def dataframe(self):
    #     """
    #     Dataframe format with columns edge,node,weight,<properties>
    #     One column should handle variable length property dictionaries
    #     """
    #     pass

    # ### By passing dataframe we might handle these in the hypergraph class.
    # def incidence_dataframe(self,value=1):
    #     ## if a dataframe then this could be a pivot table
    #     ### with data dictated by the property, eg. weight
    # def incidence_matrix(self, value=1, index=False): 
    #     ### incidence_dataframe values, currently sparse matrix and depends
    #     ### on label encoding
    #     """
    #     Implement incidence matrix creation logic here from unique incidence pairs.
    #     """
    #     pass


    ### This can be replaced by shape since we only have 2 data columns
    def dimensions(self):
        """
        Same as entity set?
        Dimensions of data i.e., the number of distinct items 
        in each level (column) of the underlying dataframe of incidences.
        Or is this just for the edge and node columns?

        """
        pass

    def restrict_to(self,level,items, inplace=True):
        """
        returns IncidenceStore of subset of incidence store restricted 
        to pairs with items in the given level
        Will return with same data or deepcopy depending on inplace

        Parameters
        ----------
        level : _type_
            _description_
        items : _type_
            _description_
        inplace : bool, optional
            _description_, by default True
        """
        pass

    def dual(self, inplace=True):
        """
        This shares the incidence data so any changes will be made to both
        unless inplace==False, in which case the current store is deep copied

        Parameters
        ----------
        inplace : bool, optional
            _description_, by default True

        Returns
        -------
        IncidenceStore 
            new instance with dual flag = True
        """
        if inplace: ### This should keep links to the same data
            return self.__class__(data,node_level,edge_level)
        else:
            return self.__class__(deepcopy(data),node_level,edge_level)
        

    def collapse(self,level, return_equivalence_classes=False):
        """
        Collapse according to level:
        edges - 0 - elements : equivalence_class
        nodes - 1 - memberships : equivalence_class
        both - 2 - first collapse on nodes then on edges

        Parameters
        ----------
        level : _type_
            _description_
        return_equivalence_classes : bool, optional
            _description_, by default False

        Returns
        -------
        IncidenceStore without properties labeled by class rep and count
        Equivalence Classes
        """
        pass

