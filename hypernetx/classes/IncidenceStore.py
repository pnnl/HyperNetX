
class IncidenceStore:
    """
    Incidence store object that stores and accesses (multi) incidences with standard methods.

    Parameters
    ----------
    incidences : Dictionary
        dicitonary of incidences with key as incidence name and value as tuple of edge and node names.
        
    Methods
    --------
    __iter__
    __len__
    __contains__
    __getitem__
    incidence_matrix
    collapse
    restrict_to
    dimensions
    membership
    elements
    dual
    """
    
    def __init__(self, incidences):
        self._incidences = incidences  # Dataframe of index, incidence pairs, and attributes

    def __iter__(self):
        """
        Iterator over the incidence pairs of the hypergraph

        """
        
        return iter(self._incidences.values)

    def __len__(self):
        """
        Total number of incidences

        Returns
        -------
        int

        """
        
        # 
        return len(self._incidences)
    
    def unique_incidences():
        """
        List of unique incidences

        Returns
        -------
        list

        """
        pass
    
    def __contains__(self, incidence_pair):
        """
        First, check if the incidence pair is of length two.
        
        Then, checks if it exists in incidence pairs.

        """
        
        
        pass

    def __getitem__(self, incidence_key):
        """
        key is incidence key (e.g., index) and returns incidence pair and attributes

        """
        
        return self._incidences[incidence_key]

    def get_incidence_attributes(self, incidence_pair):
        '''
        Given an incidence pair return all instances of that incidence with incidence keys and attributes.

        Parameters
        ----------
        incidence_pair : tuple
            (edge, node) pair.

        Returns
        -------
        dictionary of incidence pairs of that incidence with incidence key (e.g., index) as dictionary keys and 
        attributes as dictionary values.

        '''
        pass
    
    def incidence_matrix(self):
        """
        Implement incidence matrix creation logic here from unique incidence pairs.

        """
        pass

    def aggregate(self):
        # Was collapse
        """
        Collapse the multi-incidences and combine attributes of multi-incidences.
        
        Updates
        -------
        
        """
        pass

    def restrict_to(self, elements):
        """
        Should this just be a restriction condition based on a row or 

        """
        pass

    def dimensions(self):
        """
        Same as entity set?
        Dimensions of data i.e., the number of distinct items in each level (column) of the underlying dataframe of incidences.
        Or is this just for the edge and node columns?

        """
        pass

    def membership(self, key):
        """
        Not sure here.

        """
        pass

    def elements(self, level):
        """
        RNot sure here.

        """
        pass

    def dual(self):
        """
        create new dataframe by swapping the edge and row columns in the dataframe and remove incidence attributes.

        """
        pass
        