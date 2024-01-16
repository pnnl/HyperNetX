import pandas as pd

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
        self._incidences = incidences  # Dataframe of incidence pairs

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
        key is incidence key (e.g., index) and returns incidence pair

        """
        
        return self._incidences[incidence_key]

    def incidence_matrix(self):
        """
        Implement incidence matrix creation logic here

        """
        # 
        pass

    def aggregate(self, level):
        """
        First aggregate the multi incidences

        """
        pass

    def restrict_to(self, elements):
        """
        Implement restriction logic here

        """
        pass

    def dimensions(self):
        """
        Return the dimensions of the hypergraph

        """
        pass

    def membership(self, key):
        """
        Return the incidences involving a key

        """
        pass

    def elements(self, level):
        """
        Return all elements at a given level

        """
        pass

    def dual(self):
        """
        Return the dual hypergraph by flipping the incidence pairs. 
        This returns a new dataframe with (index, edge, node) tuples removed 

        """
        pass
        