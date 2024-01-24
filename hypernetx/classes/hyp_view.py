


class HypergraphView(object):
    """
    Wrapper for Property and Incidence Stores holding structural and 
    meta data for hypergraph. Provides methods matching EntitySet 
    methods in previous versions.
    """
    def __init__(incidence_store,level,property_store=None,l):

        self._incidences = incidence_store
        ### incidence store needs index or columns
        self._level = level  ## edges, nodes, or incidence pairs
        if property_store is not None:
            self._properties = property_store
            ### property store needs index column = keys
        else:
            self._properties = hnx.PropertyStore()  
            ### if no properties and level 0 or 1, 
            ### create property store that 
            ### returns weight 1 on every call for a weight 
            ### and empty properties otherwise.


    def __iter__(self):
        """
        Defined by level store
        """
        pass

    def __len__(self):
        """
        Defined by level store
        """
        pass

    def __contains__(self,item):
        """
        Defined by level store

        Parameters
        ----------
        item : _type_
            _description_
        """
        pass

    
    def to_dataframe(self):
        """
        Defined by level store.
        Returns a pandas dataframe keyed by level keys 
            with properties as columns or in a variable length dict.
            The returned data frame will either reflect the
            property store or the incidence store depending on the level."""
        pass



    def to_dict(self, data=False):
        """
        Association dictionary - neighbors from bipartite form
        returns a dictionary of key: <elements,memberships,elements>
        for level 0,1,2
        values are initlist from AttrList class
        if data = True, include data
        """
        pass



    def __getitem__(self,key):
        """
        Returns incident objects (neighbors in bipartite graph) 
        to keyed object as an AttrList.
        Returns AttrList associated with item, 
        attributes/properties may be called 
        from AttrList 
        If level 0 - elements, if level 1 - memberships,
        if level 2 - TBD - uses getitem from stores and links to props

        Parameters
        ----------
        key : _type_
            _description_
        """
        pass

    def properties(self,key=None,prop_name=None):
        """
        Return dictionary of properties or single property for key
        Currently ties into AttrList object in utils.
        Uses getitem from stores

        Parameters
        ----------
        key : _type_
            _description_
        prop_name : _type_, optional
            _description_, by default None

        Returns
        -------
        if key=None and prop=None, dictionary key:properties OR
        elif prop=None, dictionary prop: value for key
        elif key = None, dictionary of keys: prop value
        else property value
        """
        # If a dfp style dataframe use .to_dict()
        pass

    def __call__(self):
        """
        Iterator over keys in store -
        level 0 = edges, 1 = nodes, 2 = incidence pairs
        """
        pass
    

    def to_json(self):
        """
        Returns jsonified data. For levels 0,1 this will be the edge and nodes
        properties and for level 2 this will be the incidence pairs and their 
        properties
        """
        pass

    def memberships(self,item):
        """
        applies to level 1: returns edges the item belongs to.
        if level = 0 or 2 then memberships returns none.

        Parameters
        ----------
        item : _type_
            _description_
        """
        pass

    def elements(self,item):
        """
        applies to levels 0: returns nodes the item belongs to.
        if level = 1 or 2 then elements returns none.

        Parameters
        ----------
        item : _type_
            _description_
        """
        pass


    #### data,labels should be handled in the stores and accessible
    #### here - if we want them??
    def encoder(self,item=None):
        """
        returns integer encoded data and labels for use with fast
        processing methods in form of label:int dictionaries
        """
        pass

    def decoder(self):
        """
        returns int:label dictionaries 
        """




    
