# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

from collections import defaultdict
import warnings
from copy import copy
import numpy as np
import networkx as nx
from hypernetx import HyperNetXError
from hypernetx.utils.extras import HNXCount

__all__ = ["Entity", "EntitySet"]


class Entity():
    '''
    Class for objects used in building hypergraphs, such as edges and nodes.

    The Entity class simply stores an object with a uid, other elements to which it is connected, and associated properties.
    In the hypergraph case, it is used to store nodes/edges and their edge/node neighbors and properties such as weights, categories etc. 
    

    Parameters
    ----------
    uid : hashable
        a unique identifier

    elements : list or an iterable that can be cast to a list, optional, default: None
        a list of entities to which the entity is connected. Nodes can only have edges as members and vice-versa.

    entity : Entity
        an Entity object to be cloned into a new Entity with uid. If the uid is the same as
        Entity.uid then the entities will not be distinguishable and error will be raised.
        The `elements` in the signature will be added to the cloned entity.

    props : keyword arguments, optional, default: {}
        properties belonging to the entity added as key=value pairs.
        The key must be hashable.

    Notes
    -----

    An Entity is a container-like object, which has a unique identifier and
    may contain elements to which it is connected and have properties.
    The Entity class was created as a generic object to represent
    Hypergraph nodes and edges.

    - An Entity is distinguished by its identifier (sortable,hashable) :func:`Entity.uid`
    - An Entity is a container for other entities but may not contain itself, :func:`Entity.elements`
    - An Entity has properties :func:`Entity.properties`
    - An Entity has memberships to other entities, :func:`Entity.memberships`.
    - An Entity has children, :func:`Entity.children`, which are the elements of its elements.
    - :func:`Entity.children` are registered in the :func:`Entity.registry`.
    - All descendents of Entity are registered in :func:`Entity.fullregistry()`.

    **Honor System**

    The Honor System where every element has a unique uid is no longer enforced. It is only enforced among nodes and edges,
    where a node cannot have the same uid as other nodes, and an edge cannot have the same uid as other edges.
    Two entities are equal if their __dict__ objects match.
    For performance reasons many methods distinguish entities by their uids.
    It is, therefore, up to the user to ensure entities with the same uids are indeed the same.
    Not doing so will prevent these duplicate Entities from being added to an EntitySet.
    In particular, the methods in the Hypergraph class assume distinct nodes have distinct uids from each other and edges
    have distinct uids from each other.

    Examples
    --------

        >>> x = Entity('x')
        >>> y = Entity('y',[x])
        >>> z = Entity('z',[x,y],weight=1)
        >>> z
        Entity(z,['y', 'x'],{'weight': 1})
        >>> z.uid
        'z'
        >>> z.elements
        ['x', 'y']
        >>> z.properties
        {'weight': 1}
        >>> x.elements
        []
    
    See Also
    --------
    EntitySet

    """
    
    '''
    def __init__(self, uid, elements=[], entity=None, **props):

        self._uid = uid

        # Copies an entity if entity is passed into the constructor
        if entity is not None:
            if isinstance(entity, Entity):
                # copy the contents of the entity except for the ids
                self._elements = entity.elements
                self.__dict__.update(entity.properties)
        else:
            if isinstance(elements, list):
                self._elements = elements
            else:
                try:
                    self._elements = list(elements)
                except:
                    raise HyperNetXError(f"Error: elements must be able to be cast to a list.")


        # add properties to the class dictionary (arbitrary keyword/value pairs)
        self.__dict__.update(props)

    @property
    def properties(self):
        """Dictionary of properties of entity"""
        temp = self.__dict__.copy()
        del temp["_elements"]
        del temp["_uid"]
        return temp

    @property
    def uid(self):
        """String identifier for entity"""
        # each entity has an associated ID
        return self._uid

    @property
    def elements(self):
        """
        Dictionary of elements to which entity is connected
        """
        return self._elements

    @property
    def size(self):
        """
        Returns the number of elements to which an entity is connecteds
        """
        return len(self)

    # defines equality to be whether all the state variables match.
    def __eq__(self, other):
        """
        Defines equality for Entities based on equivalence of their __dict__ objects.
        """
        if (self.__class__ != other.__class__) or (self.uid != other.uid) or (self.properties != other.properties) or (self.elements != other.elements):
            return False
        return True

    def __len__(self):
        """Returns the number of elements to which an entity is connected"""
        return len(self._elements)

    def __str__(self):
        """Return the entity uid."""
        return f"{self.uid}"

    def __repr__(self):
        """Returns a string resembling the constructor for the entity"""
        return f"Entity({self._uid}),{list(self.elements)},{self.properties})"

    def __contains__(self, item):
        """
        Return whether an entity is connected to another entitys.

        Parameters
        ----------
        item : hashable

        Returns
        -------
        Boolean
        """
        return item in self._elements

    def __getitem__(self, item):
        """
        Returns Entity element by uid. Use :func:`E[uid]`.

        Parameters
        ----------
        item : hashable or Entity

        Returns
        -------
        Entity or None

        If item not in entity, returns None.
        """
        return self._elements.get(item)

    def __iter__(self):
        """Returns iterator on element ids."""
        return iter(self._elements)

    def clone(self, newuid):
        """
        Returns shallow copy of entity with newuid. Entity's elements will
        belong to two distinct Entities.

        Parameters
        ----------
        newuid : hashable
            Name of the new entity

        Returns
        -------
        clone : Entity

        """
        return Entity(newuid, entity=self)
    
    def remove(self, item):
        """
        Removes an item from the Entity's elements if it is in the list of elements, otherwise raises an error.

        Parameters
        ----------
        item : hashable or Entity
            entity to remove from the list of elements

        Returns
        -------

        """
        if isinstance(item, Entity):
            if item.uid in self:
                self._elements.remove(item.uid)
            else:
                raise HyperNetXError("Item is not in the elements of the entity.")
        else:
            if item in self:
                self._elements.remove(item)
            else:
                raise HyperNetXError("Item is not in the elements of the entity.")

    def add(self, item):
        """
        Adds an item to the Entity's elements. Warning: does not check if it is not already in the list of elements, otherwise raises an error.

        Parameters
        ----------
        item : hashable or Entity
            entity to add to the list of elements

        Returns
        -------

        """

        if isinstance(item, Entity):
            self._elements.append(item.uid)
        else:
            self._elements.append(item)

class EntitySet():
    '''
    The EntitySet class is a class that manages and contains Entity class elements. It is used to represent a list of Entity objects.
    A practical example are edge and node lists, which together form a bipartite representation of a hypergraph.
    Each instance of an EntitySet contains a uid for the class and an element dictionary of uid/Entity pairs.

    Parameters
    ----------
    uid : hashable
        a unique identifier

    elements : list or dict, optional, default: empty dictionary
        a list/dict of Entities or iterables
        If a dict of iterables or Entity objects is given, the keys become the uids and the values become Entities.
        If a list of Entities is given, the uid of the Entity is used for the keys of the elements dict.
        If list of iterables, the uids are autmatically assigned
        If uids overlap, throws an HyperNetXError

    entityset : EntitySet
        an EntitySet object to be cloned into a new EntitySet with uid. If the uid is the same as
        Entity.uid then the entities will not be distinguishable and error will be raised.

    props : keyword arguments, optional, default: {}
        properties belonging to the entityset added as key=value pairs.
        Both key and value must be hashable.

    Notes
    -----

    The EntitySet class was created to store Entities satifying the Bipartite Condition.

    **Bipartite Condition**

    *Entities that are elements of the same EntitySet, cannot contain each other as elements.*
    The elements of an EntitySet generate a specific partition for a bipartite graph,
    because the elements property of an Entity relate edges to nodes and vice-versa.
    The partition is isomorphic to a Hypergraph where the elements correspond to hyperedges and
    the children correspond to the nodes. EntitySets are the basic objects used to construct hypergraphs
    in HNX.

    Examples
    --------

        >>> edges = [Entity(0, [1, 2, 3]), Entity(1 ,[0, 2]), Entity(2 ,[0, 1, 3])] 
        >>> es = EntitySet('edges', edges)
        >>> es.uid
        'edges'
        >>> es.elements
        {0:Entity(0, [1, 2, 3]), 1:Entity(1 ,[0, 2]), 2:Entity(2 ,[0, 1, 3])] 
        >>> es.properties
        {}
        >>> es.add(Entity(3, [0]))
        >>> es.weight = 1
        >>> es.properties
        {'weight': 1}
        >>> es.uidset
        {0, 1, 2, 3}

    See Also
    --------
    Entity

    """
    '''
    def __init__(self, uid, elements=dict(), entityset=None):
        self._uid = uid
        self.count = HNXCount(0)
        if entityset is None:
            self._elements = dict()
        else:
            self._elements = entityset.elements

        self.add_elements_from(elements)

    def __len__(self):
        """Return the number of entities."""
        return len(self._elements)

    def __contains__(self, item):
        """
        Defines containment for Entities.

        Parameters
        ----------
        item : hashable or Entity

        Returns
        -------
        Boolean

        Depends on the `Honor System`_ . Allows for uids to be used as shorthand for their entity.
        This is done for performance reasons, but will fail if uids are
        not unique to their entities.
        Is not transitive.
        """
        if isinstance(item, Entity):
            return item.uid in self._elements
        else:
            return item in self._elements

    def __getitem__(self, item):
        """
        Returns Entity element by uid. Use :func:`E[uid]`.

        Parameters
        ----------
        item : hashable or Entity

        Returns
        -------
        Entity or None

        If item not in entity, returns None.
        """
        if isinstance(item, Entity):
            return self._elements.get(item.uid)
        else:
            return self._elements.get(item)

    def __str__(self):
        """Return the entityset uid."""
        return f"{self.uid}"

    def __iter__(self):
        """Returns iterator on element ids."""
        return iter(self.elements)

    def __repr__(self):
        """Returns a string resembling the constructor for entityset without any
        children"""
        return f"EntitySet({self._uid},{list(self.uidset)})"

    def __call__(self):
        """Returns an iterator on elements"""
        for e in self.elements.values():
            yield e

    def __eq__(self, other):
        """
        Defines equality for EntitySets based on equivalence of their elements.
        """
        # Compare top level properties: same class, same uids, same element names
        if (self.__class__ != other.__class__) or (self.uid != other.uid) or (set(self.elements.keys()) != set(other.elements.keys())):
                return False
        return True

    @property
    def is_empty(self):
        """Boolean indicating if entityset.elements is empty"""
        return len(self) == 0

    @property
    def uid(self):
        """
        EntitySet uid.
        """
        return self._uid

    @property
    def uidset(self):
        """
        A set of the uids of the EntitySet's element.
        """
        return frozenset(self._elements.keys())

    @property
    def incidence_dict(self):
        """
        Dictionary of element.uid:element.uidset for each element in entityset
        """
        temp = dict()
        for entity in self.elements.values():
            temp[entity.uid] = {item for item in entity.elements}
        return temp

    @property
    def elements(self):
        ''' Dictionary of the uid/element pairs'''
        return self._elements

    @property
    def children(self):
        ''' Set of all the uids of elements to which the nodes are connected'''
        children = set()
        for items in self._elements.values():
            children.update(items.elements)
        return children

    def intersection(self, other):
        """
        A dictionary of elements belonging to the entityset and another entityset.

        Parameters
        ----------
        other : EntitySet

        Returns
        -------
        Dictionary of elements : dict

        """
        return {e: self[e] for e in self if e in other}

    def get_dual(self):
        """
        Return an incidence dictionary with the roles of children and parents swapped.

        Parameters
        ----------

        Returns
        -------
        Incidence dictionary with the children uids as keys and the elements to which they belong as values.

        """
        elements = defaultdict(list)
        for id in self.elements:
            for member in self.elements[id].elements:
                elements[member].append(id)
        return elements

    def add_element(self, item=[], uid=None):
        """
        Adds a single element to entityset's elements, checking to make sure no duplicate references are
        made to element ids in the entityset.

        Parameters
        ----------
        item : One Entity object or list/set/array that is the item's element list.
        uid : Hashable which is the added Entity's uid. If you want this to be auto-assigned, only enter the item.

        Returns
        -------
        uid : returns the uid of the added element in case it was auto-generated and you need it.
        """
        
        if isinstance(item, Entity):
            if item.uid in self.elements:
                raise HyperNetXError(f"Error: {uid} references an existing Entity in the EntitySet.")
            uid = item.uid
            self.elements[uid] = item
        else:
            if uid is None:
                uid = self.count()
 
            if uid in self.elements:
                raise HyperNetXError(f"Error: {uid} references an existing Entity in the EntitySet.")
            self.elements[uid] = Entity(uid, item)
        return uid

    def add_elements_from(self, elements):
        """
        Adds items to entityset's elements, checking to make sure no duplicate references are
        made to element ids.

        Parameters
        ----------
        elements : dictionary with uid:list/Entity pairs or an iterable of Entities/lists

        Returns
        -------

        """
        # If it's a dictionary with uid/data pairs, simply add to the elements dictionary
        if isinstance(elements, dict):
            for uid, item in elements.items():
                self.add_element(item, uid)
        # If it's an iterable, a uid needs to be created. If the list item is an entity, it is already created, and if not, you can use a system-generated uid.
        else:
            for item in elements:
                self.add_element(item)

    def add(self, elements):
        """
        Adds items to entityset's elements, checking to make sure no duplicate references are
        made to element ids. This is the same as add_elements_from()

        Parameters
        ----------
        elements : dictionary with uid:list/Entity pairs or a list of Entities/lists

        Returns
        -------

        """
        self.add_elements_from(elements)

    def remove_element(self, item):
        """
        Removes item from entityset if it exists, otherwise raises a HyperNetXError.

        Parameters
        ----------
        item : Hashable or Entity

        Returns
        -------

        """
        if isinstance(item, Entity):
            item = item.uid
        if item not in self.elements:
            raise HyperNetXError(f"Error: {item} is not an existing Entity in the EntitySet.")
        else:
            del self._elements[item]

    def remove_elements_from(self, arg_set):
        """
        Similar to :func:`remove()`. Removes elements in arg_set.

        Parameters
        ----------
        arg_set : Iterable of hashables or entities

        Returns
        -------

        """
        for item in arg_set:
            self.remove_element(item)    

    def remove(self, *args):
        """
        Removes args from entityset's elements

        Parameters
        ----------
        args : One or more hashables or entities

        Returns
        -------

        """
        for item in args:
            self.remove_element(item)

    def clone(self, newuid):
        """
        Returns shallow copy of entityset with newuid.


        Parameters
        ----------
        newuid : hashable
            Name of the new entityset

        Returns
        -------
        clone : EntitySet

        """
        return EntitySet(newuid, elements=self.elements)


    def collapse_identical_elements(self, newuid, return_equivalence_classes=False):
        """
        Returns a copy of the entityset with duplicate elements combined, using representatives of equivalence classes as element keys.
        Two elements of an EntitySet are collapsed if they share the same children.

        Parameters
        ----------
        newuid : hashable

        return_equivalence_classes : boolean, default=False
            If True, return a dictionary of equivalence classes keyed by new edge names

        Returns
        -------
         : EntitySet
        eq_classes : dict (if return_equivalence_classes = True)

        Notes
        -----
        Treats elements of the entityset as equal if they have the same uidsets. Using this
        as an equivalence relation, the entityset's uidset is partitioned into equivalence classes.
        The equivalent elements are identified using a single entity by using the
        frozenset of uids associated to these elements as the uid for the new element
        and dropping the properties.
        If use_reps is set to True a representative element of the equivalence class is
        used as identifier instead of the frozenset.

        Example: ::

            >>> E = EntitySet('E',elements=[Entity('E1', ['a','b']),Entity('E2',['a','b'])])
            >>> E.incidence_dict
            {'E1': {'a', 'b'}, 'E2': {'a', 'b'}}
            >>> E.collapse_identical_elements('_',).incidence_dict
            {'E2': {'a', 'b'}}

        """

        shared_children = defaultdict(set)
        for e in self.__call__():
            shared_children[frozenset(e.elements)].add(e.uid)
        new_entity_dict = {
            f"{next(iter(v))}:{len(v)}": set(k) for k, v in shared_children.items()
        }
        if return_equivalence_classes:
            eq_classes = {
                f"{next(iter(v))}:{len(v)}": v for k, v in shared_children.items()
            }
            return EntitySet(newuid, new_entity_dict), dict(eq_classes)
        else:
            return EntitySet(newuid, new_entity_dict)


    def incidence_matrix(self, sparse=True, index=False, weight = lambda self, node, edge : 1):
        """
        An incidence matrix for the EntitySet indexed by children x uidset.

        Parameters
        ----------
        sparse : boolean, optional, default: True

        index : boolean, optional, default False
            If True return will include a dictionary of children uid : row number
            and element uid : column number

        weight : a lambda function returning a weight in the incidence matrix given the uid of the node and edge.
        The default is to return 1 when a node/edge pair exist.
        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix or np.ndarray

        row dictionary : dict
            Dictionary identifying row with item in entityset's children

        column dictionary : dict
            Dictionary identifying column with item in entityset's uidset

        Notes
        -----

        Example: ::

            >>> E = EntitySet('E',{'a':{1,2,3},'b':{2,3},'c':{1,4}})
            >>> E.incidence_matrix(sparse=False, index=True)
            (array([[0, 1, 1],
                    [1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 1]]), {0: 1, 1: 2, 2: 3, 3: 4}, {0: 'b', 1: 'a', 2: 'c'})
        """
        if sparse:
            from scipy.sparse import csr_matrix

        nchildren = len(self.children)
        nuidset = len(self.uidset)

        # given a child's label, associates it with a numerical index.
        ndict = dict(zip(self.children, range(nchildren)))
        edict = dict(zip(self.uidset, range(nuidset)))

        if len(ndict) != 0:

            if index:
                # reverses the order of the dictionary matching labels to indices.
                rowdict = {v: k for k, v in ndict.items()}
                coldict = {v: k for k, v in edict.items()}

            if sparse:
                # Create csr sparse matrix
                rows = list()
                cols = list()
                data = list()
                for e in self:
                    for n in self.elements[e].elements:
                        data.append(weight(self, n, e))
                        rows.append(ndict[n])
                        cols.append(edict[e])
                MP = csr_matrix((data, (rows, cols)), dtype=float)
            else:
                # Create an np.matrix
                MP = np.zeros((nchildren, nuidset), dtype=int)
                for e in self:
                    for n in self.elements[e].elements:
                        MP[ndict[n], edict[e]] = 1
            if index:
                return MP, rowdict, coldict
            else:
                return MP
        else:
            if index:
                return np.zeros(1), {}, {}
            else:
                return np.zeros(1)
