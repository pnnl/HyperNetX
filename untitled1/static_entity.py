# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

import warnings
from copy import copy
import numpy as np
import networkx as nx
from hypernetx.classes.entity import Entity, EntitySet
from hypernetx import HyperNetXError
from collections.abc import Iterable
from collections import defaultdict


__all__ = [
    'StaticEntity',
    'StaticEntitySet'
]

class StaticEntity(object):

    def __init__(self, uid=None, data={}, **props):
        super().__init__()

        self._uid = uid
        self.__dict__.update(props)

        if isinstance(data, dict):
            self._elements = data
        elif isinstance(data, list):
            self._elements = {d: {} for d in data}
        elif isinstance(data, Entity):
            self._elements = data.nested_incidence_dict()
            self.__dict__ = update(Entity.properties)

        self._state_dict = defaultdict(dict)  # make this a factory based on key?

    @property
    def properties(self):
        """Dictionary of properties of entity"""
        temp = self.__dict__.copy()
        del temp['_elements']
        del temp['_uid']
        del temp['_state_dict']
        return temp

    @property
    def uid(self):
        """String identifier for entity"""
        return self._uid

    @property
    def uidset(self):
        return frozenset(self._elements.keys())

    @property
    def elements(self):
        """
        Dictionary of elements belonging to entity.
        """
        return dict(self._elements)

    def __eq__(self, other):
        """
        Defines equality for Entities based on equivalence of their __dict__ objects.
        """
        return self._elements == other._elements and self.properties == other.properties

    def __len__(self):
        """Returns the number of elements in entity"""
        return len(self._elements)

    @property
    def is_empty(self):
        """Boolean indicating if entity.elements is empty"""
        return len(self) == 0

    def __str__(self):
        """Return the entity uid."""
        if self.uid:
            return f'{self.uid}'
        else:
            return ','.join([str(x) for x in self.uidset])

    def __repr__(self):
        """Returns a string resembling the constructor for entity without any
        children"""
        return f'StaticEntity({self._uid},{list(self.uidset)},{self.properties})'

    def __contains__(self, item):
        """
        Defines containment for Entities.

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
        Returns Entity element by uid. Use :code:`E[uid]`.

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
        return iter(self.elements)

    def __call__(self):
        """Returns an iterator on elements"""
        for e in self.elements.values():
            yield e

    def size(self):
        """
        Returns the number of elements in entity
        """
        return len(self)

    @property
    def incidence_dict(self):
        """
        Dictionary of element.uid:element.uidset for each element in entity

        To return an incidence dictionary of all nested entities in entity
        use nested_incidence_dict
        """
        temp = dict()
        for k, v in self._elements.items():
            if isinstance(v, Iterable):
                temp[k] = frozenset(v)
            else:
                temp[k] = frozenset([v])
        return temp

    @property
    def children(self):
        """
        Set of uids of the elements of elements of entity.

        To return set of ids for deeper level use:
        :code:`Entity.levelset(level).keys()`
        """
        return set(self.levelset(2).keys())  ###### TODO levelset

    @property
    def registry(self):
        """
        Dictionary of uid:Entity pairs for children entity.

        To return a dictionary of all entities at all depths
        :code:`Entity.complete_registry()`
        """
        return self.levelset(2)        ###### TODO levelset

    # @property
    # def is_bipartite(self):
    #     """
    #     Returns boolean indicating if the entity satisfies the `Bipartite Condition`_
    #     """
    #     if self.uidset.isdisjoint(self.children):
    #         return True
    #     else:
    #         return False

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
        return StaticEntity(uid=newuid, data=self._elements, **self.properties)

    def intersection(self, other):
        """
        A dictionary of elements belonging to entity and other.

        Parameters
        ----------
        other : Entity

        Returns
        -------
        Dictionary of elements : dict

        """
        return {e: self[e] for e in self if e in other}

    # def levelset(self, k=1):


    ########### TODO:

    def restrict_to(self, element_subset, name=None):
        """
        Shallow copy of entity removing elements not in element_subset.

        Parameters
        ----------
        element_subset : iterable
            A subset of entities elements

        name: hashable, optional
            If not given, a name is generated to reflect entity uid

        Returns
        -------
        New Entity : Entity
            Could be empty.

        """
        newelements = [self[e] for e in element_subset if e in self]
        name = name or f'{self.uid}_r'
        return Entity(name, newelements, **self.properties)


class StaticEntitySet(Entity):   ### this will default to StaticEntity - don't need a separate class, do we?
    """
    .. _entityset:

    Parameters
    ----------
    uid : hashable
        a unique identifier

    elements : list or dict, optional, default: None
        a list of entities with identifiers different than uid and/or
        hashables different than uid, see `Honor System`_

    props : keyword arguments, optional, default: {}
        properties belonging to the entity added as key=value pairs.
        Both key and value must be hashable.

    Notes
    -----
    The EntitySet class was created to distinguish Entities satifying the Bipartite Condition.

    .. _Bipartite Condition:

    **Bipartite Condition**

    *Entities that are elements of the same EntitySet, may not contain each other as elements.*
    The elements and children of an EntitySet generate a specific partition for a bipartite graph.
    The partition is isomorphic to a Hypergraph where the elements correspond to hyperedges and
    the children correspond to the nodes. EntitySets are the basic objects used to construct hypergraphs
    in HNX.

    Example: ::

        >>> y = Entity('y')
        >>> x = Entity('x')
        >>> x.add(y)
        >>> y.add(x)
        >>> w = EntitySet('w',[x,y])
        HyperNetXError: Error: Fails the Bipartite Condition for EntitySet.
        y references a child of an existing Entity in the EntitySet.

    """

    def __init__(self, uid, elements=[], **props):
        super().__init__(uid, elements, **props)
        if not self.is_bipartite:
            raise HyperNetXError('Entity does not satisfy the Bipartite Condition, elements and children are not disjoint.')

    def __str__(self):
        """Return the entityset uid."""
        return f'{self.uid}'

    def __repr__(self):
        """Returns a string resembling the constructor for entityset without any
        children"""
        return f'EntitySet({self._uid},{list(self.uidset)},{self.properties})'

    def add(self, *args):
        """
        Adds args to entityset's elements, checking to make sure no self references are
        made to element ids.
        Ensures Bipartite Condition of EntitySet.

        Parameters
        ----------
        args : One or more entities or hashables

        Returns
        -------
        self : EntitySet

        """
        for item in args:
            if isinstance(item, Entity):
                if item.uid in self.children:
                    raise HyperNetXError(f'Error: Fails the Bipartite Condition for EntitySet. {item.uid} references a child of an existing Entity in the EntitySet.')
                elif not self.uidset.isdisjoint(item.uidset):
                    raise HyperNetXError(f'Error: Fails the bipartite condition for EntitySet.')
                else:
                    Entity.add_element(self, item)
            else:
                if not item in self.children:
                    Entity.add_element(self, item)
                else:
                    raise HyperNetXError(f'Error: {item} references a child of an existing Entity in the EntitySet.')
        return self

    def clone(self, newuid):
        """
        Returns shallow copy of entityset with newuid. Entityset's
        elements will belong to two distinct entitysets.


        Parameters
        ----------
        newuid : hashable
            Name of the new entityset

        Returns
        -------
        clone : EntitySet

        """
        return EntitySet(newuid, elements=self.elements.values(), **self.properties)

    def collapse_identical_elements(self, newuid, use_reps=False, return_counts=False):
        """
        Returns a deduped copy of the entityset, using equivalence classes as element keys.

        Parameters
        ----------
        newuid : hashable

        use_reps : boolean, optional, default: False
            Choose a single element from the collapsed elements as a representative

        return_counts : boolean, optional, default: False
            If use_reps is True the new elements are keyed the size of the equivalence class
            otherwise they are keyed by a frozen set of equivalence classes

        Returns
        -------
        new entityset : EntitySet

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
            >>> E.collapse_identical_elements('_').incidence_dict
            {frozenset({'E1', 'E2'}): {'a', 'b'}}
            >>> E.collapse_identical_elements('_',use_reps=True).incidence_dict
            {'E2': {'a', 'b'}}

        """

        temp = defaultdict(set)
        for e in self.__call__():
            temp[frozenset(e.uidset)].add(e.uid)
        if use_reps:
            if return_counts:
                # labels equivalence class as (rep,count) tuple
                new_entity_dict = {(next(iter(v)), len(v)): set(k) for k, v in temp.items()}
            else:
                # labels equivalence class as rep;
                new_entity_dict = {next(iter(v)): set(k) for k, v in temp.items()}
        else:
            new_entity_dict = {frozenset(v): set(k) for k, v in temp.items()}
        return EntitySet(newuid, new_entity_dict)

    def incidence_matrix(self, sparse=True, index=False):
        """
        An incidence matrix for the EntitySet indexed by children x uidset.

        Parameters
        ----------
        sparse : boolean, optional, default: True

        index : boolean, optional, default False
            If True return will include a dictionary of children uid : row number
            and element uid : column number

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

        ndict = dict(zip(self.children, range(nchildren)))
        edict = dict(zip(self.uidset, range(nuidset)))

        if len(ndict) is not 0:

            if index:
                rowdict = {v: k for k, v in ndict.items()}
                coldict = {v: k for k, v in edict.items()}

            if sparse:
                # Create csr sparse matrix
                rows = list()
                cols = list()
                data = list()
                for e in self:
                    for n in self[e].elements:
                        data.append(1)
                        rows.append(ndict[n])
                        cols.append(edict[e])
                MP = csr_matrix((data, (rows, cols)))
            else:
                # Create an np.matrix
                MP = np.zeros((nchildren, nuidset), dtype=int)
                for e in self:
                    for n in self[e].elements:
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

    def restrict_to(self, element_subset, name=None):
        """
        Shallow copy of entityset removing elements not in element_subset.

        Parameters
        ----------
        element_subset : iterable
            A subset of the entityset's elements

        name: hashable, optional
            If not given, a name is generated to reflect entity uid

        Returns
        -------
        new entityset : EntitySet
            Could be empty.

        See also
        --------
        Entity.restrict_to

        """
        newelements = [self[e] for e in element_subset if e in self]
        name = name or f'{self.uid}_r'
        return EntitySet(name, newelements, **self.properties)
