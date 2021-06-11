# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

from collections import defaultdict
import warnings
from copy import copy
import numpy as np
import networkx as nx
from hypernetx import HyperNetXError

__all__ = ["Entity", "EntitySet"]


class Entity():
    def __init__(self, uid, memberships=[], entity=None, **props):

        self._uid = uid

        # Copies an entity if entity is passed into the constructor
        if entity is not None:
            if isinstance(entity, Entity):
                if uid == entity.uid:
                    raise HyperNetXError(
                        "The new entity will be indistinguishable from the original with the same uid. Use a differen uid."
                    )
                self._memberships = entity.memberships
                self.__dict__.update(entity.properties)
        else:
            # If no entity is provided, we must be building one up from scratch
            self._memberships = memberships

        # add properties to the class dictionary and get the levelset(2) which is all the child entities which comprises the registry
        self.__dict__.update(props)

    @property
    def properties(self):
        """Dictionary of properties of entity"""
        temp = self.__dict__.copy()
        del temp["_memberships"]
        del temp["_uid"]
        return temp

    @property
    def uid(self):
        """String identifier for entity"""
        # each entity has an associated ID
        return self._uid

    @property
    def memberships(self):
        """
        Dictionary of elements to which entity belongs.

        This assignment is done on construction and controlled by
        :func:`Entity.add_element()`
        and :func:`Entity.remove_element()` methods.
        """
        # this dictionary comprehension looks like it is not stored, but computed every time
        return self._memberships

    # defines equality to be whether all the state variables match.
    def __eq__(self, other):
        """
        Defines equality for Entities based on equivalence of their __dict__ objects.

        Checks all levels of self and other to verify they are
        referencing the same uids and that they have the same set of properties.
        If at any point we get duplicate addresses we stop checking that branch
        because we are guaranteed equality from there on.

        May cause a recursion error if depth is too great.
        """
        seen = set()
        # Define a compare method to call recursively on each level of self and other

        def _comp(a, b, seen):
            # Compare top level properties: same class? same ids? same children? same parents? same attributes?
            if (
                (a.__class__ != b.__class__)
                or (a.uid != b.uid)
                or (a.properties != b.properties)
                or (a.memberships != b.memberships)
            ):
                return False
            # If all agree then look at the next level down since a and b share uidsets.
            for uid, elt in a.elements.items():
                if isinstance(elt, Entity):
                    if uid in seen:
                        continue
                    seen.add(uid)
                    if not _comp(elt, b[uid], seen):
                        return False
                # if not an Entity then elt is hashable so we usual equality
                elif elt != b[uid]:
                    return False
            return True

        return _comp(self, other, seen)

    def __len__(self):
        """Returns the number of elements in entity"""
        return len(self._memberships)

    def __str__(self):
        """Return the entity uid."""
        return f"{self.uid}"

    def __repr__(self):
        """Returns a string resembling the constructor for entity without any
        children"""
        return f"Entity({self._uid}),{list(self.memberships)},{self.properties})"

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
        return item in self._memberships

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
        return self._memberships.get(item)

    def __iter__(self):
        """Returns iterator on membership ids."""
        return iter(self._memberships)

    def __call__(self):
        """Returns an iterator on memberships"""
        for e in self._memberships:
            yield e

    def size(self):
        """
        Returns the number of elements in membership
        """
        return len(self)

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


class EntitySet(Entity):
    def __init__(self, uid, elements=dict(), **props):
        self._uid = uid
        self._elements = dict()
        if isinstance(elements, dict):
            for id, item in elements.items():
                if isinstance(item, Entity):
                    self.add_element(id, item)
                else:
                    self.add_element(id, Entity(id, item))
        self._children = self.children

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
            return self._elements.get(item.uid, "")
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

    @property
    def is_empty(self):
        """Boolean indicating if entity.elements is empty"""
        return len(self) == 0

    @property
    def uidset(self):
        """
        Set of uids of elements of entity.
        """
        return frozenset(self._elements.keys())

    @property
    def incidence_dict(self):
        """
        Dictionary of element.uid:element.uidset for each element in entity

        To return an incidence dictionary of all nested entities in entity
        use nested_incidence_dict
        """
        temp = dict()
        for entity in self.elements.values():
            temp[entity.uid] = {item for item in entity.memberships}
        return temp

    @property
    def elements(self):
        return self._elements

    @property
    def children(self):
        children = set()
        for items in self._elements.values():
            children.update(items.memberships)
        return children

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

    def get_dual(self):
        memberships = defaultdict(list)
        for id in self.elements:
            for member in self.elements[id].memberships:
                memberships[member].append(id)
        return memberships

    def add_element(self, id, item):
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
        if isinstance(item, Entity):
            if id in self.elements:
                raise HyperNetXError(
                    f"Error: {id} references an existing Entity in the EntitySet."
                )
            self.elements[item.uid] = item
        elif isinstance(item, list):
            if id in self.elements:
                raise HyperNetXError(
                    f"Error: {id} references an existing Entity in the EntitySet."
                )
            self.elements[id] = Entity(id, item)

    def add_elements_from(self, elements):
        """
        Adds items to entityset's elements, checking to make sure no self references are
        made to element ids.
        Ensures Bipartite Condition of EntitySet.

        Parameters
        ----------
        args : dictionary

        Returns
        -------
        self : EntitySet

        """
        for id, item in elements.items():
            self.add_element(id, item)

    def add(self, elements):
        """
        Adds items to entityset's elements, checking to make sure no self references are
        made to element ids.
        Ensures Bipartite Condition of EntitySet.

        Parameters
        ----------
        args : dictionary

        Returns
        -------
        self : EntitySet

        """
        for id, item in elements.items():
            self.add_element(id, item)

    def remove_element(self, item):
        """
        Removes item from entity and reference to entity from
        item.memberships

        Parameters
        ----------
        item : Hashable or Entity

        Returns
        -------
        self : Entity


        """
        if isinstance(item, Entity):
            memberships = item.memberships
            del self._elements[item.uid]
            return {item.uid : memberships}
        else:
            memberships = self._elements[item].memberships
            del self._elements[item]
            return {item : memberships}
    

    def remove_elements_from(self, arg_set):
        """
        Similar to :func:`remove()`. Removes elements in arg_set.

        Parameters
        ----------
        arg_set : Iterable of hashables or entities

        Returns
        -------
        self : Entity

        """
        memberships = dict()
        for item in arg_set:
            Entity.remove_element(self, item)
        return self
    

    def remove(self, *args):
        """
        Removes args from entitie's elements if they belong.
        Does nothing with args not in entity.

        Parameters
        ----------
        args : One or more hashables or entities

        Returns
        -------
        self : Entity


        """
        for item in args:
            Entity.remove_element(self, item)
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
        return EntitySet(newuid, elements=self.elements, **self.properties)


    def collapse_identical_elements(self, newuid, return_equivalence_classes=False):
        """
        Returns a deduped copy of the entityset, using representatives of equivalence classes as element keys.
        Two elements of an EntitySet are collapsed if they share the same children.

        THIS SHOULD BE CHANGED TO IF THEY HAVE THE SAME MEMBERS

        Parameters
        ----------
        newuid : hashable

        return_equivalence_classes : boolean, default=False
            If True, return a dictionary of equivalence classes keyed by new edge names

        Returns
        -------
         : EntitySet
        eq_classes : dict
            if return_equivalence_classes = True

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
            shared_children[frozenset(e.uidset)].add(e.uid)
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

        if len(ndict) != 0:

            if index:
                rowdict = {v: k for k, v in ndict.items()}
                coldict = {v: k for k, v in edict.items()}

            if sparse:
                # Create csr sparse matrix
                rows = list()
                cols = list()
                data = list()
                for e in self:
                    for n in self.elements[e].memberships:
                        data.append(1)
                        rows.append(ndict[n])
                        cols.append(edict[e])
                MP = csr_matrix((data, (rows, cols)))
            else:
                # Create an np.matrix
                MP = np.zeros((nchildren, nuidset), dtype=int)
                for e in self:
                    for n in self.elements[e].memberships:
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

