# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

from collections import defaultdict
import warnings
from copy import copy
import numpy as np
import networkx as nx
from hypernetx import HyperNetXError

__all__ = [
    'Entity',
    'EntitySet'
]


class Entity(object):
    """
    Base class for objects used in building network-like objects including
    Hypergraphs, Posets, Cell Complexes.

    Parameters
    ----------
    uid : hashable
        a unique identifier

    elements : list or dict, optional, default: None
        a list of entities with identifiers different than uid and/or
        hashables different than uid, see `Honor System`_

    entity : Entity
        an Entity object to be cloned into a new Entity with uid. If the uid is the same as 
        Entity.uid then the entities will not be distinguishable and error will be raised. 
        The `elements` in the signature will be added to the cloned entity.

    props : keyword arguments, optional, default: {}
        properties belonging to the entity added as key=value pairs.
        Both key and value must be hashable.

    Notes
    -----

    An Entity is a container-like object, which has a unique identifier and
    may contain elements and have properties.
    The Entity class was created as a generic object providing structure for
    Hypergraph nodes and edges.

    - An Entity is distinguished by its identifier (sortable,hashable) :func:`Entity.uid`
    - An Entity is a container for other entities but may not contain itself, :func:`Entity.elements`
    - An Entity has properties :func:`Entity.properties`
    - An Entity has memberships to other entities, :func:`Entity.memberships`.
    - An Entity has children, :func:`Entity.children`, which are the elements of its elements.
    - :func:`Entity.children` are registered in the :func:`Entity.registry`.
    - All descendents of Entity are registered in :func:`Entity.fullregistry()`.

    .. _Honor System:

    **Honor System**

    HyperNetX has an Honor System that applies to Entity uid values.
    Two entities are equal if their __dict__ objects match.
    For performance reasons many methods distinguish entities by their uids.
    It is, therefore, up to the user to ensure entities with the same uids are indeed the same.
    Not doing so may cause undesirable side effects.
    In particular, the methods in the Hypergraph class assume distinct nodes and edges
    have distinct uids.

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
        {'x': Entity(x,[],{}), 'y': Entity(y,['x'],{})}
        >>> z.properties
        {'weight': 1}
        >>> z.children
        {'x'}
        >>> x.memberships
        {'y': Entity(y,['x'],{}), 'z': Entity(z,['y', 'x'],{'weight': 1})}
        >>> z.fullregistry()
        {'x': Entity(x,[],{}), 'y': Entity(y,['x'],{})}


    See Also
    --------
    EntitySet

    """

    def __init__(self, uid, elements=[], entity=None, **props):
        super().__init__()

        self._uid = uid

        if entity is not None:
            if isinstance(entity, Entity):
                if uid == entity.uid:
                    raise HyperNetXError('The new entity will be indistinguishable from the original with the same uid. Use a differen uid.')
                self._elements = entity.elements
                self._memberships = entity.memberships
                self.__dict__.update(entity.properties)
        else:
            self._elements = dict()
            self._memberships = dict()
            self._registry = dict()

        self.__dict__.update(props)
        self._registry = self.registry

        if isinstance(elements, dict):
            for k, v in elements.items():
                if isinstance(v, Entity):
                    self.add_element(v)
                else:
                    self.add_element(Entity(k, v))
        elif elements is not None:
            self.add(*elements)

    @property
    def properties(self):
        """Dictionary of properties of entity"""
        temp = self.__dict__.copy()
        del temp['_elements']
        del temp['_memberships']
        del temp['_registry']
        del temp['_uid']
        return temp

    @property
    def uid(self):
        """String identifier for entity"""
        return self._uid

    @property
    def elements(self):
        """
        Dictionary of elements belonging to entity.
        """
        return dict(self._elements)

    @property
    def memberships(self):
        """
        Dictionary of elements to which entity belongs.

        This assignment is done on construction and controlled by
        :func:`Entity.add_element()` 
        and :func:`Entity.remove_element()` methods.
        """

        return {k: self._memberships[k] for k in self._memberships
                if not isinstance(self._memberships[k], EntitySet)}

    @property
    def children(self):
        """
        Set of uids of the elements of elements of entity.

        To return set of ids for deeper level use:
        :code:`Entity.levelset(level).keys()`
        see: :func:`Entity.levelset`
        """
        return set(self.levelset(2).keys())

    @property
    def registry(self):
        """
        Dictionary of uid:Entity pairs for children entity.

        To return a dictionary of all entities at all depths
        :func:`Entity.complete_registry()`
        """
        return self.levelset(2)

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
        for ent in self.elements.values():
            temp[ent.uid] = {item for item in ent.elements}
        return temp

    @property
    def is_empty(self):
        """Boolean indicating if entity.elements is empty"""
        return len(self) == 0

    @property
    def is_bipartite(self):
        """
        Returns boolean indicating if the entity satisfies the `Bipartite Condition`_
        """
        if self.uidset.isdisjoint(self.children):
            return True
        else:
            return False

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
            if (a.__class__ != b.__class__) or (a.uid != b.uid) or (a.uidset != b.uidset) or (a.properties != b.properties) or (a.memberships != b.memberships):
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
        return len(self._elements)

    def __str__(self):
        """Return the entity uid."""
        return f'{self.uid}'

    def __repr__(self):
        """Returns a string resembling the constructor for entity without any
        children"""
        return f'Entity({self._uid},{list(self.uidset)},{self.properties})'

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
            return self._elements.get(item.uid, '')
        else:
            return self._elements.get(item)

    def __iter__(self):
        """Returns iterator on element ids."""
        return iter(self.elements)

    def __call__(self):
        """Returns an iterator on elements"""
        for e in self.elements.values():
            yield e

    def __setattr__(self, k, v):
        """Sets entity property.

        Parameters
        ----------
        k : hashable, property key
        v : hashable, property value
            Will not set uid or change elements or memberships.

        Returns
        -------
        None

        """
        if k == 'uid':
            raise HyperNetXError(
                'Cannot reassign uid to Entity once it'
                ' has been created. Create a clone instead.'
            )
        elif k == 'elements':
            raise HyperNetXError('To add elements to Entity use self.add().')
        elif k == 'memberships':
            raise HyperNetXError(
                'Can\'t choose your own memberships, '
                'they are like parents!'
            )
        else:
            self.__dict__[k] = v

    def _depth_finder(self, entset=None):
        """
        Helper method when working with levels.

        Parameters
        ----------
        entset : dict, optional
            a dictionary of entities keyed by uid

        Returns
        -------
        Dictionary extending entset
        """
        temp = dict()
        for uid, item in entset.items():
            temp.update(item.elements)
        return temp

    def level(self, item, max_depth=10):
        """
        The first level where item appears in self.

        Parameters
        ----------
        item : hashable
            uid for an entity

        max_depth : int, default: 10
            last level to check for entity

        Returns
        -------
        level : int

        Note
        ----
        Item must be the uid of an entity listed
        in :func:`fullregistry()`
        """
        d = 1
        currentlevel = self.levelset(1)
        while d <= max_depth + 1:
            if item in currentlevel:
                return d
            else:
                d += 1
                currentlevel = self._depth_finder(currentlevel)
        return None

    def levelset(self, k=1):
        """
        A dictionary of level k of self.

        Parameters
        ----------
        k : int, optional, default: 1

        Returns
        -------
        levelset : dict

        Note
        ----
        An Entity contains other entities, hence the relationships between entities
        and their elements may be represented in a directed graph with entity as root.
        The levelsets are sets of entities which make up the elements appearing at
        a certain level.
        """
        if k <= 0:
            return None
        currentlevel = self.elements
        if k > 1:
            for idx in range(k - 1):
                currentlevel = self._depth_finder(currentlevel)
        return currentlevel

    def depth(self, max_depth=10):
        """
        Returns the number of nonempty level sets of level <= max_depth

        Parameters
        ----------
        max_depth : int, optional, default: 10
            If full depth is desired set max_depth to number of entities in
            system + 1.

        Returns
        -------
        depth : int
            If max_depth is exceeded output will be numpy infinity.
            If there is a cycle output will be numpy infinity.

        """
        if max_depth < 0:
            return 0
        currentlevel = self.elements
        if not currentlevel:
            return 0
        else:
            depth = 1
        while depth < max_depth + 1:
            currentlevel = self._depth_finder(currentlevel)
            if not currentlevel:
                return depth
            depth += 1
        return np.inf

    def fullregistry(self, lastlevel=10, firstlevel=1):
        """
        A dictionary of all entities appearing in levels firstlevel
        to lastlevel.

        Parameters
        ----------
        lastlevel : int, optional, default: 10

        firstlevel : int, optional, default: 1

        Returns
        -------
        fullregistry : dict

        """
        currentlevel = self.levelset(firstlevel)
        accumulater = dict(currentlevel)
        for idx in range(firstlevel, lastlevel):
            currentlevel = self._depth_finder(currentlevel)
            accumulater.update(currentlevel)
        return accumulater

    def complete_registry(self):
        """
        A dictionary of all entities appearing in any level of
        entity

        Returns
        -------
        complete_registry : dict
        """
        results = dict()
        Entity._complete_registry(self, results)
        return results

    @staticmethod
    def _complete_registry(entity, results):
        """
        Helper method for complete_registry
        """
        for uid, e in entity.elements.items():
            if uid not in results:
                results[uid] = e
                Entity._complete_registry(e, results)

    def nested_incidence_dict(self, level=10):
        """
        Returns a nested dictionary with keys up to level

        Parameters
        ----------
        level : int, optional, default: 10
            If level<=1, returns the incidence_dict.

        Returns
        -------
        nested_incidence_dict : dict

        """
        if level > 1:
            return {ent.uid: ent.nested_incidence_dict(level - 1) for ent in self()}
        else:
            return self.incidence_dict

    def size(self):
        """
        Returns the number of elements in entity
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

    def add(self, *args):
        """
        Adds unpacked args to entity elements. Depends on add_element()

        Parameters
        ----------
        args : One or more entities or hashables

        Returns
        -------
        self : Entity

        Note
        ----
        Adding an element to an object in a hypergraph will not add the 
        element to the hypergraph and will cause an error. Use :func:`Hypergraph.add_edge <classes.hypergraph.Hypergraph.add_edge>`
        or :func:`Hypergraph.add_node_to_edge <classes.hypergraph.Hypergraph.add_node_to_edge>` instead.

        """
        for item in args:
            self.add_element(item)

        return self

    def add_elements_from(self, arg_set):
        """
        Similar to :func:`add()` it allows for adding from an interable.

        Parameters
        ----------
        arg_set : Iterable of hashables or entities

        Returns
        -------
        self : Entity

        """
        for item in arg_set:
            self.add_element(item)

        return self

    def add_element(self, item):
        """
        Adds item to entity elements and adds entity to item.memberships.

        Parameters
        ----------
        item : hashable or Entity
            If hashable, will be replaced with empty Entity using hashable as uid

        Returns
        -------
        self : Entity

        Notes
        -----
        If item is in entity elements, no new element is added but properties
        will be updated.
        If item is in complete_registry(), only the item already known to self will be added.
        This method employs the `Honor System`_ since membership in complete_registry is checked
        using the item's uid. It is assumed that the user will only use the same uid
        for identical instances within the entities registry.

        """
        checkelts = self.complete_registry()
        if isinstance(item, Entity):
            # if item is an Entity, descendents will be compared to avoid collisions
            if item.uid == self.uid:
                raise HyperNetXError(
                    f'Error: Self reference in submitted elements.'
                    f' Entity {self.uid} may not contain itself. '
                )
            elif item in self:
                # item is already an element so only the properties will be updated
                checkelts[item.uid].__dict__.update(item.properties)
            elif item.uid in checkelts:
                # if item belongs to an element or a descendent of an element
                # then the existing descendent becomes an element
                # and properties are updated.
                checkelts[item.uid]._memberships[self.uid] = self
                checkelts[item.uid].__dict__.update(item.properties)
                self._elements[item.uid] = checkelts[item.uid]
            else:
                # if item's uid doesn't appear in complete_registry
                # then it is added as something new
                item._memberships[self.uid] = self
                self._elements[item.uid] = item
        else:
            # item must be a hashable.
            # if it appears as a uid in checkelts then
            # the corresponding Entity will become an element of entity.
            # Otherwise, at most it will be added as an empty Entity.
            if self.uid == item:
                raise HyperNetXError(
                    f'Error: Self reference in submitted elements.'
                    f' Entity {self.uid} may not contain itself.'
                )
            elif item not in self._elements:
                if item in checkelts:
                    self._elements[item] = checkelts[item]
                    checkelts[item]._memberships[self.uid] = self
                else:
                    self._elements[item] = \
                        Entity(item, _memberships={self.uid: self})

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
        for item in arg_set:
            Entity.remove_element(self, item)
        return self

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
            del item._memberships[self.uid]
            del self._elements[item.uid]
        else:
            del self[item]._memberships[self.uid]
            del self._elements[item]

        return self

    @staticmethod
    def merge_entities(name, ent1, ent2):
        """
        Merge two entities making sure they do not conflict.

        Parameters
        ----------
        name : hashable

        ent1 : Entity
            First entity to have elements and properties added to new
            entity

        ent2 : Entity
            elements of ent2 will be checked against ent1.complete_registry()
            and only nonexisting elements will be added using add() method.
            Properties of ent2 will update properties of ent1 in new entity.

        Returns
        -------
        a new entity : Entity

        """
        newent = ent1.clone(name)
        newent.add_elements_from(ent2.elements.values())
        for k, v in ent2.properties.items():
            newent.__setattr__(k, v)
        return newent


class EntitySet(Entity):
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

    def collapse_identical_elements(self, newuid, return_equivalence_classes=False):
        """
        Returns a deduped copy of the entityset, using representatives of equivalence classes as element keys.
        Two elements of an EntitySet are collapsed if they share the same children.

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
        new_entity_dict = {f"{next(iter(v))}:{len(v)}": set(k) for k, v in shared_children.items()}
        if return_equivalence_classes:
            eq_classes = {f"{next(iter(v))}:{len(v)}": v for k, v in shared_children.items()}
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
