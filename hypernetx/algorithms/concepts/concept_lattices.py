import functools
import heapq
import typing
import bitsets
import operator
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

"""
This module implements the HypergraphLattice class, which computes the concept lattice of the incidence relation of a hypergraph.

This code is based on the Concepts python package by Sebastian Bank:
https://github.com/xflr6/concepts
"""


class Context:

    def __init__(
        self,
        objects: typing.Iterable[str],
        properties: typing.Iterable[str],
        bools: typing.Iterable[typing.Tuple[bool, ...]],
    ) -> None:
        """
        Constructs a context from a collection objects, a collection of properties,
        and a binary relation between objects and properties.

        Parameters
        ----------
        objects : iterable(str)
            Iterable of object strings.
        properties : iterable(str)
            Iterable of property strings.
        bools : iterable(tuple(bool))
            Iterable of 'len(objects)' tuples of 'len(properties)' booleans.

        Returns
        -------
        None
        """
        objects, properties = map(tuple, (objects, properties))

        for items, name in [(objects, "objects"), (properties, "properties")]:
            if not items:
                raise ValueError(f"empty {name}")
            if len(set(items)) != len(items):
                raise ValueError(f"duplicate {name}: {items!r}")

        if not set(objects).isdisjoint(properties):
            common = set(objects) & set(properties)
            raise ValueError(f"objects and properties overlap: {common!r}")

        if len(bools) != len(objects) or {len(b) for b in bools} != {len(properties)}:
            raise ValueError(
                f"bools is not {len(objects)} items" f" of length {len(properties)}"
            )

        self._intents, self._extents = Relation(
            "Properties", "Objects", properties, objects, bools
        )
        self._Properties = self._intents.BitSet
        self._Objects = self._extents.BitSet

    def __getitem__(
        self, items: typing.Iterable[str], raw: bool = False
    ) -> typing.Tuple[typing.Tuple[str, ...], typing.Tuple[str, ...]]:
        """
        A function to obtain the (extension, intension) pair from a context.

        Parameters
        ----------
        items : iterable(str)
            Iterable of string labels either taken from 'self.objects' or from 'self.properties'.
        raw: bool
            Return raw (extent, intent) pair instead of string tuples.

        Returns
        -------
        hnx.concept_lattice.Concept
            The smallest concept having all 'items' as (extent, intent) pair.
        """
        try:
            extent = self._Objects.frommembers(items)
        except KeyError:
            intent = self._Properties.frommembers(items)
            intent, extent = intent.doubleprime()
        else:
            extent, intent = extent.doubleprime()

        if raw:
            return extent, intent
        return extent.members(), intent.members()

    def intension(
        self, objects: typing.Iterable[str], raw: bool = False
    ) -> typing.Tuple[str, ...]:
        """
        Return all properties shared by the given 'objects'.

        Parameters
        ----------
        objects : Iterable of string labels taken from self.objects.
        raw : Return raw intent instead of string tuple.

        Returns
        -------
        tuple(str)
            A tuple of string labels taken from self.properties.
        """
        intent = self._Objects.frommembers(objects).prime()
        if raw:
            return intent
        return intent.members()

    def extension(
        self, properties: typing.Iterable[str], raw: bool = False
    ) -> typing.Tuple[str, ...]:
        """
        Return all objects sharing the given properties.

        Parameters
        ----------
        properties : iterable(str)
            Iterable of string labels taken from self.properties.
        raw : bool, optional
            Return raw extent instead of string tuple.

        Returns
        -------
        tuple(str)
            A tuple of string labels taken from self.objects.
        """
        extent = self._Properties.frommembers(properties).prime()
        if raw:
            return extent
        return extent.members()

    def _lattice(self):
        """
        Yield (extent, intent, upper, lower) in short lexicographic order
        for every concept of the context.

        Yields
        -------
        tuple(extent, intent, upper, lower)
            extent : bitset.meta.bitsets
            intent : bitset.meta.bitsets
            upper : tuple(hnx.concept_lattices.Concept)
            lower : tuple(hnx.concept_lattices.Concept)
        """
        return LatticeAlgorithms.lattice_element(self._Objects)

    @property
    def objects(self) -> typing.Tuple[str, ...]:
        """
        Names of the objects described by the context.

        Returns
        -------
        tuple(str)
        """
        return self._Objects._members

    @property
    def properties(self) -> typing.Tuple[str, ...]:
        """
        Names of the properties that describe the objects.

        Returns
        -------
        tuple(str)
        """
        return self._Properties._members


# bitset storage of object, attribute lists and relations.

BoolVector = bitsets.bases.MemberBits
"""Single row or column of a boolean matrix as bit vector."""


class BoolVectorPair(bitsets.series.Tuple):

    def _pair_with(self, relation, index, other):
        """
        Paired collection of rows or columns of a boolean matrix relation.
        Computes the FCA derivation operations with respect to the given
        relation.

        Paramters
        ---------
        relation : hnx.concept_lattices.Relation
        index : int
        other : hnx.concept_lattices.Relation

        Returns
        -------
        None
        """

        if hasattr(self, "prime"):
            raise RuntimeError(f"{self!r} attempt _pair_with {other!r}")

        self.relation = relation
        self.relation_index = index

        Prime = other.BitSet.supremum
        Double = self.BitSet.supremum

        make_prime = other.BitSet.fromint
        make_double = self.BitSet.fromint

        def prime(bitset):
            """Formal Concept Analysis derivation operator (extent->intent, intent->extent)."""
            prime = Prime

            i = 0
            while bitset:
                shift = (bitset & -bitset).bit_length() - 1  # trailing zero(s)
                if not shift:
                    shift = 1
                    prime &= other[i]
                i += shift
                bitset >>= shift

            return make_prime(prime)

        def double(bitset):
            """FCA double derivation operator (extent->extent, intent->intent)."""
            prime = Prime

            i = 0
            while bitset:
                shift = (bitset & -bitset).bit_length() - 1
                if not shift:
                    shift = 1
                    prime &= other[i]
                i += shift
                bitset >>= shift

            double = Double

            i = 0
            while prime:
                shift = (prime & -prime).bit_length() - 1
                if not shift:
                    shift = 1
                    double &= self[i]
                i += shift
                prime >>= shift

            return make_double(double)

        def doubleprime(bitset):
            """FCA single and double derivation (extent->extent+intent, intent->intent+extent)."""
            prime = Prime

            i = 0
            while bitset:
                shift = (bitset & -bitset).bit_length() - 1
                if not shift:
                    shift = 1
                    prime &= other[i]
                i += shift
                bitset >>= shift

            bitset = prime
            double = Double

            i = 0
            while bitset:
                shift = (bitset & -bitset).bit_length() - 1
                if not shift:
                    shift = 1
                    double &= self[i]
                i += shift
                bitset >>= shift

            return make_double(double), make_prime(prime)

        self.prime = self.BitSet.prime = prime
        self.double = self.BitSet.double = double
        self.doubleprime = self.BitSet.doubleprime = doubleprime

    def __reduce__(self):
        """Serialization for pickling."""
        return self.relation, (self.relation_index,)


class Relation(tuple):
    """
    This class implements binary relations as an interconnected pair of bitset collections.
    """

    __slots__ = ()

    def __new__(cls, xname, yname, xmembers, ymembers, xbools, _ids=None):
        if _ids is not None:  # unpickle reconstruction
            xid, yid = _ids
            X = bitsets.meta.bitset(
                xname, xmembers, xid, BoolVector, None, BoolVectorPair
            )
            Y = bitsets.meta.bitset(
                yname, ymembers, yid, BoolVector, None, BoolVectorPair
            )
        else:
            X = bitsets.bitset(xname, xmembers, BoolVector, tuple=BoolVectorPair)
            Y = bitsets.bitset(yname, ymembers, BoolVector, tuple=BoolVectorPair)

        x = X.Tuple.frombools(xbools)
        y = Y.Tuple.frombools(zip(*x.bools()))

        self = super().__new__(cls, (x, y))

        x._pair_with(self, 0, y)
        y._pair_with(self, 1, x)

        return self

    __call__ = tuple.__getitem__

    def __reduce__(self):
        """Serialization for pickling."""
        X, Y = (v.BitSet for v in self)
        return (
            self.__class__,
            (
                X.__name__,
                Y.__name__,
                X._members,
                Y._members,
                self[0].bools(),
                (X._id, Y._id),
            ),
        )


class Concept:
    objects = ()
    properties = ()

    def __init__(self, lattice, extent, intent, upper, lower) -> None:
        self.lattice = lattice  #: The lattice containing the concept.
        self._extent = extent
        self._intent = intent
        self.upper_neighbors = upper  #: The directly implied concepts.
        self.lower_neighbors = lower  #: The directly subsumed concepts.

    @property
    def extent(self) -> typing.Tuple[str, ...]:
        """
        Objects of the concept.
        """
        return self._extent.members()

    @property
    def intent(self) -> typing.Tuple[str, ...]:
        """
        Properties of the concept.
        """
        return self._intent.members()

    def upset(self):
        """
        Yield concepts lower in lattice order (including 'self').

        Yields
        ------
        hnx.concept_lattice.Concept
        """
        sortkey = operator.attrgetter("index")
        next_concepts = operator.attrgetter("upper_neighbors")
        return LatticeAlgorithms.iterunion([self], sortkey, next_concepts)

    def downset(self):
        """
        Yield concepts higher in lattice order (including 'self').

        Yields
        ------
        hnx.concept_lattice.Concept
        """
        sortkey = operator.attrgetter("dindex")
        next_concepts = operator.attrgetter("lower_neighbors")
        return LatticeAlgorithms.iterunion([self], sortkey, next_concepts)

    def __str__(self) -> str:
        extent = ", ".join(self._extent.members())
        intent = ", ".join(self._intent.members())
        return f"{{{extent}}} : [{intent}]"

    def shortform_string(self) -> str:
        objects = "{}".format(" ".join(self.objects)) if self.objects else ""
        properties = "{}".format(" ".join(self.properties)) if self.properties else ""
        return f"{objects} : {properties}"


class Lattice:
    @staticmethod
    def _annotate(context, mapping):
        """
        Helper method to annotate concepts with their objects/properties.

        Parameters
        ----------
        context : hnx.concept_lattices.Context
        mapping : dict
            The keys in the mapping dictionary are bitsets of extents of 'context'.
            The values are the hnx.concept_lattices.Concept objects constructed from 'context'.

        Returns
        -------
        None
        """
        touched = set()
        for o in context.objects:
            extent = context.extension(context.intension([o]), raw=True)
            c = mapping[extent]
            if c.objects:
                c.objects.append(o)
            else:
                c.objects = [o]
                touched.add(c)

        for c in touched:
            c.objects = tuple(c.objects)

        touched = set()
        for p in context.properties:
            extent = context.extension([p], raw=True)
            c = mapping[extent]
            if c.properties:
                c.properties.append(p)
            else:
                c.properties = [p]
                touched.add(c)

        for c in touched:
            c.properties = tuple(c.properties)

    @property
    def infimum(self):
        """
        The smallest element of the lattice.

        Returns
        -------
        hnx.concept_lattice.Concept
        """
        return self._concepts[0]

    @property
    def supremum(self):
        """
        The largest element of the lattice.

        Returns
        -------
        hnx.concept_lattice.Concept
        """
        return self._concepts[-1]

    @property
    def atoms(self):
        """
        The minimal non-infimum (i.e. atomic) concepts of the lattice.

        Returns
        -------
        tuple(hnx.concept_lattice.Concept)
        """
        return self.infimum.upper_neighbors

    @staticmethod
    def _longlex(concept):
        return concept._extent.longlex()

    @staticmethod
    def _shortlex(concept):
        return concept._extent.shortlex()

    def __init__(self, context) -> None:
        """
        Create the concept lattice object from a context.

        Parameters
        ----------
        context : hnx.concept_lattices.Context

        Return
        ------
        None
        """
        concepts = [Concept(self, *args) for args in context._lattice()]
        mapping = self._make_mapping(concepts)

        shortlex = self._shortlex
        longlex = self._longlex
        for index, c in enumerate(concepts):
            c.index = index
            upper = (mapping[u] for u in c.upper_neighbors)
            lower = (mapping[l] for l in c.lower_neighbors)
            c.upper_neighbors = tuple(sorted(upper, key=shortlex))
            c.lower_neighbors = tuple(sorted(lower, key=longlex))

        self._init(self, context, concepts, mapping=mapping)

    @staticmethod
    def _make_mapping(concepts):
        return {c._extent: c for c in concepts}

    @staticmethod
    def _init(inst, context, concepts, mapping=None) -> None:
        inst._context = context
        inst._concepts = concepts

        if mapping is None:
            mapping = inst._make_mapping(inst._concepts)
        inst._mapping = mapping

        # downward
        atoms = inst.atoms
        for dindex, c in enumerate(sorted(inst._concepts, key=inst._longlex)):
            c.dindex = dindex
            e = c._extent
            c.atoms = tuple(a for a in atoms if e | a._extent == e)

        inst._annotate(inst._context, inst._mapping)

    def __str__(self) -> str:
        """Return the full string representation of the lattice."""
        concepts = "\n".join(f"    {c}" for c in self._concepts)
        return f"{self!r}\n{concepts}"

    def join(self, concepts: typing.Iterable[Concept]) -> Concept:
        """
        Computes the lattice join operation of an iterable of concepts from the Lattice.

        Parameters
        ----------
        concepts : iterable(hnx.concept_lattices.Concept)
            An iterable of concept instance from this lattice.

        Returns
        -------
        hnx.concept_lattices.Concept
        """
        extents = (c._extent for c in concepts)
        join = self._context._Objects.reduce_or(extents)
        return self._mapping[join.double()]

    def meet(self, concepts: typing.Iterable[Concept]) -> Concept:
        """
        Computes the lattice meet operation of an iterable of concepts from the Lattice.

        Parameters
        ----------
        concepts : iterable(hnx.concept_lattices.Concept)
            An iterable of concept instance from this lattice.

        Returns
        -------
        hnx.concept_lattices.Concept
        """
        extents = (c._extent for c in concepts)
        meet = self._context._Objects.reduce_and(extents)
        return self._mapping[meet.double()]

    def __getitem__(self, key: typing.Union[int, typing.Tuple[str, ...]]) -> Concept:
        """
        Retrieve concept from the lattice by its index, intension, or extension.

        Parameters
        ----------
        key : int, int slice, or tuple(str).
            If an integer or slice is passed, then this method retrieves the concept at that index/slice as stored in self._concepts.
            If a tuple of strings is passed, then they must be strings strictly of objects or strcitly of properties.
            The tuple need not be an extent or intent, but the concept returned will be the smallest concept
            containing the given tuple as an extent/intent.

        Returns
        -------
        hnx.concept_lattices.Concept
        """
        if isinstance(key, (int, slice)):
            return self._concepts[key]

        if not key:
            return self.supremum

        extent, intent = self._context.__getitem__(key, raw=True)
        return self._mapping[extent]

    def __iter__(self) -> typing.Iterator[Concept]:
        """
        Iterable to retrieve all the concepts of a lattice.

        Yields
        ------
        hnx.concept_lattice.Concept
        """
        return iter(self._concepts)

    def __len__(self) -> int:
        """
        Return the number of concepts in the lattice.

        Returns
        -------
        int
        """
        return len(self._concepts)

    def join_irreducibles(self):
        """
        Compute all join-irreducible elements of the lattice.

        Returns
        -------
        list
        """

        join_reducibles = set()

        atoms = self.atoms
        for x in atoms:
            for y in atoms:
                if x != y:
                    join_reducibles.add(self.join([x, y]))

        join_irreducibles = set(self._concepts).difference(join_reducibles)
        return join_irreducibles.union(atoms)

    def to_networkx(self):
        """
        Constructs directed acyclic graph representation of the lattice.

        Returns
        -------
        networkx.DiGraph
        """
        G = nx.DiGraph()
        for i in range(len(self._concepts)):
            G.add_node(i, concept=self._concepts[i])
        concepts = nx.get_node_attributes(G, "concept")
        for node in G.nodes:
            curr_concept = concepts[node]
            for concept in curr_concept.upper_neighbors:
                nghbr = list(concepts.keys())[list(concepts.values()).index(concept)]
                G.add_edge(node, nghbr)
        return G


class LatticeAlgorithms:

    def iterunion(concepts, sortkey, next_concepts):
        """
        Yields concept objects according to specified assortment.

        Parameters
        ----------
        concepts : iterabel(hnx.concept_lattices.Concept)
            iterable of Concept objects
        sortkey : function
            input: concept_lattices.Concept
            output: str???
        next_concepts : function
            input: concept_lattices.Concept
            output: list(concept_lattices.Concept)

        Yields
        -------
        hnx.concept_lattices.Concept
        """
        heap = [(sortkey(c), c) for c in concepts]
        heapq.heapify(heap)

        push = functools.partial(heapq.heappush, heap)
        pop = functools.partial(heapq.heappop, heap)

        seen = -1

        while heap:
            index, concept = pop()
            # requires sortkey to be an extension of the lattice order
            # (a toplogical sort of it) in the direction of next_concepts
            # assert index >= seen
            if index > seen:
                seen = index
                yield concept
                for c in next_concepts(concept):
                    push((sortkey(c), c))

    def lattice_element(context_objects):
        """
        A function that yields each concept associated to context_objects.
        Algorithm is derived from C. Lindig. 2000. Fast Concept Analysis

        Parameters
        ----------
        context_objects : hnx.concept_lattices.BoolVectorPair
            BoolVectorPair instance representing the objects of a context.

        Yields
        -------
        tuple(extent, intent, upper, lower)
            extent : bitset.meta.bitsets
            intent : bitset.meta.bitsets
            upper : tuple(hnx.concept_lattices.Concept)
            lower : tuple(hnx.concept_lattices.Concept)
        """
        infimum = ()
        extent, intent = context_objects.frommembers(infimum).doubleprime()
        concept = (extent, intent, [], [])
        mapping = {extent: concept}
        heap = [(extent.shortlex(), concept)]

        push = functools.partial(heapq.heappush, heap)
        pop = functools.partial(heapq.heappop, heap)

        while heap:
            _, concept = pop()

            extent, _, upper, _ = concept

            for n_extent, n_intent in LatticeAlgorithms.neighbors(
                extent, context_objects
            ):
                upper.append(n_extent)

                if n_extent in mapping:
                    mapping[n_extent][3].append(extent)
                else:
                    mapping[n_extent] = neighbor = (n_extent, n_intent, [], [extent])
                    push((n_extent.shortlex(), neighbor))

            yield concept

    def neighbors(objects, context_objects):
        """
        A helper function for LatticeAlgorithms.lattice_element. Yields the upper neighbors
        of a bitset with respect to a context.

        Parameters
        ----------
        objects : bitset.meta.bitsets
            a bitset of objects from the context_objects
        context_objects : hnx.concept_lattices.BoolVectorPair
            BoolVectorPair instance containing the collection of objects in context
            and the derivation operators of the underlying context.

        Yields
        ------
        tuple(extent, intent)
            extent : bitset.meta.bitsets
            intent : bitset.meta.bitsets
        """
        doubleprime = context_objects.doubleprime

        minimal = ~objects

        for add in context_objects.atomic(minimal):
            objects_and_add = objects | add

            extent, intent = doubleprime(objects_and_add)

            if extent & ~objects_and_add & minimal:
                minimal &= ~add
            else:
                yield extent, intent


class HypergraphLattice(Lattice):
    def __init__(self, hypergraph):
        """
        Constructs a class containing the concept lattice of a hypergraph.

        Parameters
        ----------
        hypergraph : hnx.Hypergraph
            The index and column labels of this hypergraph must be strings.

        Returns
        -------
        None
        """
        df = hypergraph.incidence_dataframe()

        if not all(isinstance(item, str) for item in df.index):
            raise ValueError("Index labels must be strings.")

        if not all(isinstance(item, str) for item in df.columns):
            raise ValueError("Column labels must be strings.")

        bools = list(df.fillna(False).astype(bool).itertuples(index=False, name=None))
        ctx = Context(df.index, df.columns, bools)
        super().__init__(ctx)
        self.digraph = self.to_networkx()

    def concept_index(self, concept):
        """
        Returns the index of a given concept.

        Parameters
        ----------
        hypergraph : hnx.concept_lattice.Concept

        Returns
        -------
        int
        """
        return self._concepts.index(concept)

    def distance(self, x, y, metric="shortest_path"):
        """
        Returns distance between the ith and jth concepts in the lattice.

        Parameters
        ----------
        x : int or hnx.concept_lattices.Concept
        y : int or hnx.concept_lattices.Concept

        Returns
        -------
        float
        """

        if isinstance(x, Concept):
            i = self.concept_index(x)
        else:
            i = x

        if isinstance(y, Concept):
            j = self.concept_index(y)
        else:
            j = y

        if metric == "upper_valuation":
            node_set = set(self.digraph.nodes())
            i_upset = nx.descendants(self.digraph, i).union({i})
            j_upset = nx.descendants(self.digraph, j).union({j})
            v_i = node_set.difference(i_upset)
            v_j = node_set.difference(j_upset)
            v_ij = v_i.union(v_j)
            return float(2 * len(v_ij) - len(v_i) - len(v_j))

        if metric == "lower_valuation":
            dual = nx.DiGraph.reverse(self.digraph)
            node_set = set(self.digraph.nodes())
            i_upset = nx.descendants(dual, i).union({i})
            j_upset = nx.descendants(dual, j).union({j})
            v_i = node_set.difference(i_upset)
            v_j = node_set.difference(j_upset)
            v_ij = v_i.union(v_j)
            return float(2 * len(v_ij) - len(v_i) - len(v_j))

        if metric == "shortest_path":
            G = self.digraph.to_undirected()
            return float(nx.shortest_path_length(G, source=i, target=j))

        if isinstance(metric, dict):
            if not all(metric[key] > 0 for key in metric.keys()):
                raise ValueError("Weights must be positive.")

            v_i = np.sum([metric[x] for x in self[i]._extent.members()])
            v_j = np.sum([metric[x] for x in self[j]._extent.members()])
            v_ij = np.sum(
                [metric[x] for x in self.meet([self[i], self[j]])._extent.members()]
            )
            return v_i + v_j - 2 * v_ij

    def all_distances(self, metric="shortest_path"):
        """
        Returns a dictionary of distances between each of concepts of the lattice.

        Paramters
        ---------
        metric : str or dict
            Default is shortest path metric. Other options are 'upper_valuation', lower_valuation' or a dictionary of vertex indices and
            positive float weights.

        Returns
        -------
        dict
        """

        if metric == "upper_valuation":
            nodes = self.digraph.nodes()
            num_nodes = len(nodes)
            node_indices = range(num_nodes)
            d_upper = dict()

            for i in range(num_nodes):
                d_upper[i] = dict()
                for j in range(num_nodes):
                    d_upper[i][j] = self.distance(
                        node_indices[i], node_indices[j], metric="upper_valuation"
                    )

            return d_upper

        if metric == "lower_valuation":
            nodes = self.digraph.nodes()
            num_nodes = len(nodes)
            node_indices = range(num_nodes)
            d_lower = dict()

            for i in range(num_nodes):
                d_lower[i] = dict()
                for j in range(num_nodes):
                    d_lower[i][j] = self.distance(
                        node_indices[i], node_indices[j], metric="lower_valuation"
                    )

            return d_lower

        if metric == "shortest_path":
            G = self.digraph.to_undirected()
            return dict(nx.all_pairs_shortest_path_length(G))

        if isinstance(metric, dict):
            if not all(metric[key] >= 0 for key in metric.keys()):
                raise ValueError("Weights must be non-negative.")

            nodes = self.digraph.nodes()
            num_nodes = len(nodes)
            node_indices = range(num_nodes)
            d = dict()

            for i in range(num_nodes):
                d[i] = dict()
                for j in range(num_nodes):
                    d[i][j] = self.distance(node_indices[i], node_indices[j], metric)

            return d

    def draw_lattice(
        self,
        concept_labels=True,
        shortform=True,
        node_alpha=0.5,
        edge_alpha=0.5,
        edge_width=0.8,
        arrow_size=5,
        label_offset=0.1,
        ax=None,
        return_pos=False,
        horizontal=True,
        font_size=8,
    ):
        """
        Draw a hypergraph's concept lattice as a Matplotlib figure using
        networkx draw functionality


        Parameters
        ----------
        concept_labels : bool
            Set to False to draw lattice digraph without any vertex labellings.
        shortform : bool
            If drawing concept labels, set to True to draw full extent/intent pairs.
        ax: Axis
            matplotlib axis on which the plot is rendered
        horizontal: bool
            Set to False to draw lattice with ordering from order increasing vertically

        Returns
        -------
        None or pos
        """
        ax = ax or plt.gca()
        G = self.digraph
        # topological layout
        for layer, nodes in enumerate(nx.topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer

        pos = nx.multipartite_layout(G, subset_key="layer")  # node positions
        if horizontal == False:
            rotation_matrix = np.array([[0, -1], [1, 0]])
            new_pos = {
                node: np.dot(rotation_matrix, np.array(coord))
                for node, coord in pos.items()
            }
            pos = new_pos
        if not concept_labels:
            nx.draw_networkx(G, pos=pos, alpha=node_alpha, arrowsize=arrow_size)
        else:
            concepts = nx.get_node_attributes(G, "concept")
            label_pos = dict(
                [(key, np.array([x[0], x[1] - label_offset])) for key, x in pos.items()]
            )  # make shift depend on node radius
            if shortform:
                tups = [
                    (key, value.shortform_string()) for key, value in concepts.items()
                ]
                nx.draw_networkx_nodes(G, pos=pos, alpha=node_alpha)
                nx.draw_networkx_edges(
                    G, pos=pos, alpha=edge_alpha, width=edge_width, arrowsize=arrow_size
                )
                nx.draw_networkx_labels(
                    G, pos=label_pos, labels=dict(tups), font_size=font_size
                )
            else:
                tups = [(key, str(value)) for key, value in concepts.items()]
                nx.draw_networkx_nodes(G, pos=pos, alpha=node_alpha)
                nx.draw_networkx_edges(
                    G, pos=pos, alpha=edge_alpha, width=edge_width, arrowsize=arrow_size
                )
                nx.draw_networkx_labels(
                    G, pos=label_pos, labels=dict(tups), font_size=font_size
                )

        ax.axis("off")
        if return_pos:
            return pos
