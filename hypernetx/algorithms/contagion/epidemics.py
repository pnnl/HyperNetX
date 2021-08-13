import random
import heapq
import numpy as np
from collections import defaultdict
from collections import Counter

# Canned Contagion Functions
def collective_contagion(node, status, edge):
    """
    The collective contagion mechanism described in
    "The effect of heterogeneity on hypergraph contagion models" by Landry and Restrepo
    https://doi.org/10.1063/5.0020034

    Parameters
    ----------
    node : hashable
        the node uid to infect (If it doesn't have status "S", it will automatically return False)
    status : dictionary
        The nodes are keys and the values are statuses (The infected state denoted with "I")
    edge : iterable
        Iterable of node ids (node must be in the edge or it will automatically return False)

    Returns
    -------
    bool
        False if there is no potential to infect and True if there is.

    Notes
    -----

    Example::

        >>> status = {0:"S", 1:"I", 2:"I", 3:"S", 4:"R"}
        >>> collective_contagion(0, status, (0, 1, 2))
            True
        >>> collective_contagion(1, status, (0, 1, 2))
            False
        >>> collective_contagion(3, status, (0, 1, 2))
            False
    """
    if status[node] != "S" or node not in edge:
        return False

    neighbors = set(edge).difference({node})
    for i in neighbors:
        if status[i] != "I":
            return False
    return True


def individual_contagion(node, status, edge):
    """
    The individual contagion mechanism described in
    "The effect of heterogeneity on hypergraph contagion models" by Landry and Restrepo
    https://doi.org/10.1063/5.0020034

    Parameters
    ----------
    node : hashable
        The node uid to infect (If it doesn't have status "S", it will automatically return False)
    status : dictionary
        The nodes are keys and the values are statuses (The infected state denoted with "I")
    edge : iterable
        Iterable of node ids (node must be in the edge or it will automatically return False)

    Returns
    -------
    bool
        False if there is no potential to infect and True if there is.

    Notes
    -----

    Example::

        >>> status = {0:"S", 1:"I", 2:"I", 3:"S", 4:"R"}
        >>> individual_contagion(0, status, (0, 1, 3))
            True
        >>> individual_contagion(1, status, (0, 1, 2))
            False
        >>> collective_contagion(3, status, (0, 3, 4))
            False
    """
    if status[node] != "S" or node not in edge:
        return False

    neighbors = set(edge).difference({node})
    for i in neighbors:
        if status[i] == "I":
            return True
    return False


def threshold(node, status, edge, tau=0.1):
    """
    The threshold contagion mechanism

    Parameters
    ----------
    node : hashable
        The node uid to infect (If it doesn't have status "S", it will automatically return False)
    status : dictionary
        The nodes are keys and the values are statuses (The infected state denoted with "I")
    edge : iterable
        Iterable of node ids (node must be in the edge or it will automatically return False)
    tau : float between 0 and 1, default: 0.1
        The fraction of nodes in an edge that must be infected for the edge to be able to transmit to the node

    Returns
    -------
    bool
        False if there is no potential to infect and True if there is.

    Notes
    -----

    Example::

        >>> status = {0:"S", 1:"I", 2:"I", 3:"S", 4:"R"}
        >>> threshold(0, status, (0, 2, 3, 4), tau=0.2)
            True
        >>> threshold(0, status, (0, 2, 3, 4), tau=0.5)
            False
        >>> threshold(3, status, (1, 2, 3), tau=1)
            False
    """
    if status[node] != "S" or node not in edge:
        return False

    neighbors = set(edge).difference({node})
    if len(neighbors) > 0:
        fraction_infected = sum([status[i] == "I" for i in neighbors]) / len(neighbors)
    # The isolated node case
    else:
        fraction_infected = 0
    return fraction_infected >= tau


def majority_vote(node, status, edge):
    """
    The majority vote contagion mechanism. If a majority of neighbors are contagious,
    it is possible for an individual to change their opinion. If opinions are divided equally,
    choose randomly.


    Parameters
    ----------
    node : hashable
        The node uid to infect (If it doesn't have status "S", it will automatically return False)
    status : dictionary
        The nodes are keys and the values are statuses (The infected state denoted with "I")
    edge : iterable
        Iterable of node ids (node must be in the edge or it will automatically return False
    Returns
    -------
    bool
        False if there is no potential to infect and True if there is.

    Notes
    -----

    Example::

        >>> status = {0:"S", 1:"I", 2:"I", 3:"S", 4:"R"}
        >>> majority_vote(0, status, (0, 1, 2))
            True
        >>> majority_vote(0, status, (0, 1, 2, 3))
            True
        >>> majority_vote(1, status, (0, 1, 2))
            False
        >>> majority_vote(3, status, (0, 1, 2))
            False
    """

    if status[node] != "S" or node not in edge:
        return False

    neighbors = set(edge).difference({node})
    if len(neighbors) > 0:
        fraction_infected = sum([status[i] == "I" for i in neighbors]) / len(neighbors)
    else:
        fraction_infected = 0

    if fraction_infected < 0.5:
        return False
    elif fraction_infected > 0.5:
        return True
    else:
        return random.choice([False, True])


# Auxiliary functions

# The ListDict class is copied from Joel Miller's Github repository Mathematics-of-Epidemics-on-Networks
class _ListDict_(object):
    r"""
    The Gillespie algorithm will involve a step that samples a random element
    from a set based on its weight.  This is awkward in Python.

    So I'm introducing a new class based on a stack overflow answer by
    Amber (http://stackoverflow.com/users/148870/amber)
    for a question by
    tba (http://stackoverflow.com/users/46521/tba)
    found at
    http://stackoverflow.com/a/15993515/2966723

    This will allow me to select a random element uniformly, and then use
    rejection sampling to make sure it's been selected with the appropriate
    weight.
    """

    def __init__(self, weighted=False):
        self.item_to_position = {}
        self.items = []

        self.weighted = weighted
        if self.weighted:
            self.weight = defaultdict(int)  # presume all weights positive
            self.max_weight = 0
            self._total_weight = 0
            self.max_weight_count = 0

    def __len__(self):
        return len(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def _update_max_weight(self):
        C = Counter(
            self.weight.values()
        )  # may be a faster way to do this, we only need to count the max.
        self.max_weight = max(C.keys())
        self.max_weight_count = C[self.max_weight]

    def insert(self, item, weight=None):
        r"""
        If not present, then inserts the thing (with weight if appropriate)
        if already there, replaces the weight unless weight is 0

        If weight is 0, then it removes the item and doesn't replace.

        WARNING:
            replaces weight if already present, does not increment weight.


        """
        if self.__contains__(item):
            self.remove(item)
        if weight != 0:
            self.update(item, weight_increment=weight)

    def update(self, item, weight_increment=None):
        r"""
        If not present, then inserts the thing (with weight if appropriate)
        if already there, increments weight

        WARNING:
            increments weight if already present, cannot overwrite weight.
        """
        if (
            weight_increment is not None
        ):  # will break if passing a weight to unweighted case
            if weight_increment > 0 or self.weight[item] != self.max_weight:
                self.weight[item] = self.weight[item] + weight_increment
                self._total_weight += weight_increment
                if self.weight[item] > self.max_weight:
                    self.max_weight_count = 1
                    self.max_weight = self.weight[item]
                elif self.weight[item] == self.max_weight:
                    self.max_weight_count += 1
            else:  # it's a negative increment and was at max
                self.max_weight_count -= 1
                self.weight[item] = self.weight[item] + weight_increment
                self._total_weight += weight_increment
                self.max_weight_count -= 1
                if self.max_weight_count == 0:
                    self._update_max_weight
        elif self.weighted:
            raise Exception("if weighted, must assign weight_increment")

        if item in self:  # we've already got it, do nothing else
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items) - 1

    def remove(self, choice):
        position = self.item_to_position.pop(choice)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

        if self.weighted:
            weight = self.weight.pop(choice)
            self._total_weight -= weight
            if weight == self.max_weight:
                # if we find ourselves in this case often
                # it may be better just to let max_weight be the
                # largest weight *ever* encountered, even if all remaining weights are less
                #
                self.max_weight_count -= 1
                if self.max_weight_count == 0 and len(self) > 0:
                    self._update_max_weight()

    def choose_random(self):
        # r'''chooses a random node.  If there is a weight, it will use rejection
        # sampling to choose a random node until it succeeds'''
        if self.weighted:
            while True:
                choice = random.choice(self.items)
                if random.random() < self.weight[choice] / self.max_weight:
                    break
            # r = random.random()*self.total_weight
            # for item in self.items:
            #     r-= self.weight[item]
            #     if r<0:
            #         break
            return choice

        else:
            return random.choice(self.items)

    def random_removal(self):
        r"""uses other class methods to choose and then remove a random node"""
        choice = self.choose_random()
        self.remove(choice)
        return choice

    def total_weight(self):
        if self.weighted:
            return self._total_weight
        else:
            return len(self)

    def update_total_weight(self):
        self._total_weight = sum(self.weight[item] for item in self.items)


# Contagion Functions
def discrete_SIR(
    H,
    tau,
    gamma,
    transmission_function=threshold,
    initial_infecteds=None,
    initial_recovereds=None,
    rho=None,
    tmin=0,
    tmax=float("Inf"),
    dt=1.0,
    return_full_data=False,
    **args
):
    """
    A discrete-time SIR model for hypergraphs similar to the construction described in
    "The effect of heterogeneity on hypergraph contagion models" by Landry and Restrepo
    https://doi.org/10.1063/5.0020034 and
    "Simplicial models of social contagion" by Iacopini et al.
    https://doi.org/10.1038/s41467-019-10431-6

    Parameters
    ----------
    H : HyperNetX Hypergraph object
    tau : dictionary
        Edge sizes as keys (must account for all edge sizes present) and rates of infection for each size (float)
    gamma : float
        The healing rate
    transmission_function : lambda function, default: threshold
        A lambda function that has required arguments (node, status, edge) and optional arguments
    initial_infecteds : list or numpy array, default: None
        Iterable of initially infected node uids
    initial_recovereds : list or numpy array, default: None
        An iterable of initially recovered node uids
    rho : float from 0 to 1, default: None
        The fraction of initially infected individuals. Both rho and initially infected cannot be specified.
    tmin : float, default: 0
        Time at the start of the simulation
    tmax : float, default: float('Inf')
        Time at which the simulation should be terminated if it hasn't already.
    dt : float > 0, default: 1.0
        Step forward in time that the simulation takes at each step.
    return_full_data : bool, default: False
        This returns all the infection and recovery events at each time if True.
    **args : Optional arguments to transmission function
        This allows user-defined transmission functions with extra parameters.

    Returns
    -------
    if return_full_data
        dictionary
            Time as the keys and events that happen as the values.
    else
        t, S, I, R : numpy arrays
            time (t), number of susceptible (S), infected (I), and recovered (R) at each time.

    Notes
    -----

    Example::

        >>> import hypernetx.algorithms.contagion as contagion
        >>> import random
        >>> import hypernetx as hnx
        >>> n = 1000
        >>> m = 10000
        >>> hyperedgeList = [random.sample(range(n), k=random.choice([2,3])) for i in range(m)]
        >>> H = hnx.Hypergraph(hyperedgeList)
        >>> tau = {2:0.1, 3:0.1}
        >>> gamma = 0.1
        >>> tmax = 100
        >>> dt = 0.1
        >>> t, S, I, R = contagion.discrete_SIR(H, tau, gamma, rho=0.1, tmin=0, tmax=tmax, dt=dt)
    """

    if rho is not None and initial_infecteds is not None:
        raise Exception("Cannot define both initial_infecteds and rho")

    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(H.number_of_nodes() * rho))
        initial_infecteds = random.sample(list(H.nodes), initial_number)
    else:
        # check to make sure that the initially infected nodes are in the hypergraph
        initial_infecteds = list(set(H.nodes).intersection(set(initial_infecteds)))

    if initial_recovereds is None:
        initial_recovereds = []
    else:
        # check to make sure that the initially recovered nodes are in the hypergraph
        initial_recovereds = list(set(H.nodes).intersection(set(initial_recovereds)))

    status = defaultdict(lambda: "S")

    if return_full_data:
        transition_events = dict()
        transition_events[tmin] = list()

    for node in initial_infecteds:
        status[node] = "I"
        if return_full_data:
            transition_events[tmin].append(("I", node, None))

    for node in initial_recovereds:
        status[node] = "R"
        if return_full_data:
            transition_events[tmin].append(("R", node))

    I = [len(initial_infecteds)]
    R = [len(initial_recovereds)]
    S = [H.number_of_nodes() - I[-1] - R[-1]]

    t = tmin
    times = [t]
    newStatus = status.copy()

    if H.isstatic:
        edge_neighbors = lambda node: H.edges.memberships[node]
    else:
        edge_neighbors = lambda node: H.nodes[node].memberships

    while t < tmax and I[-1] != 0:
        # Initialize the next step with the same numbers of S, I, and R as the last step before computing the changes
        S.append(S[-1])
        I.append(I[-1])
        R.append(R[-1])

        if return_full_data:
            transition_events[t + dt] = list()

        for node in H.nodes:
            if status[node] == "I":
                # recover the node. If it is not healed, it stays infected.
                if random.random() <= gamma * dt:
                    newStatus[node] = "R"
                    I[-1] += -1
                    R[-1] += 1
                    if return_full_data:
                        transition_events[t + dt].append(("R", node))
            elif status[node] == "S":
                for edge_id in edge_neighbors(node):
                    members = H.edges[edge_id]
                    if (
                        random.random()
                        <= tau[len(members)]
                        * transmission_function(node, status, members, **args)
                        * dt
                    ):
                        newStatus[node] = "I"
                        S[-1] += -1
                        I[-1] += 1
                        if return_full_data:
                            transition_events[t + dt].append(("I", node, edge_id))
                        break
                # This executes after the loop has executed normally without hitting the break statement which indicates infection
                else:
                    newStatus[node] = "S"
        status = newStatus.copy()
        t += dt
        times.append(t)
    if return_full_data:
        return transition_events
    else:
        return np.array(times), np.array(S), np.array(I), np.array(R)


def discrete_SIS(
    H,
    tau,
    gamma,
    transmission_function=threshold,
    initial_infecteds=None,
    rho=None,
    tmin=0,
    tmax=100,
    dt=1.0,
    return_full_data=False,
    **args
):
    """
    A discrete-time SIS model for hypergraphs as implemented in
    "The effect of heterogeneity on hypergraph contagion models" by Landry and Restrepo
    https://doi.org/10.1063/5.0020034 and
    "Simplicial models of social contagion" by Iacopini et al.
    https://doi.org/10.1038/s41467-019-10431-6

    Parameters
    ----------
    H : HyperNetX Hypergraph object
    tau : dictionary
        Edge sizes as keys (must account for all edge sizes present) and rates of infection for each size (float)
    gamma : float
        The healing rate
    transmission_function : lambda function, default: threshold
        A lambda function that has required arguments (node, status, edge) and optional arguments
    initial_infecteds : list or numpy array, default: None
        Iterable of initially infected node uids
    rho : float from 0 to 1, default: None
        The fraction of initially infected individuals. Both rho and initially infected cannot be specified.
    tmin : float, default: 0
        Time at the start of the simulation
    tmax : float, default: 100
        Time at which the simulation should be terminated if it hasn't already.
    dt : float > 0, default: 1.0
        Step forward in time that the simulation takes at each step.
    return_full_data : bool, default: False
        This returns all the infection and recovery events at each time if True.
    **args : Optional arguments to transmission function
        This allows user-defined transmission functions with extra parameters.

    Returns
    -------
    if return_full_data
        dictionary
            Time as the keys and events that happen as the values.
    else
        t, S, I : numpy arrays
            time (t), number of susceptible (S), and infected (I) at each time.

    Notes
    -----

    Example::

        >>> import hypernetx.algorithms.contagion as contagion
        >>> import random
        >>> import hypernetx as hnx
        >>> n = 1000
        >>> m = 10000
        >>> hyperedgeList = [random.sample(range(n), k=random.choice([2,3])) for i in range(m)]
        >>> H = hnx.Hypergraph(hyperedgeList)
        >>> tau = {2:0.1, 3:0.1}
        >>> gamma = 0.1
        >>> tmax = 100
        >>> dt = 0.1
        >>> t, S, I = contagion.discrete_SIS(H, tau, gamma, rho=0.1, tmin=0, tmax=tmax, dt=dt)
    """

    if rho is not None and initial_infecteds is not None:
        raise Exception("Cannot define both initial_infecteds and rho")

    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(H.number_of_nodes() * rho))
        initial_infecteds = random.sample(list(H.nodes), initial_number)
    else:
        # check to make sure that the initially infected nodes are in the hypergraph
        initial_infecteds = list(set(H.nodes).intersection(set(initial_infecteds)))

    status = defaultdict(lambda: "S")

    if return_full_data:
        transition_events = dict()
        transition_events[tmin] = list()

    for node in initial_infecteds:
        status[node] = "I"
        if return_full_data:
            transition_events[tmin].append(("I", node, None))

    I = [len(initial_infecteds)]
    S = [H.number_of_nodes() - I[-1]]

    t = tmin
    times = [t]
    newStatus = status.copy()

    if H.isstatic:
        edge_neighbors = lambda node: H.edges.memberships[node]
    else:
        edge_neighbors = lambda node: H.nodes[node].memberships

    while t < tmax and I[-1] != 0:
        # Initialize the next step with the same numbers of S, I, and R as the last step before computing the changes
        S.append(S[-1])
        I.append(I[-1])
        if return_full_data:
            transition_events[t + dt] = list()

        for node in H.nodes:
            if status[node] == "I":
                # recover the node. If it is not healed, it stays infected.
                if random.random() <= gamma * dt:
                    newStatus[node] = "S"
                    I[-1] += -1
                    S[-1] += 1
                    if return_full_data:
                        transition_events[t + dt].append(("S", node))
            elif status[node] == "S":
                for edge_id in edge_neighbors(node):
                    members = H.edges[edge_id]

                    if (
                        random.random()
                        <= tau[len(members)]
                        * transmission_function(node, status, members, **args)
                        * dt
                    ):
                        newStatus[node] = "I"
                        S[-1] += -1
                        I[-1] += 1
                        if return_full_data:
                            transition_events[t + dt].append(("I", node, edge_id))
                        break
                # This executes after the loop has executed normally without hitting the break statement which indicates infection, though I'm not sure we even need it
                else:
                    newStatus[node] = "S"
        status = newStatus.copy()
        t += dt
        times.append(t)
    if return_full_data:
        return transition_events
    else:
        return np.array(times), np.array(S), np.array(I)


def Gillespie_SIR(
    H,
    tau,
    gamma,
    transmission_function=threshold,
    initial_infecteds=None,
    initial_recovereds=None,
    rho=None,
    tmin=0,
    tmax=float("Inf"),
    **args
):
    """
    A continuous-time SIR model for hypergraphs similar to the model in
    "The effect of heterogeneity on hypergraph contagion models" by Landry and Restrepo
    https://doi.org/10.1063/5.0020034 and
    implemented for networks in the EoN package by Joel C. Miller
    https://epidemicsonnetworks.readthedocs.io/en/latest/

    Parameters
    ----------
    H : HyperNetX Hypergraph object
    tau : dictionary
        Edge sizes as keys (must account for all edge sizes present) and rates of infection for each size (float)
    gamma : float
        The healing rate
    transmission_function : lambda function, default: threshold
        A lambda function that has required arguments (node, status, edge) and optional arguments
    initial_infecteds : list or numpy array, default: None
        Iterable of initially infected node uids
    initial_recovereds : list or numpy array, default: None
        An iterable of initially recovered node uids
    rho : float from 0 to 1, default: None
        The fraction of initially infected individuals. Both rho and initially infected cannot be specified.
    tmin : float, default: 0
        Time at the start of the simulation
    tmax : float, default: float('Inf')
        Time at which the simulation should be terminated if it hasn't already.
    return_full_data : bool, default: False
        This returns all the infection and recovery events at each time if True.
    **args : Optional arguments to transmission function
        This allows user-defined transmission functions with extra parameters.

    Returns
    -------
    t, S, I, R : numpy arrays
        time (t), number of susceptible (S), infected (I), and recovered (R) at each time.

    Notes
    -----

    Example::

        >>> import hypernetx.algorithms.contagion as contagion
        >>> import random
        >>> import hypernetx as hnx
        >>> n = 1000
        >>> m = 10000
        >>> hyperedgeList = [random.sample(range(n), k=random.choice([2,3])) for i in range(m)]
        >>> H = hnx.Hypergraph(hyperedgeList)
        >>> tau = {2:0.1, 3:0.1}
        >>> gamma = 0.1
        >>> tmax = 100
        >>> t, S, I, R = contagion.Gillespie_SIR(H, tau, gamma, rho=0.1, tmin=0, tmax=tmax)
    """
    # Initial infecteds and recovereds should be lists or None. Add a check here.

    if rho is not None and initial_infecteds is not None:
        raise Exception("Cannot define both initial_infecteds and rho")

    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(H.number_of_nodes() * rho))
        initial_infecteds = random.sample(list(H.nodes), initial_number)
    else:
        # check to make sure that the initially infected nodes are in the hypergraph
        initial_infecteds = list(set(H.nodes).intersection(set(initial_infecteds)))

    if initial_recovereds is None:
        initial_recovereds = []
    else:
        # check to make sure that the initially recovered nodes are in the hypergraph
        initial_recovereds = list(set(H.nodes).intersection(set(initial_recovereds)))

    status = defaultdict(lambda: "S")

    size_dist = np.unique(H.edge_size_dist())

    for node in initial_infecteds:
        status[node] = "I"

    for node in initial_recovereds:
        status[node] = "R"

    I = [len(initial_infecteds)]
    R = [len(initial_recovereds)]
    S = [H.number_of_nodes() - I[-1] - R[-1]]

    if H.isstatic:
        edge_neighbors = lambda node: H.edges.memberships[node]
    else:
        edge_neighbors = lambda node: H.nodes[node].memberships

    t = tmin
    times = [t]

    infecteds = _ListDict_()

    infectious_edges = dict()
    for size in size_dist:
        infectious_edges[size] = _ListDict_()

    for node in initial_infecteds:
        infecteds.update(node)
        for edge_id in edge_neighbors(node):
            members = H.edges[edge_id]
            for node in members:
                is_infectious = transmission_function(node, status, members, **args)
                if is_infectious:
                    infectious_edges[len(members)].update((edge_id, node))

    total_rates = dict()
    total_rates[1] = gamma * infecteds.total_weight()

    for size in size_dist:
        total_rates[size] = tau[size] * infectious_edges[size].total_weight()

    total_rate = sum(total_rates.values())

    dt = random.expovariate(total_rate)
    t += dt

    while t < tmax and I[-1] != 0:
        # choose type of event that happens
        while True:
            choice = random.choice(list(total_rates.keys()))
            if random.random() <= total_rates[choice] / total_rate:
                break

        if choice == 1:  # recover
            recovering_node = infecteds.random_removal()
            status[recovering_node] = "R"

            # remove edges that are no longer infectious because of this node recovering
            for edge_id in edge_neighbors(recovering_node):
                members = H.edges[edge_id]
                for node in members:
                    is_infectious = transmission_function(node, status, members, **args)
                    if is_infectious:
                        try:
                            infectious_edges[len(members)].remove((edge_id, node))
                        except:
                            pass
            times.append(t)
            S.append(S[-1])
            I.append(I[-1] - 1)
            R.append(R[-1] + 1)

        else:
            _, recipient = infectious_edges[choice].choose_random()
            status[recipient] = "I"

            infecteds.update(recipient)

            # remove the infectious links, because they can't infect an infected node.
            for edge_id in edge_neighbors(recipient):
                members = H.edges[edge_id]
                try:
                    infectious_edges[len(members)].remove((edge_id, recipient))
                except:
                    pass

            # add edges that are infectious because of this node being infected
            for edge_id in edge_neighbors(recipient):
                members = H.edges[edge_id]
                for node in members:
                    is_infectious = transmission_function(node, status, members, **args)
                    if is_infectious:
                        try:
                            infectious_edges[len(members)].update((edge_id, node))
                        except:
                            pass
            times.append(t)
            S.append(S[-1] - 1)
            I.append(I[-1] + 1)
            R.append(R[-1])

        total_rates[1] = gamma * infecteds.total_weight()
        for size in size_dist:
            total_rates[size] = tau[size] * infectious_edges[size].total_weight()

        total_rate = sum(total_rates.values())

        if total_rate > 0:
            dt = random.expovariate(total_rate)
        else:
            dt = float("Inf")
        t += dt
    return np.array(times), np.array(S), np.array(I), np.array(R)


def Gillespie_SIS(
    H,
    tau,
    gamma,
    transmission_function=threshold,
    initial_infecteds=None,
    rho=None,
    tmin=0,
    tmax=float("Inf"),
    return_full_data=False,
    sim_kwargs=None,
    **args
):
    """
    A continuous-time SIS model for hypergraphs similar to the model in
    "The effect of heterogeneity on hypergraph contagion models" by Landry and Restrepo
    https://doi.org/10.1063/5.0020034 and
    implemented for networks in the EoN package by Joel C. Miller
    https://epidemicsonnetworks.readthedocs.io/en/latest/

    Parameters
    ----------
    H : HyperNetX Hypergraph object
    tau : dictionary
        Edge sizes as keys (must account for all edge sizes present) and rates of infection for each size (float)
    gamma : float
        The healing rate
    transmission_function : lambda function, default: threshold
        A lambda function that has required arguments (node, status, edge) and optional arguments
    initial_infecteds : list or numpy array, default: None
        Iterable of initially infected node uids
    rho : float from 0 to 1, default: None
        The fraction of initially infected individuals. Both rho and initially infected cannot be specified.
    tmin : float, default: 0
        Time at the start of the simulation
    tmax : float, default: 100
        Time at which the simulation should be terminated if it hasn't already.
    return_full_data : bool, default: False
        This returns all the infection and recovery events at each time if True.
    **args : Optional arguments to transmission function
        This allows user-defined transmission functions with extra parameters.

    Returns
    -------
    t, S, I : numpy arrays
        time (t), number of susceptible (S), and infected (I) at each time.

    Notes
    -----

    Example::

        >>> import hypernetx.algorithms.contagion as contagion
        >>> import random
        >>> import hypernetx as hnx
        >>> n = 1000
        >>> m = 10000
        >>> hyperedgeList = [random.sample(range(n), k=random.choice([2,3])) for i in range(m)]
        >>> H = hnx.Hypergraph(hyperedgeList)
        >>> tau = {2:0.1, 3:0.1}
        >>> gamma = 0.1
        >>> tmax = 100
        >>> t, S, I = contagion.Gillespie_SIS(H, tau, gamma, rho=0.1, tmin=0, tmax=tmax)
    """
    # Initial infecteds and recovereds should be lists or None. Add a check here.

    if rho is not None and initial_infecteds is not None:
        raise Exception("Cannot define both initial_infecteds and rho")

    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(H.number_of_nodes() * rho))
        initial_infecteds = random.sample(list(H.nodes), initial_number)
    else:
        # check to make sure that the initially infected nodes are in the hypergraph
        initial_infecteds = list(set(H.nodes).intersection(set(initial_infecteds)))

    status = defaultdict(lambda: "S")

    size_dist = np.unique(H.edge_size_dist())

    for node in initial_infecteds:
        status[node] = "I"

    I = [len(initial_infecteds)]
    S = [H.number_of_nodes() - I[-1]]

    if H.isstatic:
        edge_neighbors = lambda node: H.edges.memberships[node]
    else:
        edge_neighbors = lambda node: H.nodes[node].memberships

    t = tmin
    times = [t]

    infecteds = _ListDict_()

    infectious_edges = dict()
    for size in size_dist:
        infectious_edges[size] = _ListDict_()

    for node in initial_infecteds:
        infecteds.update(node)
        for edge_id in edge_neighbors(node):
            members = H.edges[edge_id]
            for node in members:
                is_infectious = transmission_function(node, status, members, **args)
                if is_infectious:
                    infectious_edges[len(members)].update((edge_id, node))

    total_rates = dict()
    total_rates[1] = gamma * infecteds.total_weight()
    for size in size_dist:
        total_rates[size] = tau[size] * infectious_edges[size].total_weight()

    total_rate = sum(total_rates.values())

    dt = random.expovariate(total_rate)
    t += dt

    while t < tmax and I[-1] != 0:
        # choose type of event that happens
        # this can be improved
        while True:
            choice = random.choice(list(total_rates.keys()))
            if random.random() <= total_rates[choice] / total_rate:
                break

        if choice == 1:  # recover
            recovering_node = infecteds.random_removal()
            status[recovering_node] = "S"

            # remove edges that are no longer infectious because of this node recovering
            for edge_id in edge_neighbors(recovering_node):
                members = H.edges[edge_id]
                for node in members:
                    is_infectious = transmission_function(node, status, members, **args)
                    if is_infectious:
                        try:
                            infectious_edges[len(members)].remove((edge_id, node))
                        except:
                            pass
            times.append(t)
            S.append(S[-1] + 1)
            I.append(I[-1] - 1)

        else:
            _, recipient = infectious_edges[choice].choose_random()
            status[recipient] = "I"

            infecteds.update(recipient)

            # remove the infectious links, because they can't infect an infected node.
            for edge_id in edge_neighbors(recipient):
                members = H.edges[edge_id]
                try:
                    infectious_edges[len(members)].remove((edge_id, recipient))
                except:
                    pass

            # add edges that are infectious because of this node being infected
            for edge_id in edge_neighbors(recipient):
                members = H.edges[edge_id]
                for node in members:
                    is_infectious = transmission_function(node, status, members, **args)
                    if is_infectious:
                        try:
                            infectious_edges[len(members)].update((edge_id, node))
                        except:
                            pass
            times.append(t)
            S.append(S[-1] - 1)
            I.append(I[-1] + 1)

        total_rates[1] = gamma * infecteds.total_weight()
        for size in size_dist:
            total_rates[size] = tau[size] * infectious_edges[size].total_weight()

        total_rate = sum(total_rates.values())

        if total_rate > 0:
            dt = random.expovariate(total_rate)
        else:
            dt = float("Inf")
        t += dt

    return np.array(times), np.array(S), np.array(I)
