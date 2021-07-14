import random
import heapq
import numpy as np
from collections import defaultdict
from collections import Counter

# Canned Contagion Functions
def collective_contagion(status, neighbors):
    for i in neighbors:
        if status[i] != 'I':
            return 0
    return 1

def individual_contagion(status, neighbors):
    for i in neighbors:
        if status[i] == 'I':
            return 1
    return 0

def threshold(status, neighbors, threshold=0.1):
    mean_contagion = sum([status[i] == 'I' for i in neighbors])/len(neighbors)
    return mean_contagion >= threshold

def majority_vote(status, neighbors):
    mean_contagion = sum([status[i] == 'I' for i in neighbors])/len(neighbors)
    if mean_contagion < 0.5:
        return 0
    elif mean_contagion > 0.5:
        return 1
    else:
        return random.choice([0, 1])


# Auxiliary functions
def _truncated_exponential_(rate, T):
    r'''returns a number between 0 and T from an
    exponential distribution conditional on the outcome being between 0 and T'''
    t = random.expovariate(rate)
    L = int(t/T)
    return t - L*T

# The myQueue and ListDict classes are from Joel Miller's repository Mathematics-of-Epidemics-on-Networks
class myQueue(object):
    r'''

    This class is used to store and act on a priority queue of events for
    event-driven simulations.  It is based on heapq.
    Each queue is given a tmax (default is infinity) so that any event at later
    time is ignored.

    This is a priority queue of 4-tuples of the form
                   ``(t, counter, function, function_arguments)``
    The ``'counter'`` is present just to break ties, which generally only occur when
    multiple events are put in place for the initial condition, but could also
    occur in cases where events tend to happen at discrete times.
    note that the function is understood to have its first argument be t, and
    the tuple ``function_arguments`` does not include this first t.
    So function is called as
        ``function(t, *function_arguments)``
    Previously I used a class of events, but sorting using the __lt__ function

    I wrote was significantly slower than simply using tuples.
    '''
    def __init__(self, tmax=float("Inf")):
        self._Q_ = []
        self.tmax=tmax
        self.counter = 0 #tie-breaker for putting things in priority queue
    def add(self, time, function, args = ()):
        r'''time is the time of the event.  args are the arguments of the
        function not including the first argument which must be time'''
        if time<self.tmax:
            heapq.heappush(self._Q_, (time, self.counter, function, args))
            self.counter += 1
    def pop_and_run(self):
        r'''Pops the next event off the queue and performs the function'''
        t, counter, function, args = heapq.heappop(self._Q_)
        function(t, *args)
    def __len__(self): 
        r'''this will allow us to use commands like ``while Q:`` '''
        return len(self._Q_)



class _ListDict_(object):
    r'''

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

    I believe a faster data structure can be created with a (binary) tree.
    We add an object with a weight to the tree.  The nodes track their weights
    and the sum of the weights below it.  So choosing a random object (by weight)
    means that we choose a random number between 0 and weight_sum.  Then
    if it's less than the first node's weight, we choose that.  Otherwise,
    we see if the remaining bit is less than the total under the first child.
    If so, go there, otherwise, it's the other child.  Then iterate.  Adding
    a node would probably involve placing higher weight nodes higher in
    the tree.  Currently I don't have a fast enough implementation of this
    for my purposes.  So for now I'm sticking to the mixture of lists &
    dictionaries.

    I believe this structure I'm describing is similar to a "partial sum tree"
    or a "Fenwick tree", but they seem subtly different from this.
    '''
    def __init__(self, weighted = False):
        self.item_to_position = {}
        self.items = []

        self.weighted = weighted
        if self.weighted:
            self.weight = defaultdict(int) #presume all weights positive
            self.max_weight = 0
            self._total_weight = 0
            self.max_weight_count = 0

    def __len__(self):
        return len(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def _update_max_weight(self):
        C = Counter(self.weight.values())  #may be a faster way to do this, we only need to count the max.
        self.max_weight = max(C.keys())
        self.max_weight_count = C[self.max_weight]


    def insert(self, item, weight = None):
        r'''
        If not present, then inserts the thing (with weight if appropriate)
        if already there, replaces the weight unless weight is 0


        If weight is 0, then it removes the item and doesn't replace.

        WARNING:
            replaces weight if already present, does not increment weight.

        '''
        if self.__contains__(item):
            self.remove(item)
        if weight != 0:
            self.update(item, weight_increment=weight)


    def update(self, item, weight_increment = None):
        r'''
        If not present, then inserts the thing (with weight if appropriate)
        if already there, increments weight

        WARNING:
            increments weight if already present, cannot overwrite weight.
        '''
        if weight_increment is not None: #will break if passing a weight to unweighted case
            if weight_increment >0 or self.weight[item] != self.max_weight:
                self.weight[item] = self.weight[item] + weight_increment
                self._total_weight += weight_increment
                if self.weight[item] > self.max_weight:
                    self.max_weight_count = 1
                    self.max_weight = self.weight[item]
                elif self.weight[item] == self.max_weight:
                    self.max_weight_count += 1
            else: #it's a negative increment and was at max
                self.max_weight_count -= 1
                self.weight[item] = self.weight[item] + weight_increment
                self._total_weight += weight_increment
                self.max_weight_count -= 1
                if self.max_weight_count == 0:
                    self._update_max_weight

        elif self.weighted:
            raise Exception('if weighted, must assign weight_increment')

        if item in self: #we've already got it, do nothing else
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

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

                #if we find ourselves in this case often
                #it may be better just to let max_weight be the
                #largest weight *ever* encountered, even if all remaining weights are less
                #
                self.max_weight_count -= 1
                if self.max_weight_count == 0 and len(self)>0:
                    self._update_max_weight()

    def choose_random(self):
        # r'''chooses a random node.  If there is a weight, it will use rejection
        # sampling to choose a random node until it succeeds'''
        if self.weighted:
            while True:
                choice = random.choice(self.items)
                if random.random() < self.weight[choice]/self.max_weight:
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
        r'''uses other class methods to choose and then remove a random node'''
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
def discrete_SIR(H, tau, gamma, transmission_function=threshold, initial_infecteds=None, initial_recovereds = None, rho = None, tmin = 0, tmax = float('Inf'), dt=1.0, return_full_data = False, **args):

    # Initial infecteds and recovereds should be lists or None. Add a check here.
    if rho is not None and initial_infecteds is not None:
        raise Exception("Cannot define both initial_infecteds and rho")


    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(H.number_of_nodes()*rho))
        initial_infecteds = random.sample(list(H.nodes), initial_number)
    else:
        # check to make sure that the initially infected nodes are in the hypergraph
        initial_infecteds = list(set(H.nodes).intersection(set(initial_infecteds)))

    if initial_recovereds is None:
        initial_recovereds = []
    else:
        # check to make sure that the initially recovered nodes are in the hypergraph
        initial_recovereds = list(set(H.nodes).intersection(set(initial_recovereds)))

    status = defaultdict(lambda : 'S')

    if return_full_data:
        transition_events = dict()
        transition_events[tmin] = list()

    for node in initial_infecteds:
        status[node] = 'I'
        if return_full_data:
            transition_events[tmin].append(("I", node, None))

    for node in initial_recovereds:
        status[node] = 'R'
        if return_full_data:
            transition_events[tmin].append(("R", node))

    I = [len(initial_infecteds)]
    R = [len(initial_recovereds)]
    S = [H.number_of_nodes() - I[-1] - R[-1]]

    t = tmin
    times = [t]
    newStatus = status.copy()

    while t < tmax and I[-1] != 0:
        # Initialize the next step with the same numbers of S, I, and R as the last step before computing the changes
        S.append(S[-1])
        I.append(I[-1])
        R.append(R[-1])

        if return_full_data:
            transition_events[t+dt] = list()

        for node in H.nodes:
            if status[node] == 'I':
                # recover the node. If it is not healed, it stays infected.
                if random.random() <= gamma*dt:
                    newStatus[node] = 'R'
                    I[-1] += -1
                    R[-1] += 1
                    if return_full_data:
                        transition_events[t+dt].append(('R', node))
            elif status[node] == 'S':
                for edge_id in H.nodes[node]:
                    members = H.edges[edge_id]
                    neighbors = list(set(members).difference({node}))
                    if random.random() <= tau[len(members)]*transmission_function(status, neighbors, **args)*dt:
                        newStatus[node] = 'I'
                        S[-1] += -1
                        I[-1] += 1
                        if return_full_data:
                            transition_events[t+dt].append(('I', node, edge_id))
                        break
                # This executes after the loop has executed normally without hitting the break statement which indicates infection
                else:
                    newStatus[node] = 'S'
        status = newStatus.copy()
        t += dt
        times.append(t)
    if return_full_data:
        return transition_events
    else:
        return np.array(times), np.array(S), np.array(I), np.array(R)


def discrete_SIS(H, tau, gamma, transmission_function=collective_contagion, initial_infecteds=None, rho=None, tmin=0, tmax=100, dt=1.0, return_full_data=False, **args):
    # Initial infecteds and recovereds should be lists or None. Add a check here.

    if rho is not None and initial_infecteds is not None:
        raise Exception("Cannot define both initial_infecteds and rho")

    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(H.number_of_nodes()*rho))
        initial_infecteds = random.sample(list(H.nodes), initial_number)
    else:
        # check to make sure that the initially infected nodes are in the hypergraph
        initial_infecteds = list(set(H.nodes).intersection(set(initial_infecteds)))

    status = defaultdict(lambda : 'S')

    if return_full_data:
        transition_events = dict()
        transition_events[tmin] = list()

    for node in initial_infecteds:
        status[node] = 'I'
        if return_full_data:
            transition_events[tmin].append(("I", node, None))

    I = [len(initial_infecteds)]
    S = [H.number_of_nodes() - I[-1]]

    t = tmin
    times = [t]
    newStatus = status.copy()

    while t < tmax and I[-1] != 0:
        # Initialize the next step with the same numbers of S, I, and R as the last step before computing the changes
        S.append(S[-1])
        I.append(I[-1])
        if return_full_data:
            transition_events[t+dt] = list()

        for node in H.nodes:
            if status[node] == 'I':
                # recover the node. If it is not healed, it stays infected.
                if random.random() <= gamma*dt:
                    newStatus[node] = 'S'
                    I[-1] += -1
                    S[-1] += 1
                    if return_full_data:
                        transition_events[t+dt].append(('R', node))
            elif status[node] == 'S':
                for edge_id in H.nodes[node]:
                    members = H.edges[edge_id]
                    neighbors = list(set(members).difference({node}))
                    if random.random() <= tau[len(members)]*transmission_function(status, neighbors, **args)*dt:
                        newStatus[node] = 'I'
                        S[-1] += -1
                        I[-1] += 1
                        if return_full_data:
                            transition_events[t+dt].append(('I', node, edge_id))
                        break
                # This executes after the loop has executed normally without hitting the break statement which indicates infection, though I'm not sure we even need it
                else:
                    newStatus[node] = 'S'
        status = newStatus.copy()
        t += dt
        times.append(t)
    if return_full_data:
        return transition_events
    else:
        return np.array(times), np.array(S), np.array(I)


def Gillespie_SIR(H, tau, gamma, transmission_function=collective_contagion, initial_infecteds=None, initial_recovereds = None, rho = None, tmin = 0, tmax = float('Inf'), return_full_data = False, sim_kwargs = None, **args):

    # Initial infecteds and recovereds should be lists or None. Add a check here.

    if rho is not None and initial_infecteds is not None:
        raise Exception("Cannot define both initial_infecteds and rho")


    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(H.number_of_nodes()*rho))
        initial_infecteds = random.sample(list(H.nodes), initial_number)
    else:
        # check to make sure that the initially infected nodes are in the hypergraph
        initial_infecteds = list(set(H.nodes).intersection(set(initial_infecteds)))

    if initial_recovereds is None:
        initial_recovereds = []
    else:
        # check to make sure that the initially recovered nodes are in the hypergraph
        initial_recovereds = list(set(H.nodes).intersection(set(initial_recovereds)))

    status = defaultdict(lambda : 'S')

    size_dist = np.unique(H.edge_size_dist())

    for node in initial_infecteds:
        status[node] = 'I'

    for node in initial_recovereds:
        status[node] = 'R'

    I = [len(initial_infecteds)]
    R = [len(initial_recovereds)]
    S = [H.number_of_nodes() - I[-1] - R[-1]]

    t = tmin
    times = [t]

    infecteds = _ListDict_()

    infectious_edges = dict()
    for size in size_dist:
        infectious_edges[size] = _ListDict_()

    for node in initial_infecteds:
        infecteds.update(node)
        for edge_id in H.nodes[node]:
            members = H.edges[edge_id]
            for node in members:
                if status[node] == 'S':
                    neighbors = tuple(set(members).difference({node}))
                    contagion = transmission_function(status, neighbors, **args)
                    if contagion != 0:
                        infectious_edges[len(members)].update((neighbors, node))

    total_rates = dict()
    total_rates[1] = gamma*infecteds.total_weight()

    for size in size_dist:
        total_rates[size] = tau[size]*infectious_edges[size].total_weight()

    total_rate = sum(total_rates.values())

    dt = random.expovariate(total_rate)
    t += dt

    while t < tmax and I[-1] != 0:
        # choose type of event that happens
        # this can be improved

        while True:
            choice = random.choice(list(total_rates.keys()))
            if random.random() <= total_rates[choice]/total_rate:
                break

        if choice == 1: # recover
            recovering_node = infecteds.random_removal()
            status[recovering_node] = 'R'

            # remove edges that are no longer infectious because of this node recovering
            for edge_id in H.nodes[recovering_node]:
                members = H.edges[edge_id]
                for node in members:
                    if status[node] == 'S':
                        neighbors = tuple(set(members).difference({node}))
                        contagion = transmission_function(status, neighbors, **args)
                        if contagion == 0:
                            try:
                                infectious_edges[len(members)].remove((neighbors, node))
                            except:
                                pass
            times.append(t)
            S.append(S[-1])
            I.append(I[-1] - 1)
            R.append(R[-1] + 1)

        else:
            transmitters, recipient = infectious_edges[choice].choose_random()
            status[recipient] = 'I'

            infecteds.update(recipient)

            # remove the infectious links, because they can't infect an infected node.
            for edge_id in H.nodes[recipient]:
                members = H.edges[edge_id]
                neighbors = tuple(set(members).difference({recipient}))
                try:
                    infectious_edges[len(members)].remove((neighbors, recipient))
                except:
                    pass

            # add edges that are infectious because of this node being infected
            for edge_id in H.nodes[recipient]:
                members = H.edges[edge_id]
                for node in members:
                    if status[node] == 'S':
                        neighbors = tuple(set(members).difference({node}))
                        contagion = transmission_function(status, neighbors, **args)
                        if contagion == 1:
                            try:
                                infectious_edges[len(members)].update((neighbors, node))
                            except:
                                pass
            times.append(t)
            S.append(S[-1] - 1)
            I.append(I[-1] + 1)
            R.append(R[-1])

        total_rates[1] = gamma*infecteds.total_weight()
        for size in size_dist:
            total_rates[size] = tau[size]*infectious_edges[size].total_weight()

        total_rate = sum(total_rates.values())

        if total_rate > 0:
            dt = random.expovariate(total_rate)
        else:
            dt = float('Inf')
        t += dt
    return np.array(times), np.array(S), np.array(I), np.array(R)


def Gillespie_SIS(H, tau, gamma, transmission_function=collective_contagion, initial_infecteds=None, rho = None, tmin = 0, tmax = float('Inf'), return_full_data = False, sim_kwargs = None, **args):

    # Initial infecteds and recovereds should be lists or None. Add a check here.

    if rho is not None and initial_infecteds is not None:
        raise Exception("Cannot define both initial_infecteds and rho")


    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(H.number_of_nodes()*rho))
        initial_infecteds = random.sample(list(H.nodes), initial_number)
    else:
        # check to make sure that the initially infected nodes are in the hypergraph
        initial_infecteds = list(set(H.nodes).intersection(set(initial_infecteds)))

    status = defaultdict(lambda : 'S')

    size_dist = np.unique(H.edge_size_dist())

    for node in initial_infecteds:
        status[node] = 'I'

    I = [len(initial_infecteds)]
    S = [H.number_of_nodes() - I[-1]]

    t = tmin
    times = [t]

    infecteds = _ListDict_()

    infectious_edges = dict()
    for size in size_dist:
        infectious_edges[size] = _ListDict_()

    for node in initial_infecteds:
        infecteds.update(node)
        for edge_id in H.nodes[node]:
            members = H.edges[edge_id]
            for node in members:
                if status[node] == 'S':
                    neighbors = tuple(set(members).difference({node}))
                    contagion = transmission_function(status, neighbors, **args)
                    if contagion != 0:
                        infectious_edges[len(members)].update((neighbors, node))

    total_rates = dict()
    total_rates[1] = gamma*infecteds.total_weight()
    for size in size_dist:
        total_rates[size] = tau[size]*infectious_edges[size].total_weight()

    total_rate = sum(total_rates.values())

    dt = random.expovariate(total_rate)
    t += dt

    while t < tmax and I[-1] != 0:
        # choose type of event that happens
        # this can be improved
        while True:
            choice = random.choice(list(total_rates.keys()))
            if random.random() <= total_rates[choice]/total_rate:
                break

        if choice == 1: # recover
            recovering_node = infecteds.random_removal()
            status[recovering_node] = 'S'

            # remove edges that are no longer infectious because of this node recovering
            for edge_id in H.nodes[recovering_node]:
                members = H.edges[edge_id]
                for node in members:
                    if status[node] == 'S':
                        neighbors = tuple(set(members).difference({node}))
                        contagion = transmission_function(status, neighbors, **args)
                        if contagion == 0:
                            try:
                                infectious_edges[len(members)].remove((neighbors, node))
                            except:
                                pass
            times.append(t)
            S.append(S[-1] + 1)
            I.append(I[-1] - 1)

        else:
            transmitters, recipient = infectious_edges[choice].choose_random()
            status[recipient] = 'I'

            infecteds.update(recipient)

            # remove the infectious links, because they can't infect an infected node.
            for edge_id in H.nodes[recipient]:
                members = H.edges[edge_id]
                neighbors = tuple(set(members).difference({recipient}))
                try:
                    infectious_edges[len(members)].remove((neighbors, recipient))
                except:
                    pass

            # add edges that are infectious because of this node being infected
            for edge_id in H.nodes[recipient]:
                members = H.edges[edge_id]
                for node in members:
                    if status[node] == 'S':
                        neighbors = tuple(set(members).difference({node}))
                        contagion = transmission_function(status, neighbors, **args)
                        if contagion == 1:
                            try:
                                infectious_edges[len(members)].update((neighbors, node))
                            except:
                                pass
            times.append(t)
            S.append(S[-1] - 1)
            I.append(I[-1] + 1)

        total_rates[1] = gamma*infecteds.total_weight()
        for size in size_dist:
            total_rates[size] = tau[size]*infectious_edges[size].total_weight()

        total_rate = sum(total_rates.values())

        if total_rate > 0:
            dt = random.expovariate(total_rate)
        else:
            dt = float('Inf')
        t += dt

    return np.array(times), np.array(S), np.array(I)
