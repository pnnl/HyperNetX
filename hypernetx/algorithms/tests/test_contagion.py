import numpy as np
from hypernetx.algorithms.contagion import (
    Gillespie_SIR,
    Gillespie_SIS,
    discrete_SIR,
    discrete_SIS,
    majority_vote,
    collective_contagion,
    individual_contagion,
    threshold,
)
import hypernetx as hnx
import random


# Test the contagion functions
def test_collective_contagion():
    status = {0: "S", 1: "I", 2: "I", 3: "S", 4: "R"}
    assert collective_contagion(0, status, (0, 1, 2)) == True
    assert collective_contagion(1, status, (0, 1, 2)) == False
    assert collective_contagion(3, status, (0, 1, 2)) == False


def test_individual_contagion():
    status = {0: "S", 1: "I", 2: "I", 3: "S", 4: "R"}
    assert individual_contagion(0, status, (0, 1, 3)) == True
    assert individual_contagion(1, status, (0, 1, 2)) == False
    assert individual_contagion(3, status, (0, 3, 4)) == False


def test_threshold():
    status = {0: "S", 1: "I", 2: "I", 3: "S", 4: "R"}
    assert threshold(0, status, (0, 2, 3, 4), tau=0.2) == True
    assert threshold(0, status, (0, 2, 3, 4), tau=0.5) == False
    assert threshold(1, status, (1, 2, 3), tau=1) == False


def test_majority_vote():
    status = {0: "S", 1: "I", 2: "I", 3: "S", 4: "R"}
    assert majority_vote(0, status, (0, 1, 2)) == True
    assert majority_vote(0, status, (0, 1, 2, 3)) == True
    assert majority_vote(1, status, (0, 1, 2)) == False
    assert majority_vote(3, status, (0, 1, 2)) == False


# Test the epidemic simulations
def test_discrete_SIR():
    random.seed(42)
    n = 100
    m = 1000
    hyperedgeList = [random.sample(range(n), k=random.choice([2, 3])) for i in range(m)]
    H = hnx.Hypergraph(hyperedgeList)
    tau = {2: 0.1, 3: 0.1}
    gamma = 0.1
    tmax = 100
    dt = 0.1
    t, S, I, R = discrete_SIR(H, tau, gamma, rho=0.1, tmin=0, tmax=tmax, dt=dt)
    assert max(t) < tmax + dt
    assert t[1] - t[0] == dt
    # checks population conservation over all time
    assert np.array_equal(S + I + R, n * np.ones(len(t))) == True


def test_discrete_SIS():
    random.seed(42)
    n = 100
    m = 1000
    hyperedgeList = [random.sample(range(n), k=random.choice([2, 3])) for i in range(m)]
    H = hnx.Hypergraph(hyperedgeList)
    tau = {2: 0.1, 3: 0.1}
    gamma = 0.1
    tmax = 100
    dt = 0.1
    t, S, I = discrete_SIS(H, tau, gamma, rho=0.1, tmin=0, tmax=tmax, dt=dt)
    assert max(t) < tmax + dt
    assert t[1] - t[0] == dt
    # checks population conservation over all time
    assert np.array_equal(S + I, n * np.ones(len(t))) == True


def test_Gillespie_SIR():
    random.seed(42)
    n = 100
    m = 1000
    hyperedgeList = [random.sample(range(n), k=random.choice([2, 3])) for i in range(m)]
    H = hnx.Hypergraph(hyperedgeList)
    tau = {2: 0.1, 3: 0.1}
    gamma = 0.1
    tmax = 100
    t, S, I, R = Gillespie_SIR(H, tau, gamma, rho=0.1, tmin=0, tmax=tmax)
    assert max(t) < tmax
    # checks population conservation over all time
    assert np.array_equal(S + I + R, n * np.ones(len(t))) == True


def test_Gillespie_SIS():
    random.seed(42)
    n = 100
    m = 1000
    hyperedgeList = [random.sample(range(n), k=random.choice([2, 3])) for i in range(m)]
    H = hnx.Hypergraph(hyperedgeList)
    tau = {2: 0.1, 3: 0.1}
    gamma = 0.1
    tmax = 100
    t, S, I = Gillespie_SIS(H, tau, gamma, rho=0.1, tmin=0, tmax=tmax)
    assert max(t) < tmax
    # checks population conservation over all time
    assert np.array_equal(S + I, n * np.ones(len(t))) == True
