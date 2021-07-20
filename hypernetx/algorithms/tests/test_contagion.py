import numpy as np
import pytest
import warnings
import hypernetx.algorithms.contagion as contagion
import hypernetx as hnx

# Test the contagion functions
def test_collective_contagion():
    status = {0:"S", 1:"I", 2:"I", 3:"S", 4:"R"}
    assert contagion.collective_contagion(0, status, (0, 1, 2)) == True
    assert contagion.collective_contagion(1, status, (0, 1, 2)) == False
    assert contagion.collective_contagion(3, status, (0, 1, 2)) == False

def test_individual_contagion():
    status = {0:"S", 1:"I", 2:"I", 3:"S", 4:"R"}
    assert contagion.individual_contagion(0, status, (0, 1, 3)) == True
    assert contagion.individual_contagion(1, status, (0, 1, 2)) == False
    assert contagion.individual_contagion(3, status, (0, 3, 4)) == False

def test_threshold():
    status = {0:"S", 1:"I", 2:"I", 3:"S", 4:"R"}
    assert contagion.threshold(0, status, (0, 2, 3, 4), tau=0.2) == True
    assert contagion.threshold(0, status, (0, 2, 3, 4), tau=0.5) == False
    assert contagion.threshold(1, status, (1, 2, 3), tau=1) == False

def test_majority_vote():
    status = {0:"S", 1:"I", 2:"I", 3:"S", 4:"R"}
    assert contagion.majority_vote(0, status, (0, 1, 2)) == True
    assert contagion.majority_vote(0, status, (0, 1, 2, 3)) == True
    assert contagion.majority_vote(1, status, (0, 1, 2)) == False
    assert contagion.majority_vote(3, status, (0, 1, 2)) == False
