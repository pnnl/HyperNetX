# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

import pickle

def to_pickle(obj,filename):
    '''Writes object to a pickle file'''
    with open(f'{filename}', 'wb') as f:
        pickle.dump(obj,f)

def load_from_pickle(filepath):
    '''Returns object from file'''
    with open(filepath, 'rb') as f:
        temp = pickle.load(f)
    return temp
