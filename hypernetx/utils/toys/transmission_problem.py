import pandas as pd
import os


__all__ = ['TransmissionProblem']

csvfile = 'ChungLuTransmissionData.csv'
csvfile = os.path.join(os.path.dirname(__file__), csvfile)


class TransmissionProblem(object):
    def __init__(self, csvfile=csvfile):
        csvfile = csvfile
        try:
            self.df = pd.read_csv(csvfile, header=None, names=['receivers', 'senders'])
        except:
            fname = "https://raw.githubusercontent.com/pnnl/HyperNetX/master/hypernetx/utils/toys/ChungLuTransmissionData.csv"
            self.df = pd.read_csv(fname, header=None, names=['receivers', 'senders'])
