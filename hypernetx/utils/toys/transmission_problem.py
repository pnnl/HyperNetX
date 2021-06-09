import pandas as pd
import os


csvfile = 'ChungLuTransmissionData.csv'
csvfile = os.path.join(os.path.dirname(__file__), csvfile)


class TransmissionProblem(object):
    def __init__(self, csvfile=csvfile):
        csvfile = csvfile
        self.df = pd.read_csv(csvfile, header=None, names=['receivers', 'senders'])
