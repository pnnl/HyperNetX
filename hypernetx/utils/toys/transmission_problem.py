import pandas as pd


csvfile = 'ChungLuTransmissionData_lg.csv'


class TransmissionProblem(object):
    def __init__(self, csvfile=csvfile):
        csvfile = csvfile
        self.df = pd.read_csv(csvfile, header=None, names=['receivers', 'senders'])
