import pandas as pd
import os


__all__ = ["TransmissionProblem"]

current_dir = os.path.dirname(os.path.abspath(__file__))


class TransmissionProblem(object):
    def __init__(self):

        try:
            csvfile = "https://raw.githubusercontent.com/pnnl/HyperNetX/master/hypernetx/utils/toys/ChungLuTransmissionData.csv"
            self.df = pd.read_csv(csvfile, header=None, names=["receivers", "senders"])
        except:
            csvfile = f"{current_dir}/ChungLuTransmissionData.csv"
            self.df = pd.read_csv(csvfile, header=None, names=["receivers", "senders"])
