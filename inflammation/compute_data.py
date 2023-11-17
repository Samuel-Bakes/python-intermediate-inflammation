"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np
from pathlib import Path

from inflammation import models, views

class CSVDataSource:
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.data_file_path = glob.glob(os.path.join(data_dir, 'inflammation*.csv'))

    def read_patient_data(self):
        if len(self.data_file_path) == 0:
            raise ValueError(f"No inflammation csv's found in path {self.data_dir}")
        data = map(models.load_csv, self.data_file_path)

        return data
    
class JSONDataSource:
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.data_file_path = glob.glob(os.path.join(data_dir, 'inflammation*.json'))

    def read_patient_data(self):
        if len(self.data_file_path) == 0:
            raise ValueError(f"No inflammation csv's found in path {self.data_dir}")
        data = map(models.load_json, self.data_file_path)

        return data

def analyse_data(data_source):
    """Calculate the standard deviation by day between datasets

    Gets all the inflammation csvs within a directory, works out the mean
    inflammation value for each day across all datasets, then graphs the
    standard deviation of these means."""
    data = data_source.read_patient_data()

    daily_standard_deviation = compute_standard_deviation_by_day(data)
    graph_data = {
        'standard deviation by day': daily_standard_deviation,
    }
    views.visualize(graph_data)

    return daily_standard_deviation

def compute_standard_deviation_by_day(data):
    """Calculate the standard deviation by day between datasets"""
    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))
    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)
    return daily_standard_deviation

