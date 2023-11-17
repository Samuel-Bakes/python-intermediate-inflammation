import numpy.testing as npt
from pathlib import Path
import glob
import os
import pytest
import numpy as np

from inflammation import models, views

def test_analyse_data():
    """Test for changes in behaviour by comparing current behaviour with old behaviour"""
    from inflammation.compute_data import analyse_data
    path = Path.cwd() / "data"
    new_result = analyse_data()

    data_file_paths = glob.glob(os.path.join(path, 'inflammation*.csv'))
    if len(data_file_paths) == 0:
        raise ValueError(f"No inflammation csv's found in path {data_dir}")
    data = map(models.load_csv, data_file_paths)


    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    old_result = np.std(means_by_day_matrix, axis=0)

    npt.assert_array_equal(new_result, old_result)

def test_standard_deviation():
    from inflammation.compute_data import compute_standard_deviation_by_day
    file = [[0, 1, 0],[0,2,0]]
    input_data = [file]

    result = compute_standard_deviation_by_day(input_data)

    npt.assert_array_equal(result, [0,0,0])