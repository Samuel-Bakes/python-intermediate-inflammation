import numpy.testing as npt
from pathlib import Path
import glob
import os
import pytest
import numpy as np
from pathlib import Path

from inflammation import models
from unittest.mock import Mock

def test_analyse_data():
    """Test for changes in behaviour by comparing current behaviour with old behaviour"""
    from inflammation.compute_data import analyse_data
    from inflammation.compute_data import CSVDataSource

    data_dir = Path.cwd() / "data"
    data_source = CSVDataSource(data_dir)
    new_result = analyse_data(data_source)

    data_file_paths = glob.glob(os.path.join(data_dir, 'inflammation*.csv'))
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

def test_compute_data_mock_source():
  from inflammation.compute_data import analyse_data
  data_source = Mock()

  data_source.read_patient_data.return_value = [[[0, 1, 0],[0,2,0]]]
  
  result = analyse_data(data_source)

  npt.assert_array_equal(result, [0,0,0])
