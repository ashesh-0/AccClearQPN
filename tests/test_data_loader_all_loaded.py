import os
import sys
from datetime import datetime, timedelta

import numpy as np
from skimage.transform import resize

# conda install -c conda-forge basemap-data-hires
from core.constants import LOGALT_Q95, RADAR_Q95, RAIN_Q95, TIME_GRANULARITY_MIN
from core.base_enum import DataType
from core.file_utils import RadarFileManager
from data_loader import DataLoader
from data_loader_all_loaded import DataLoaderAllLoaded

os.environ['PROJ_LIB'] = '/tmp2/ashesh/miniconda3/envs/venv/share/proj/'
os.environ['ROOT_DATA_DIR'] = '/home/ashesh/mount/precipitation/'

s = datetime(2017, 1, 1)
e = datetime(2017, 1, 31, 23, 50)
input_len = 5
target_len = 3
target_offset = 2
target_avg_length = 6

loader = DataLoaderAllLoaded(
    s,
    e,
    input_len,
    target_len,
    target_offset=target_offset,
    target_avg_length=target_avg_length,
    data_type=DataType.Radar + DataType.Rain,
    is_train=True,
    img_size=(540, 420),
)


def test_data_loader_future_target():
    global input_len, target_len
    s = input_len + target_offset
    e = s + target_avg_length
    data = [loader._ccrop(loader._get_raw_rain_data(idx))[None, ...] for idx in range(s, e)]
    data = np.concatenate(data, axis=0)
    inp, target, mask = loader[0]
    assert np.all(target[0] == np.mean(data, axis=0))

    data = [loader._ccrop(loader._get_raw_rain_data(idx))[None, ...] for idx in range(0, input_len)]
    data = np.concatenate(data, axis=0)
    data[data < 0] = 0
    data = data / RAIN_Q95

    assert np.all(inp[:, 1, :] == data)


def test_data_loader_disjoint_target():
    global input_len, target_len
    s = input_len + target_offset
    e = s + target_avg_length * target_len
    data = [loader._ccrop(loader._get_raw_rain_data(idx))[None, ...] for idx in range(s, e)]
    data = np.concatenate(data, axis=0)
    inp, target, mask = loader[0]
    assert np.all(target[0] == np.mean(data[:target_avg_length], axis=0))
    assert np.all(target[1] == np.mean(data[target_avg_length:2 * target_avg_length], axis=0))
    assert np.all(target[2] == np.mean(data[2 * target_avg_length:3 * target_avg_length], axis=0))
