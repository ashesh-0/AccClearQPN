"""
This is a QPESUMS 1 hour baseline.
"""
import os
import pickle
from datetime import datetime, timedelta

from tqdm import tqdm

from core.constants import INPUT_LEN, TIME_GRANULARITY_MIN
from core.dataset import load_data
from core.raw_data import RawQpesumsData
from data_loaders.data_loader_all_loaded import DataLoaderAllLoaded


def load_qpesum_data(fpath, month_start, month_end, pkl_path='/tmp2/ashesh/precipitation/2018_QPESUMS.pkl'):
    # if isinstance(pkl_path, str):
    #     print('Loading QPESUMS benchmark from pickle file:', pkl_path)
    #     assert os.path.exists(pkl_path) and len(pkl_path) > 0
    #     with open(pkl_path, 'rb') as f:
    #         return pickle.load(f)
    output_dict = {}
    year = 2018
    for month in tqdm(range(month_start, month_end + 1)):
        sdir = os.path.join(fpath, f'{year}{month:02d}')
        for fname in os.listdir(sdir):
            data = RawQpesumsData(os.path.join(sdir, fname)).load()
            output_dict[data['dt']] = data['rain']
    return output_dict


class QpesumsBaseline(DataLoaderAllLoaded):
    def __init__(self,
                 start,
                 end,
                 target_len=1,
                 fpath='/tmp2/ashesh/precipitation/QPESUMS/',
                 img_size=(540, 420),
                 is_validation=False,
                 is_test=False,
                 is_train=False):
        self._benchmark = load_qpesum_data(fpath, start.month, end.month)
        super().__init__(
            start,
            end,
            6,
            target_len,
            0,
            data_type=1,
            img_size=img_size,
            sampling_rate=5,
            is_test=is_test,
            is_validation=is_validation,
            is_train=is_train)

    def _set_index_map(self):
        # self._benchmark = load_qpesum_data(fpath, start.month, end.month)
        raw_idx = 0
        skip_counter = 0
        benchmark_skip_counter = 0
        while raw_idx < len(self._time):
            target_offset = self._ilen + self._toffset + self._tavg_len * self._tlen
            if raw_idx + target_offset >= len(self._time):
                break

            if self._time[raw_idx + target_offset] - self._time[raw_idx] > timedelta(
                    seconds=TIME_GRANULARITY_MIN * target_offset * 60):
                skip_counter += 1
            elif self._time[raw_idx + self._ilen + self._toffset] not in self._benchmark:
                skip_counter += 1
                benchmark_skip_counter += 1
            else:
                self._index_map.append(raw_idx)

            raw_idx += 1
        print(f'[{self.__class__.__name__}] Size:{len(self._index_map)//1000}K Skipped:{skip_counter} '
              f'Bmk:{benchmark_skip_counter}')

    def __getitem__(self, index):
        output = super().__getitem__(index)
        ts = self.target_ts(index)
        prediction = self._benchmark[ts]
        prediction = self._ccrop(prediction)
        return (*output, prediction[None, ...])
