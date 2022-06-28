"""
Here, we aggregate the radar data. We take the MAX across all 21 channels as its value.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from core.compressed_radar_data import CompressedRadarData
from core.constants import NX, NY
from core.file_utils import RadarFileManager


class AggregateMode:
    MAX = 1


class RadarDataAggregator:
    def __init__(self, source_dir, destination_dir, amode, overwrite=True):
        self._sdir = source_dir
        self._ddir = destination_dir
        self._amode = amode
        self._overwrite = overwrite

    def get_target_fpath(self, src_fname, create_new=True):
        fm = RadarFileManager()
        dt = fm.dt_from_fpath(src_fname)
        dirname = fm.get_dir(dt, self._ddir)
        if create_new:
            os.makedirs(dirname, exist_ok=True)

        fpath = os.path.join(dirname, src_fname)
        return fpath

    def save_one(self, data, src_fname):
        fpath = self.get_target_fpath(src_fname, self._ddir)
        np.savetxt(fpath, data, fmt='%i')

    def preprocess_data(self, data):
        f1 = data != -9990
        f2 = data != -990
        filtr = np.logical_and(f1, f2)
        d0, d1 = np.where(filtr)

        sane_values = data[d0, d1]
        compressed_data = np.vstack([d0, d1, sane_values])
        return compressed_data

    def aggregate_one(self, src_fname, day_dir):
        assert self._amode == AggregateMode.MAX
        fpath = os.path.join(day_dir, src_fname)
        if os.path.exists(self.get_target_fpath(src_fname, self._ddir)) and self._overwrite is False:
            return

        data = CompressedRadarData(fpath).load().max(axis=-1)
        data = self.preprocess_data(data)
        self.save_one(data, src_fname)

    def run(self, workers):
        for year in os.listdir(self._sdir):
            year_dir = os.path.join(self._sdir, year)
            for month in os.listdir(year_dir):
                print(month)
                month_dir = os.path.join(year_dir, month)
                for day in tqdm(list(os.listdir(month_dir))):
                    day_dir = os.path.join(month_dir, day)
                    arguments = []
                    # for fname in os.listdir(day_dir):
                    #     self.aggregate_one(fname, day_dir)
                    for fname in os.listdir(day_dir):
                        arguments.append((fname, day_dir))

                    with Pool(processes=workers) as pool:
                        pool.starmap(self.aggregate_one, arguments)


class CompressedAggregatedRadarData:
    INVALID_VALUE = -990

    def __init__(self, fpath):
        self._fpath = fpath
        assert self._fpath[-3:] == '.gz'

    def load_raw(self):
        return np.loadtxt(self._fpath, dtype=np.int16)

    def load(self):
        raw_data = self.load_raw()
        return self.load_from_raw(raw_data)

    @staticmethod
    def load_from_raw(raw_data):
        data = CompressedAggregatedRadarData.INVALID_VALUE * np.ones((NX, NY), dtype=np.int16)
        if len(raw_data) == 0:
            return data
        assert raw_data.shape[0] == 3, f'{raw_data.shape}'

        if len(raw_data) == 3 and len(raw_data.shape) == 1:
            data[raw_data[0], raw_data[1]] = raw_data[2]
        else:
            data[raw_data[0, :], raw_data[1, :]] = raw_data[2, :]
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='source directory. It should contain year as subdirectories')
    parser.add_argument('dest', type=str, help='destination directory')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    comp = RadarDataAggregator(args.src, args.dest, AggregateMode.MAX, overwrite=args.overwrite)
    comp.run(workers=args.workers)
