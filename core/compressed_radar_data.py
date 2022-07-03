import argparse
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from core.constants import DBZ_Z, NX, NY
from core.file_utils import RadarFileManager
from core.raw_data import RawRadarData



class CompressedRadarData:
    INVALID_VALUE = -990

    def __init__(self, fpath):
        self._fpath = fpath
        assert self._fpath[-3:] == '.gz'

    def datetime(self):
        tokens = os.path.basename(self._fpath).split('.')
        assert len(tokens) == 4
        return datetime.strptime('-'.join(tokens[-2:]), '%Y%m%d-%H%M')

    def load_raw(self):
        return np.loadtxt(self._fpath, dtype=np.int16)

    def load(self):
        raw_data = self.load_raw()
        return self.load_from_raw(raw_data)

    def load_from_raw(self, raw_data):
        data = self.__class__.INVALID_VALUE * np.ones((NX, NY, DBZ_Z), dtype=np.int16)
        if len(raw_data) == 0:
            return data
        elif len(raw_data) == 4 and len(raw_data.shape) == 1:
            data[raw_data[0], raw_data[1], raw_data[2]] = raw_data[3]
        else:
            data[raw_data[0, :], raw_data[1, :], raw_data[2, :]] = raw_data[3, :]
        return data


class RadarDataCompressor:
    def __init__(self, source_dir: str, destination_dir: str, overwrite: bool = True):
        self._sdir = source_dir
        self._ddir = destination_dir
        self._fm = RadarFileManager()
        self._overwrite = overwrite
        if not os.path.exists(self._ddir):
            os.mkdir(self._ddir)
        print(f'[{self.__class__.__name__}] SRC:{self._sdir} DST:{self._ddir} Overwrite:{self._overwrite}')

    def create_dir(self, dt: datetime):
        day_dir = self._fm.get_dir(dt, self._ddir)
        os.makedirs(day_dir, exist_ok=True)

    def preprocess_data(self, data):
        f1 = data != -9990
        f2 = data != -990
        filtr = np.logical_and(f1, f2)
        d0, d1, d2 = np.where(filtr)
        sane_values = data[d0, d1, d2]
        compressed_data = np.vstack([d0, d1, d2, sane_values])
        return compressed_data

    def target_path(self, dt: datetime, fname: str):
        dirname = self._fm.get_dir(dt, self._ddir)
        fpath = os.path.join(dirname, fname + '.gz')
        return fpath

    def save(self, data: np.array, dt: datetime, fname: str):
        fpath = self.target_path(dt, fname)
        np.savetxt(fpath, self.preprocess_data(data), fmt='%i')

    def generate_one(self, fname: str, day_dir: str):
        fpath = os.path.join(day_dir, fname)
        loader = RawRadarData(fpath)
        dt = loader.datetime()
        if self._overwrite is False and os.path.exists(self.target_path(dt, fname)):
            return
        data = loader.load()
        self.create_dir(dt)
        self.save(data, dt, fname)

    def generate(self, workers: int = 2):
        for year in os.listdir(self._sdir):
            year_dir = os.path.join(self._sdir, year)
            for month in os.listdir(year_dir):
                print(month)
                month_dir = os.path.join(year_dir, month)
                for day in tqdm(list(os.listdir(month_dir))):
                    day_dir = os.path.join(month_dir, day)
                    arguments = []
                    for fname in os.listdir(day_dir):
                        arguments.append((fname, day_dir))

                    with Pool(processes=workers) as pool:
                        pool.starmap(self.generate_one, arguments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='source directory. It should contain year as subdirectories')
    parser.add_argument('dest', type=str, help='destination directory')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    comp = RadarDataCompressor(args.src, args.dest, overwrite=args.overwrite)
    comp.generate(workers=args.workers)
