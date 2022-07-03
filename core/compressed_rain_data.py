import argparse
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from core.constants import NX, NY
from core.file_utils import RainFileManager
from core.raw_data import RawRainData


class CompressedRainData:
    def __init__(self, fpath):
        self._fpath = fpath

    def datetime(self):
        tokens = os.path.basename(self._fpath).split('.')
        assert len(tokens) == 3
        return datetime.strptime(tokens[0], '%Y%m%d_%H%M')

    def load_raw(self, can_raise_error=True):
        if (not can_raise_error) and (not os.path.exists(self._fpath)):
            return np.array([])

        return np.loadtxt(self._fpath, dtype=np.int16)

    def load(self):
        raw_data = self.load_raw()
        return self.load_from_raw(raw_data)

    @staticmethod
    def load_from_raw(raw_data):
        data = np.zeros((NX, NY), dtype=np.int16)
        if len(raw_data) == 0:
            return data
        elif len(raw_data) == 3 and len(raw_data.shape) == 1:
            data[raw_data[0], raw_data[1]] = raw_data[2]
        else:
            data[raw_data[0, :], raw_data[1, :]] = raw_data[2, :]
        return data


class RainDataCompressor:
    def __init__(self, source_dir: str, destination_dir: str, overwrite: bool = False):
        self._sdir = source_dir
        self._ddir = destination_dir
        self._fm = RainFileManager()
        self._overwrite = overwrite
        if not os.path.exists(self._ddir):
            os.mkdir(self._ddir)
        print(f'[{self.__class__.__name__}] SRC:{self._sdir} DST:{self._ddir} Overwrite:{self._overwrite}')

    def create_dir(self, dt: datetime):
        day_dir = self._fm.get_dir(dt, self._ddir)
        os.makedirs(day_dir, exist_ok=True)

    def preprocess_data(self, data):
        d0, d1 = np.where(data > 0)
        sane_values = data[d0, d1]
        compressed_data = np.vstack([d0, d1, sane_values])
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
        try:
            loader = RawRainData(fpath)
            dt = loader.datetime()
            if self._overwrite is False and os.path.exists(self.target_path(dt, fname)):
                return
            data = loader.load()['rain']
            self.create_dir(dt)
            self.save(data, dt, fname)
        except:
            print(f'Issue detected for {fpath}')

    def generate(self, workers: int = 2):
        tqdm_inst = tqdm(os.listdir(self._sdir))
        for year_month in tqdm_inst:
            tqdm_inst.set_description(f'Processing {year_month}')
            ym_dir = os.path.join(self._sdir, year_month)
            arguments = []
            # for fname in tqdm(os.listdir(ym_dir)):
            #     self.generate_one(fname, ym_dir)
            for fname in os.listdir(ym_dir):
                arguments.append((fname, ym_dir))
            with Pool(processes=workers) as pool:
                pool.starmap(self.generate_one, arguments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='source directory. It should contain year as subdirectories')
    parser.add_argument('dest', type=str, help='destination directory')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    comp = RainDataCompressor(args.src, args.dest, overwrite=args.overwrite)
    comp.generate(workers=args.workers)
