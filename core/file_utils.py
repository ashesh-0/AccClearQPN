import os
from datetime import datetime

from core.constants import RADAR_DIR, RAINFALL_DIR
from core.time_utils import TimeSteps


class FileManager(TimeSteps):
    def __init__(self, directory):
        super().__init__()
        self._schema = None
        self._dir = directory

    def nth_next(self, fpath, n):
        dt = self.dt_from_fpath(fpath)
        dt = super().nth_next(dt, n)
        return self.fpath_from_dt(dt)

    def nth_previous(self, fpath, n):
        dt = self.dt_from_fpath(fpath)
        dt = super().nth_previous(dt, n)
        return self.fpath_from_dt(dt)

    def next(self, fpath):
        return self.nth_next(fpath, 1)

    def previous(self, fpath):
        return self.nth_previous(fpath, 1)

    def get_dir(self, dt: datetime, base_dir: str):
        return os.path.join(
            base_dir,
            str(dt.year),
            f'{dt.year}{dt.month:02}',
            f'{dt.year}{dt.month:02}{dt.day:02}',
        )

    def get_base_dir(self, fpath):
        # /tmp2/ashesh/precipitation/compressed_radarV2/2015/201506/20150601/MREF3D21L.20150601.1430.gz
        output_dir = fpath
        for _ in range(4):
            output_dir = os.path.dirname(output_dir)
        return output_dir

    def fpath_from_dt(self, dt):
        return os.path.join(self._dir, dt.strftime(self._schema))


class RadarFileManager(FileManager):
    def __init__(self, directory=RADAR_DIR):
        super().__init__(directory)
        self._schema = '%Y/%Y%m/%Y%m%d/MREF3D21L.%Y%m%d.%H%M.gz'

    def dt_from_fpath(self, fpath):
        # MREF3D21L.20150601.1430.gz
        tokens = os.path.basename(fpath).split('.')
        assert len(tokens) == 4
        return datetime.strptime('-'.join(tokens[1:3]), '%Y%m%d-%H%M')


class RainFileManager(FileManager):
    def __init__(self, directory=RAINFALL_DIR, compressed=False):
        super().__init__(directory)
        self._compressed = compressed
        self._schema = '%Y/%Y%m/%Y%m%d/%Y%m%d_%H%M.nc'

        if self._compressed:
            self._schema += '.gz'

    def dt_from_fpath(self, fpath):
        # 20150601_0700.nc
        tokens = os.path.basename(fpath).split('.')
        assert len(tokens) == 2 or (len(tokens) == 3 and self._compressed)
        token = tokens[0]
        return datetime.strptime(token, '%Y%m%d_%H%M')


if __name__ == '__main__':
    fpath = '2015/201506/20150601/MREF3D21L.20150601.0000.gz'
    print('Original', fpath)
    print('Next', RadarFileManager().next(fpath))
    print('Prev', RadarFileManager().previous(fpath))
    print('')
    fpath = '/tmp2/ashesh/precipitation/rainfall_organized/2015/201506/20150601/20150601_0000.nc'
    print('Original', fpath)
    print('Next', RainFileManager().next(fpath))
    print('Prev', RainFileManager().previous(fpath))
    # from shutil import copyfile
    # direc = '/tmp2/ashesh/precipitation/rainfall/201506/'
    # new_dir_root = '/tmp2/ashesh/precipitation/rainfall_organized/'
    # fm = RainFileManager()
    # for fname in os.listdir(direc):
    #     dt = fm.dt_from_fpath(fname)
    #     new_dir = fm.get_dir(dt, new_dir_root)
    #     os.makedirs(new_dir, exist_ok=True)
    #     copyfile(os.path.join(direc, fname), os.path.join(new_dir, fname))
