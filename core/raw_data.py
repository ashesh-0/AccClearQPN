import os
import struct
from datetime import datetime

import numpy as np

from core.constants import DBZ_Z, NX, NY
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/


class RawRadarData:
    def __init__(self, fpath):
        """
        Example fpath: MREF3D21L.20180908.1300
        """
        self._fpath = fpath

    def datetime(self):
        tokens = os.path.basename(self._fpath).split('.')
        assert len(tokens) == 3
        return datetime.strptime('-'.join(tokens[-2:]), '%Y%m%d-%H%M')

    def load(self):
        dBZ = np.zeros((NX, NY, DBZ_Z))

        f1 = open(self._fpath, 'rb')
        _ = f1.read(242)
        Num = f1.read(4)
        Num_i = struct.unpack('i', Num)[0]

        for i in range(Num_i):
            _ = f1.read(4)

        for k in range(DBZ_Z):
            for i in range(NX):
                for j in range(NY):
                    data = f1.read(2)
                    data_float = struct.unpack('h', data)[0]
                    dBZ[i][j][k] = data_float

        # NOTE: Buo Fu mentioned that we need to divide by 10
        return dBZ


class RawRainData:
    def __init__(self, fpath: str):
        """
        Example fpath: 'QPESUMS_rain_2018/201809/20180908_1300.nc'
        """
        self._fpath = fpath

    def datetime(self):
        assert self._fpath[-3:] == '.nc', self._fpath
        dt_str = os.path.basename(self._fpath)[:-3]
        return datetime.strptime(dt_str, '%Y%m%d_%H%M')

    def load(self):
        assert os.path.exists(self._fpath), f'{self._fpath} does not exist!'
        data = Dataset(self._fpath, 'r')
        RR = data.variables['qperr'][:]
        lon = data.variables['lon'][:]
        lat = data.variables['lat'][:]
        return {'rain': np.array(RR), 'lat': np.array(lat), 'lon': np.array(lon)}


class RawQpesumsData:
    def __init__(self, fpath: str):
        """
        Example fpath: '20180114_0650_f1hr.nc'
        """
        self._fpath = fpath

    def datetime(self):
        assert self._fpath[-3:] == '.nc', self._fpath
        dt_str = os.path.basename(self._fpath)[:-3]
        dt_str = '_'.join(dt_str.split('_')[:2])
        return datetime.strptime(dt_str, '%Y%m%d_%H%M')

    def load(self):
        assert os.path.exists(self._fpath), f'{self._fpath} does not exist!'
        data = Dataset(self._fpath, 'r')
        RR = data.variables['qpfrr'][:]
        return {'rain': np.array(RR), 'dt': self.datetime()}
