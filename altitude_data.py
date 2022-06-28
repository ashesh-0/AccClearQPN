import struct
from bisect import bisect_left

import numpy as np

from core.constants import TERRAIN_DIR
from core.map import Taiwan


def find_closest_pos(num_list, num):
    pos = bisect_left(num_list, num)
    if pos == 0:
        return pos
    if pos == len(num_list):
        return pos - 1
    before = num_list[pos - 1]
    after = num_list[pos]
    if after - num < num - before:
        return pos
    else:
        return pos - 1


class Region:
    def __init__(self, lon, lat):
        self._lat_list = lat
        self._lon_list = lon
        self._eps = 0.02

    def inland(self, lon, lat):
        if lat < self._lat_list[0] - self._eps:
            return False
        if lat > self._lat_list[-1] + self._eps:
            return False
        if lon < self._lon_list[0] - self._eps:
            return False
        if lon > self._lon_list[-1] + self._eps:
            return False
        return True


class AltitudeLoader:
    def __init__(self, ):
        self._fpath = TERRAIN_DIR
        self._data = self._load_data()

    def _load_data(self):
        r_lat, r_lon = self._load_latlon()
        alt = self._load_altitude()
        lat, lon = Taiwan.get_latlon()
        data = np.zeros((len(lat), len(lon)))

        region = Region(r_lon, r_lat)
        for idx_lat in range(len(lat)):
            nidx_lat = find_closest_pos(r_lat, lat[idx_lat])
            for idx_lon in range(len(lon)):
                if region.inland(lon[idx_lon], lat[idx_lat]):
                    nidx_lon = find_closest_pos(r_lon, lon[idx_lon])
                    data[idx_lat, idx_lon] = alt[nidx_lat, nidx_lon]
        return data

    def _load_altitude(self):
        f1 = open(TERRAIN_DIR, 'rb')
        nx = 400
        ny = 200
        alt = np.zeros((nx, ny))

        for i in range(nx):
            for j in range(ny):
                data = f1.read(4)
                data_float = struct.unpack('f', data)[0]
                alt[i][j] = data_float
        f1.close()
        return alt

    def _load_latlon(self):
        lon = np.arange(120.0176, 122.0020877, 0.0099723)
        lat = np.arange(21.742, 25.3391845, 0.0090155)
        return (lat, lon)

    def get(self):
        return self._data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    loader = AltitudeLoader()
    plt.contourf(loader._data)
    plt.show()
