import os
from datetime import datetime

from mpl_toolkits.basemap import Basemap

from core.constants import ROOT_DIR
from core.file_utils import RainFileManager
from core.raw_data import RawRainData


class Taiwan:
    @staticmethod
    def get_map(ax):
        img_map = Basemap(
            projection='merc',
            llcrnrlat=21,
            urcrnrlat=26,
            llcrnrlon=119,
            urcrnrlon=123,
            resolution='h',
            lat_ts=20,
            ax=ax,
        )
        img_map.drawcoastlines()
        img_map.drawmapboundary()
        img_map.drawmeridians([118, 120, 122, 124])
        img_map.drawparallels([19, 21, 23, 25, 27])
        return img_map

    @staticmethod
    def get_latlon():
        fm = RainFileManager(directory=os.path.join(ROOT_DIR, 'rain/test/'))
        random_fpath = fm.fpath_from_dt(datetime(2017, 1, 1))
        dic = RawRainData(random_fpath).load()
        lat = dic['lat']
        lon = dic['lon']
        return lat, lon
