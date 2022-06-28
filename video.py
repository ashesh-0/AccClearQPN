import argparse
import os
import pickle
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from PIL import Image
from tqdm import tqdm

from core.compressed_radar_data import CompressedRadarDataV2
from core.file_utils import RadarFileManager, RainFileManager
from core.map import Taiwan
from core.raw_data import RawRainData


class CreateVideo:
    MAX_RAINFALL_LIMIT = 150
    MAX_RADAR_LIMIT = 600

    def __init__(self, dt, num_frames):
        self._dt = dt
        self._Nframes = num_frames
        self._frame_fname = '/tmp2/ashesh/tmp/sample_frame.jpg'
        self._taiwan_map_fname = '/tmp2/ashesh/tmp/taiwan_map.pkl'
        self._sample_radarfpath = '/tmp2/ashesh/precipitation/compressed_radarV2/2015/201506/20150601/MREF3D21L.20150601.0000.gz'
        self._sample_rainfpath = '/tmp2/ashesh/precipitation/rainfall_organized/2015/201506/20150601/20150601_0000.nc'

    def _rain_fpaths(self):
        paths = []
        rain = RainFileManager()
        paths.append(RainFileManager().fpath_from_dt(self._dt, self._sample_rainfpath))
        for _ in range(self._Nframes - 1):
            paths.append(rain.next(paths[-1]))
        return paths

    def _radar_fpaths(self):
        paths = []
        radar = RadarFileManager()
        paths.append(RadarFileManager().fpath_from_dt(self._dt, self._sample_radarfpath))
        for _ in range(self._Nframes - 1):
            paths.append(radar.next(paths[-1]))
        return paths

    def _assert_same_time(self, rain_fpath, radar_fpath):
        assert RainFileManager().dt_from_fpath(rain_fpath) == RadarFileManager().dt_from_fpath(radar_fpath)

    def _concat(self, im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def _create_rain_subframe(self, ax, lon2d, lat2d, rain_data):
        tw_map = Taiwan.get_map(ax)
        x, y = tw_map(lon2d, lat2d)
        levels = np.linspace(0.0, self.MAX_RAINFALL_LIMIT, 15)
        img = tw_map.contourf(x, y, rain_data, levels=levels, cmap=plt.cm.jet, extend='min')
        tw_map.colorbar(img)
        ax.set_title('Rainfall')

    def _create_radar_subframe(self, ax, lon2d, lat2d, radar_data):
        tw_map = Taiwan.get_map(ax)
        x, y = tw_map(lon2d, lat2d)
        levels = np.linspace(0.0, self.MAX_RADAR_LIMIT, 15)
        img = tw_map.contourf(x, y, radar_data, levels=levels, cmap=plt.cm.jet, extend='min')
        tw_map.colorbar(img)
        ax.set_title('Radar')

    def create_frame(self, rf_fpath, rd_fpath):
        if os.path.exists(self._frame_fname):
            os.remove(self._frame_fname)

        fig = plt.figure()
        rf_data = RawRainData(rf_fpath).load()
        rd_data = CompressedRadarDataV2(rd_fpath).load()
        rd_data[rd_data < 0] = 0
        rd_data = np.median(rd_data, axis=-1)

        rf_data['rain'][rf_data['rain'] < 0] = 0

        lon2d, lat2d = np.meshgrid(rf_data['lon'], rf_data['lat'])

        ax = fig.add_subplot(121)
        self._create_rain_subframe(ax, lon2d, lat2d, rf_data['rain'])

        ax = fig.add_subplot(122)
        self._create_radar_subframe(ax, lon2d, lat2d, rd_data)

        fig.suptitle(f"{RainFileManager().dt_from_fpath(rf_fpath).strftime('%Y%m%d-%H:%M')}")

        plt.savefig(self._frame_fname, bbox_inches='tight')
        plt.close(fig)

    def create(self, video_path='/tmp2/ashesh/tmp/rainfall_video.mp4'):
        rain_fpaths = self._rain_fpaths()
        radar_fpaths = self._radar_fpaths()
        frames = []
        for rf_fpath, rd_fpath in tqdm(list(zip(rain_fpaths, radar_fpaths))):
            self._assert_same_time(rf_fpath, rd_fpath)
            self.create_frame(rf_fpath, rd_fpath)
            frames.append(cv2.imread(self._frame_fname))

        self._create_video(frames, video_path)

    def _create_video(self, frames, video_path):

        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 4, frames[0].shape[:2][::-1])
        for frame in frames:
            out.write(frame)
        out.release()


def format_time(time_str):
    return datetime.strptime(time_str, '%Y%m%d-%H:%M')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datetime', type=format_time, help='Start time. Example: 20150601-00:15')
    parser.add_argument('frames', type=int, help='number of consequtive frames')
    args = parser.parse_args()

    vid = CreateVideo(args.datetime, args.frames)
    vid.create()
