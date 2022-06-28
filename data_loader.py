from datetime import datetime

import numpy as np
import torch.utils.data as data
from skimage.transform import resize

from altitude_data import AltitudeLoader
from core.compressed_radar_data import CompressedRadarData
from core.constants import RADAR_DIR, RAINFALL_RAW_DIR, SKIP_TIME_LIST
from core.dataset import create_dataset
from core.file_utils import RadarFileManager, RainFileManager
from core.radar_data_aggregator import CompressedAggregatedRadarData
from core.raw_data import RawRainData
from core.time_utils import TimeSteps


class DataLoader(data.Dataset):
    def __init__(
            self,
            start: datetime,
            end: datetime,
            input_len,
            target_len,
            use_aggregated_data=True,
            threshold=0.5,
            disjoint_entries=False,
    ):
        super().__init__()
        self._s = start
        self._e = end
        self._sz = self._compute_size()
        self._ilen = input_len
        self._tlen = target_len
        self._threshold = threshold
        self._dataset = create_dataset(
            self._s, self._e, self._ilen, self._tlen, rain_dir=RAINFALL_RAW_DIR, disjoint_entries=disjoint_entries)
        self._alt_loader = AltitudeLoader()
        # assert len(self._dataset) == self._sz - (self._ilen - 1) - self._tlen
        self._use_agg_data = use_aggregated_data

    def _compute_size(self):
        sz = 0
        s = self._s
        e = self._e
        while e > s:
            if s not in SKIP_TIME_LIST:
                sz += 1
            s = TimeSteps.next(s)
        assert s == e, f'End time stamp:{e} should match current {s}'
        # +1 includes self._e
        return sz + 1

    def _get_radar_data(self, fpath):
        if self._use_agg_data:
            data = CompressedAggregatedRadarData(fpath).load()
        else:
            data = CompressedRadarData(fpath).load().max(axis=-1)
        data = data.astype(np.float32)
        return data

    def __getitem__(self, index):
        inp_fpaths, tar_fpaths = self._dataset[index]
        radar = [resize(self._get_radar_data(fpath_tuple[0]), (480, 480))[None, ...] for fpath_tuple in inp_fpaths]
        radar = np.concatenate(radar, axis=0).astype(np.float32)
        rain = [resize(RawRainData(fpath_tuple[1]).load()['rain'], (480, 480))[None, ...] for fpath_tuple in inp_fpaths]
        rain = np.concatenate(rain, axis=0).astype(np.float32)
        rain[rain < 0] = 0
        radar[radar < 0] = 0

        # TODO: normalize radar and rain
        rain = rain / 50
        radar = radar / 500
        altitude = resize(self._alt_loader.get(), (480, 480))
        altitude = np.log(1 + np.repeat(altitude[None, ...], self._ilen, axis=0), dtype=np.float32)
        assert not np.isnan(altitude).any()
        inp = np.concatenate([radar[:, None, ...], rain[:, None, ...], altitude[:, None, ...]], axis=1)
        # TODO: may be predict a subset
        target = [resize(RawRainData(fpath).load()['rain'], (480, 480))[None, ...] for fpath in tar_fpaths]
        target = np.concatenate(target, axis=0).astype(np.float32)
        mask = np.zeros_like(target)
        # TODO: add a filter to just include taiwan mainland
        mask[target > self._threshold] = 1

        return inp, target, mask

    def __len__(self):
        return len(self._dataset)
