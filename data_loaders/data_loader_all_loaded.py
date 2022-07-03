from datetime import datetime, timedelta
from typing import Counter

import numpy as np
import torch.utils.data as data

#from altitude_data import AltitudeLoader
from core.compressed_rain_data import CompressedRainData
# from core.compressed_radar_data import CompressedRadarData
from core.constants import LOGALT_Q95, RADAR_Q95, RAIN_Q95, TIME_GRANULARITY_MIN
from core.dataset import load_data
from core.base_enum import DataType
#from core.map import Taiwan
from core.radar_data_aggregator import CompressedAggregatedRadarData


class CenterCropNumpy:
    def __init__(self, crop_shape):
        self._cropx, self._cropy = crop_shape

    def __call__(self, img):
        x, y = img.shape[-2:]
        startx = x // 2 - self._cropx // 2
        starty = y // 2 - self._cropy // 2
        return img[..., startx:startx + self._cropx, starty:starty + self._cropy] #[21, 540, 420]
        #return img[...,325:445, 215:335] #[21,120,120]
        #return img

def moving_average(a, n=6):
    output = np.zeros_like(a)
    for i in range(a.shape[0]):
        output[i:i + n] += a[i:i + 1]

    output[n:] = output[n:] / n
    output[:n] = output[:n] / np.arange(1, n + 1).reshape(-1, 1, 1)
    return output


class DataLoaderAllLoaded(data.Dataset):
    def __init__(self,
                 start: datetime,
                 end: datetime,
                 input_len,
                 target_len,
                 target_offset=0,
                 target_avg_length=6,
                 threshold=0.5,
                 data_type=DataType.NoneAtAll,
                 residual=False,
                 random_std=0,
                 is_train=False,
                 is_test=False,
                 is_validation=False,
                 hourly_data=False,
                 hetero_data=False,
                 img_size=None,
                 sampling_rate=None,
                 workers=8):
        super().__init__()
        self._s = start
        self._e = end
        self._ilen = input_len
        self._tlen = target_len
        self._toffset = target_offset
        self._tavg_len = target_avg_length
        self._sampling_rate = sampling_rate
        self._train = is_train
        self._random_std = random_std
        self._dtype = data_type
        if self._sampling_rate is None:
            self._sampling_rate = self._ilen

        # Predict difference in average rain rate. Difference is taken from last hour's rain rate.
        self._residual = residual
        self._threshold = threshold
        self._index_map = []
        # There is an issue in python 3.6.10 with multiprocessing. workers should therefore be set to 0.
        self._dataset = load_data(
            self._s,
            self._e,
            is_validation=is_validation,
            is_test=is_test,
            is_train=is_train,
            workers=0,
        )
        self._ccrop = CenterCropNumpy(img_size)
        self._time = sorted(list(self._dataset['radar'].keys()))
        #self._altitude = self._altitude_data()
        #self._latlon = self._ccrop(self._compute_latlon_data())
        # whether to also give last 5 hours hour averaged rain rate
        self._hourly_data = hourly_data
        self._hetero_data = hetero_data
        self._set_index_map()

        DataType.print(self._dtype, prefix=self.__class__.__name__)
        print(f'[{self.__class__.__name__}] {self._s}<->{self._e} ILen:{self._ilen} TLen:{self._tlen} '
              f'Toff:{self._toffset} TAvgLen:{self._tavg_len} Residual:{int(self._residual)} Hrly:{int(hourly_data)} '
              f'Sampl:{self._sampling_rate} RandStd:{self._random_std} Th:{self._threshold}')

    def _set_index_map(self):
        raw_idx = 0
        skip_counter = 0
        target_offset = self._ilen + self._toffset + self._tavg_len * self._tlen
        while raw_idx < len(self._time):
            if raw_idx + target_offset >= len(self._time):
                break

            if self._time[raw_idx + target_offset] - self._time[raw_idx] != timedelta(seconds=TIME_GRANULARITY_MIN *
                                                                                     target_offset * 60):
                skip_counter += 1
            else:
                self._index_map.append(raw_idx)

            raw_idx += 1
        print(f'[{self.__class__.__name__}] Size:{len(self._index_map)} Skipped:{skip_counter}')

    def _compute_latlon_data(self):
        lat, lon = Taiwan.get_latlon()
        lon2D, lat2D = np.meshgrid(lon, lat)
        lon2D = (lon2D - np.mean(lon2D)) / np.std(lon2D)
        lat2D = (lat2D - np.mean(lat2D)) / np.std(lat2D)
        output = np.concatenate([lat2D[None, ...], lon2D[None, ...]], axis=0)
        output = np.repeat(output[None, ...], self._ilen, axis=0)
        return output.astype(np.float32)

    def _altitude_data(self):
        alt_loader = AltitudeLoader()
        raw_alt = alt_loader.get()
        raw_alt = self._ccrop(raw_alt)
        altitude = np.log(1 + np.repeat(raw_alt[None, ...], self._ilen, axis=0), dtype=np.float32)
        altitude = altitude / LOGALT_Q95

        assert not np.isnan(altitude).any()
        return altitude

    def _get_raw_radar_data(self, index):
        key = self._time[index]
        raw_data = self._dataset['radar'][key]
        #2D radar
        return CompressedAggregatedRadarData.load_from_raw(raw_data)
        #3D radar for 21lv
        #return CompressedRadarData.load_from_raw(raw_data).transpose(2,0,1) # [NZ, NX, NY]
        #3D radar for 5lv
        #return CompressedRadarData.load_from_raw(raw_data).transpose(2,0,1)[0:10:2] # [NZ, NX, NY]

    def _input_end(self, index):
        return index + self._ilen

    def _input_range(self, index):
        return range(index, self._input_end(index))

    def _get_radar_data(self, index):
        return self._get_radar_data_from_range(self._input_range(index))

    def _get_radar_data_from_range(self, index_range):
        radar = [self._ccrop(self._get_raw_radar_data(idx))[None, ...] for idx in index_range]
        radar = np.concatenate(radar, axis=0).astype(np.float32) # [6, 120, 120]
        radar[radar < 0] = 0
        radar = radar / RADAR_Q95
        return radar

    def _get_raw_rain_data(self, index):
        key = self._time[index]
        raw_data = self._dataset['rain'][key]
        data = CompressedRainData.load_from_raw(raw_data)
        # (561,441)
        return data

    def _get_past_hourly_data(self, data_from_range_function, index):
        end_idx = self._input_end(index)
        output = []
        period = 6
        last_hour_data = None
        for _ in range(self._ilen):
            if end_idx <= 0:
                output.append(last_hour_data)
                continue

            start_idx = max(0, end_idx - period)
            data = data_from_range_function(range(start_idx, end_idx))
            last_hour_data = np.mean(data, axis=0, keepdims=True)
            output.append(last_hour_data)
            end_idx -= period

        return np.concatenate(output, axis=0)

    def _get_rain_data_from_range(self, index_range):
        rain = [self._ccrop(self._get_raw_rain_data(idx))[None, ...] for idx in index_range]
        rain = np.concatenate(rain, axis=0).astype(np.float32)
        rain[rain < 0] = 0
        rain = rain / RAIN_Q95
        return rain

    def _get_rain_data(self, index):
        # NOTE: average of last 5 frames is the rain data.
        return self._get_rain_data_from_range(self._input_range(index))

        
    def _get_era5_data(self, index):
        index = [index + 6, index + 12, index + 18]
        keys = self._half_hour(index)
        era5 = [self._dataset['era5'][key][None,...] for key in keys]
        era5 = np.concatenate(era5, axis=0).astype(np.float32) # [3, 4, 20, 29, 23]
        return era5

    def _half_hour(self, idx):
        t = [self._time[i] for i in idx]
        counter = 0
        while counter < len(idx):
            if t[counter].minute >= 30:
                delta = 60 - t[counter].minute
                t[counter] = (lambda x: x + timedelta(minutes=delta))(t[counter])
            counter += 1
        return t

    def _get_most_recent_target(self, index, tavg_len=None):
        """
        Returns the averge rainfall which has happened in last self._tlen*10 minutes.
        """
        if tavg_len is None:
            tavg_len = self._tavg_len

        target_end_idx = self._input_end(index)
        target_start_idx = target_end_idx - tavg_len
        target_start_idx = max(0, target_start_idx)

        temp_data = [
            self._ccrop(self._get_raw_rain_data(idx))[None, ...] for idx in range(target_start_idx, target_end_idx)
        ]
        # print('Recent Target', list(range(target_start_idx, target_end_idx)))
        target = np.concatenate(temp_data, axis=0).astype(np.float32)
        target[target < 0] = 0
        assert target.shape[0] == tavg_len or target_start_idx == 0
        return target.mean(axis=0, keepdims=True)

    def _get_avg_target(self, index):
        target_start_idx = self._input_end(index) + self._toffset
        target_end_idx = target_start_idx + self._tlen * self._tavg_len
        temp_data = [
            self._ccrop(self._get_raw_rain_data(idx))[None, ...] for idx in range(target_start_idx, target_end_idx)
        ]
        temp_data = np.concatenate(temp_data, axis=0).astype(np.float32)

        target = []
        for i in range(self._tavg_len - 1, len(temp_data), self._tavg_len):
            target.append(temp_data[i - (self._tavg_len - 1):i + 1].mean(axis=0, keepdims=True))
        assert len(target) == self._tlen
        target = np.concatenate(target, axis=0).astype(np.float32)
        target[target < 0] = 0
#         target /= RAIN_Q95
        return target

    def __len__(self):
        return len(self._index_map) // self._sampling_rate

    def _get_internal_index(self, input_index):
        # If total we have 500 entries. Then with _ilen being 5, input index will vary in [0,100]
        input_index = input_index * self._sampling_rate
        index = self._index_map[input_index]
        return index

    def _get_past_hourly_rain_data(self, index):
        return self._get_past_hourly_data(self._get_rain_data_from_range, index)

    def _get_past_hourly_radar_data(self, index):
        return self._get_past_hourly_data(self._get_radar_data_from_range, index)

    def _random_perturbation(self, target):
        assert self._train is True

        def _rhs_idx(eps, N):
            return (eps, N) if eps > 0 else (0, N + eps)

        def _lhs_idx(eps, N):
            lidx = abs(eps) // 2
            ridx = N - (abs(eps) - lidx)
            return (lidx, ridx)

        Nx, Ny = target.shape[-2:]
        eps_x = int(np.random.normal(scale=self._random_std))
        eps_y = int(np.random.normal(scale=self._random_std))
        d_lx, d_rx = _rhs_idx(eps_x, Nx)
        d_ly, d_ry = _rhs_idx(eps_y, Ny)

        lx, rx = _lhs_idx(eps_x, Nx)
        ly, ry = _lhs_idx(eps_y, Ny)

        target[:, lx:rx, ly:ry] = target[:, d_lx:d_rx, d_ly:d_ry]
        target[:, :lx] = 0
        target[:, rx:] = 0
        target[:, :, :ly] = 0
        target[:, :, ry:] = 0
        return target

    def initial_time(self, index):
        index = self._get_internal_index(index)
        index = index + 5 + self._toffset
        return self._time[index]

    def get_index_from_target_ts(self, ts):
        if ts in self._time:
            internal_index = self._time.index(ts)
            internal_index -= (self._toffset + self._ilen)
            index = self._index_map.index(internal_index)
            assert index % self._sampling_rate == 0
            return index // self._sampling_rate

        return None

    def get_info_for_model(self):
        return {'input_shape': self[0][0].shape[2:]}

    def __getitem__(self, input_index):
        index = self._get_internal_index(input_index)
        input_data = []

        if self._dtype & DataType.Radar:
            #input_data.append(self._get_radar_data(index)) #numpy array [6, 21, 120, 120]
            input_data.append(self._get_radar_data(index)[:, None, ...]) #numpy array [6, 1, 120, 120]

        if self._dtype & DataType.Rain:
            input_data.append(self._get_rain_data(index)[:, None, ...])

        if self._dtype & DataType.Altitude:
            input_data.append(self._altitude[:, None, ...])

        if self._hourly_data:
            input_data.append(self._get_past_hourly_rain_data(index)[:, None, ...])
            input_data.append(self._get_past_hourly_radar_data(index)[:, None, ...])

        if self._dtype & DataType.Latitude:
            input_data.append(self._latlon[:, :1])

        if self._dtype & DataType.Longitude:
            input_data.append(self._latlon[:, 1:])

        if len(input_data) > 1:
            inp = np.concatenate(input_data, axis=1)
        else:
            inp = input_data[0]

        target = self._get_avg_target(index)
        if self._train and self._random_std > 0:
            target = self._random_perturbation(target)

        # NOTE: mask needs to be created before tackling the residual option. We wouldn't know which entries are relevant
        # in the residual space.
        mask = np.zeros_like(target)
        mask[target > self._threshold] = 1
        
        if self._hetero_data:
            _hetero_data = self._get_era5_data(index) # [3, 4, 20, 29, 23]
            return inp, target, mask, _hetero_data

        assert target.max() < 1000
        # NOTE: There can be a situation where previous data is absent when self._tlen + self._tavg_len -1 > self._ilen
        if self._residual:
            assert self._random_std == 0
            recent_target = self._get_most_recent_target(index)
            target -= recent_target
            return inp, target, mask, recent_target

        return inp, target, mask
