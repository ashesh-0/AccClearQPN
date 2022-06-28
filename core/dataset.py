import calendar
import os
import pickle
from datetime import datetime, timedelta
from multiprocessing import Pool

from tqdm import tqdm

from core.compressed_radar_data import CompressedRadarData
from core.compressed_rain_data import CompressedRainData
from core.constants import DATA_PKL_DIR, RADAR_DIR, RAINFALL_DIR, SKIP_TIME_LIST
from core.file_utils import RadarFileManager, RainFileManager
from core.time_utils import TimeSteps


def create_dataset(
        start_dt,
        end_dt,
        input_len,
        target_len,
        radar_dir=RADAR_DIR,
        rain_dir=RAINFALL_DIR,
        disjoint_entries=False,
):
    radar_fm = RadarFileManager(radar_dir)
    rain_fm = RainFileManager(rain_dir)
    cur_dt = start_dt
    dt_list = [cur_dt]
    while end_dt > cur_dt:
        cur_dt = TimeSteps.next(cur_dt)
        if cur_dt in SKIP_TIME_LIST:
            continue
        dt_list.append(cur_dt)

    dataset = []
    N = len(dt_list) - (input_len - 1) - target_len
    stepsize = 1
    if disjoint_entries:
        stepsize = input_len

    for i in range(0, N, stepsize):
        inp_radar_fpaths = [radar_fm.fpath_from_dt(dt) for dt in dt_list[i:i + input_len]]
        inp_rain_fpaths = [rain_fm.fpath_from_dt(dt) for dt in dt_list[i:i + input_len]]
        inp = list(zip(inp_radar_fpaths, inp_rain_fpaths))

        target = [rain_fm.fpath_from_dt(dt) for dt in dt_list[i + input_len:i + input_len + target_len]]
        dataset.append((inp, target))

    print(f"Created Dataset:{start_dt.strftime('%Y%m%d_%H%M')}-{end_dt.strftime('%Y%m%d_%H%M')}, "
          f" Disjoint:{int(disjoint_entries)} InpLen:{input_len} TarLen:{target_len} {len(dataset)/1000}K points")

    return dataset


def keep_first_half(dic):
    """
    Keep data only for first 15 days of each month.
    """
    output = {k: v for k, v in dic.items() if k.day <= 15}
    return output


def keep_later_half(dic):
    """
    Keep only that data which is not being kept in keep_first_half()
    """
    fh_dict = keep_first_half(dic)
    return {k: v for k, v in dic.items() if k not in fh_dict}


def load_data(start_dt, end_dt, is_validation=False, is_test=False, is_train=False, workers=8):
    assert int(is_validation) + int(is_test) + int(is_train) == 1, 'Data must be either train,test or validation'
    dtype_str = ['Train'] * int(is_train) + ['Validation'] * int(is_validation) + ['Test'] * int(is_test)
    print(f'[Loading {dtype_str[0]} Data] {start_dt} {end_dt}')

    arguements = []
    assert start_dt < end_dt
    cur_dt = start_dt
    while cur_dt < end_dt:
        last_day_month = calendar.monthrange(cur_dt.year, cur_dt.month)[1]
        # NOTE: 23:50 is the last event. this may change if we change the granularity
        offset_min = 23 * 60 + 50 - (cur_dt.hour * 60 + cur_dt.minute)
        cur_end_dt = min(end_dt, cur_dt + timedelta(days=last_day_month - cur_dt.day, seconds=60 * offset_min))
        arguements.append((cur_dt, cur_end_dt))
        cur_dt = TimeSteps.next(cur_end_dt)

    print('')
    data_dicts = []
    if workers > 0:
        with Pool(processes=workers) as pool:
            with tqdm(total=len(arguements)) as pbar:
                for i, data_dict in enumerate(pool.imap_unordered(_load_data, arguements)):
                    if data_dict.get('fpath'):
                        pbar.set_description(f"Loaded from {data_dict['fpath']}")
                    pbar.update()
                    data_dicts.append(data_dict)
    else:
        for args in tqdm(arguements):
            data_dicts.append(_load_data(args))

    radar_dict = {}
    rain_dict = {}
    for d in data_dicts:
        if is_test:
            d = {'radar': keep_later_half(d['radar']), 'rain': keep_later_half(d['rain'])}
        elif is_validation:
            d = {'radar': keep_first_half(d['radar']), 'rain': keep_first_half(d['rain'])}

        radar_dict = {**radar_dict, **d['radar']}
        rain_dict = {**rain_dict, **d['rain']}

    return {'rain': rain_dict, 'radar': radar_dict}


def _load_data(args):
    start_dt, end_dt = args
    fname = os.path.join(DATA_PKL_DIR, 'AllDataDict_{start}_{end}.pkl')
    fname = fname.format(
        start=start_dt.strftime('%Y%m%d-%H%M'),
        end=end_dt.strftime('%Y%m%d-%H%M'),
    )
    if os.path.exists(fname):
        # print('Loading data from', fname)
        with open(fname, 'rb') as f:
            output = pickle.load(f)
        output['fpath'] = fname
        return output

    radar_fm = RadarFileManager(RADAR_DIR)
    rain_fm = RainFileManager(RAINFALL_DIR, compressed=True)
    cur_dt = start_dt
    dt_list = []

    while end_dt >= cur_dt:
        if cur_dt not in SKIP_TIME_LIST:
            dt_list.append(cur_dt)

        cur_dt = TimeSteps.next(cur_dt)

    radar_data = {}
    rain_data = {}
    for dt in dt_list:
        radar_data[dt] = CompressedRadarData(radar_fm.fpath_from_dt(dt)).load_raw()
        rain_data[dt] = CompressedRainData(rain_fm.fpath_from_dt(dt)).load_raw(can_raise_error=False)

    output = {'radar': radar_data, 'rain': rain_data}
    with open(fname, 'wb') as f:
        pickle.dump(output, f)

    return output


if __name__ == '__main__':
    start_dt = datetime(2015, 1, 1)
    end_dt = datetime(2018, 12, 31, 23, 50)
    load_data(start_dt, end_dt)

    # 9: 1.28677
    # 13: 1.25356
