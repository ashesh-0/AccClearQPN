import os
from datetime import datetime, timedelta

ROOT_DIR = os.environ['ROOT_DATA_DIR']
# If .pkl files exist, one only needs to set this directory appropriately to
# run the training/evaluation with default configuration.
DATA_PKL_DIR = f'{ROOT_DIR}/PKL_2Drd_rain10m/'


def whole_day(year, month, day):
    ts = []
    dt = datetime(year, month, day)
    while dt.day == day:
        ts.append(dt)
        dt += timedelta(seconds=60 * TIME_GRANULARITY_MIN)
    return ts

def whole_hour(year, month, day, hour):
    ts = []
    dt = datetime(year, month, day, hour)
    while dt.hour == hour:
        ts.append(dt)
        dt += timedelta(minutes=10)
    return ts

NX = 561
NY = 441
DBZ_Z = 21
TIME_GRANULARITY_MIN = 10
# If one has the rain and radar data directories and has not yet generated the .pkl files,
# then one needs to set the following two directories appropriately.
RAINFALL_DIR =None# f'{ROOT_DIR}/rain_compressed/'
RADAR_DIR = None #f'{ROOT_DIR}/compressed_radarV2/'


ERA_DIR = None #f'{ROOT_DIR}/era5/'
ERA_QV_DIR = None #f'{ROOT_DIR}/era5_qv/'

# _dir_path = os.path.dirname(os.path.realpath(__file__))
TERRAIN_DIR = None #f'{os.path.dirname(_dir_path)}/Terrain.dat'

INPUT_LEN = 5
BALANCING_WEIGHTS = [1, 2, 5, 10, 30]

RAIN_Q95 = 10
RADAR_Q95 = 35
LOGALT_Q95 = 5.7

SKIP_TIME_LIST = []
