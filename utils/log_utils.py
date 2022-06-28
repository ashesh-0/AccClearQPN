import os

import pandas as pd

from core.model_type import ModelType
from utils.run_utils import checkpoint_parser


def get_configuration(
        model_type=None,
        version=None,
        checkpoint_fpath=None,
        log_directory='/tmp2/ashesh/mount/rainfall_prediction/logs',
):
    assert (model_type is not None and version is not None) ^ (checkpoint_fpath is not None)

    if checkpoint_fpath is not None:
        # RF_10101_31231_mt-2_tlen-3_lt-0_la-0_lsz-5_res-0_ilen-6_ptyp-1_v-13-_epoch=3_val_loss=0.761.ckpt
        dic = checkpoint_parser(checkpoint_fpath)
        version = int(dic['v'])
        model_type = int(dic['mt'])

    mdata_dir = os.path.join(log_directory, ModelType.name(model_type), f'version_{version}')
    return pd.read_csv(os.path.join(mdata_dir, 'meta_tags.csv')).set_index('key')['value']
