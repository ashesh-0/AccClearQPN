import argparse
import os
import pickle
import socket
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.base_enum import DataType, Enum
# from data_loaders.data_loader_all_loaded import DataLoaderAllLoaded
from data_loaders.cur_value_baseline import CurrentValueBaseLine, DataLoaderAllLoaded
from utils.prediction_utils import get_prediction, get_worstK_prediction
from utils.run_utils import checkpoint_file_prefix, checkpoint_parser, parse_date_end, parse_date_start

IMG_SIZE = (540, 420)

if __name__ == '__main__':
    print(socket.gethostname(), datetime.now().strftime("%y-%m-%d-%H:%M:%S"))
    print('Python version', sys.version)
    print('CUDA_HOME', CUDA_HOME)
    print('CudaToolKit Version', torch.version.cuda)
    print('torch Version', torch.__version__)
    print('torchvision Version', torchvision.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--start', type=parse_date_start, default=datetime(2018, 1, 1))
    parser.add_argument('--end', type=parse_date_end, default=datetime(2018, 12, 31, 23, 50))
    parser.add_argument('--checkpoint_fpath', type=str)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--initialize', default='')

    args = parser.parse_args()
    batch_size = args.batch_size
    assert args.checkpoint_fpath != '', 'Please provide a valid checkpoint path'
    ckp_kwargs = checkpoint_parser(args.checkpoint_fpath)
    data_kwargs = {
        'data_type': int(ckp_kwargs.get('dt', DataType.Radar + DataType.Rain + DataType.Altitude)),
        'residual': bool(int(ckp_kwargs.get('res', 0))),
        'target_offset': int(ckp_kwargs.get('toff', 0)),
        'target_len': int(ckp_kwargs.get('tlen', 3)),
        'input_len': int(ckp_kwargs.get('ilen', 6)),
        'hourly_data': bool(int(ckp_kwargs.get('hrly', 0))),
        'sampling_rate': int(ckp_kwargs.get('sampl', 5)),
        'prior_dtype': int(ckp_kwargs.get('ptyp', DataType.NoneAtAll)),
        'random_std': int(ckp_kwargs.get('r_std', 0)),
        'ith_grid': int(ckp_kwargs.get('Igrid', -1)),
        'pad_grid': int(ckp_kwargs.get('pad_grid', 10)),
    }
    assert data_kwargs['ith_grid'] >= 0

    loss_kwargs = {
        'type': int(ckp_kwargs['lt']),
        'aggregation_mode': int(ckp_kwargs.get('la', -1)),
        'kernel_size': int(ckp_kwargs.get('lsz', -1)),
        'w': float(ckp_kwargs.get('lw', 1)),
    }

    data = get_prediction(
        int(ckp_kwargs['mt']),
        args.start,
        args.end,
        data_kwargs,
        loss_kwargs,
        args.checkpoint_fpath,
        batch_size,
    )
    ckp_kwargs.pop('Igrid')
    fname = checkpoint_file_prefix(datetime(2015, 1, 1), datetime(2017, 12, 31), kwargs=ckp_kwargs)
    assert fname[-5:] == '.ckpt'
    fname = fname[:-5] + '.pkl'
    fname = os.path.join(args.output_dir, fname)

    # Size
    data = data.astype(np.float16)

    with open(fname, 'wb') as f:
        pickle.dump(data, f)
    print(f'[CsvPrediction]: Prediction written to {fname}')
