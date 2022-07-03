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
from core.model_type import ModelType
# from data_loaders.data_loader_all_loaded import DataLoaderAllLoaded
from data_loaders.cur_value_baseline import CurrentValueBaseLine, DataLoaderAllLoaded
from utils.prediction_utils import get_prediction, get_worstK_prediction
from utils.run_utils import checkpoint_parser, parse_date_end, parse_date_start


class PredictionType(Enum):
    Prediction = 0
    Target = 1
    Baseline = 2
    WorstKPrediction = 3
    WorstKPredictionLean = 4


IMG_SIZE = (540, 420)


def get_baseline(start_date,
                 end_date,
                 pred_avg_length=1,
                 sampling_rate=6,
                 is_validation=True,
                 is_test=False,
                 num_workers=4):
    # All other parameters except start and end date don't have any effect on baseline.
    dataset = CurrentValueBaseLine(
        start_date,
        end_date,
        5,
        3,
        data_type=DataType.Rain,
        img_size=IMG_SIZE,
        pred_avg_length=pred_avg_length,
        sampling_rate=sampling_rate,
        is_validation=is_validation,
        is_test=is_test,
        workers=num_workers,
    )

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    baseline_all = []
    for batch in tqdm(loader):
        inp, target, mask, baseline = batch
        baseline_all.append(baseline)
    return np.concatenate(baseline_all, axis=0)


def get_target(start_date,
               end_date,
               input_len,
               target_len,
               target_offset,
               sampling_rate,
               batch_size,
               is_validation=True,
               is_test=False,
               num_workers=4):
    dataset = DataLoaderAllLoaded(
        start_date,
        end_date,
        input_len,
        target_len,
        data_type=DataType.Rain,  # does not matter
        target_offset=target_offset,
        img_size=IMG_SIZE,
        residual=False,
        sampling_rate=sampling_rate,
        is_validation=is_validation,
        is_test=is_test,
        workers=num_workers)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    target_all = []
    ts_all = []
    idx = 0
    for batch in tqdm(loader):
        inp, target, mask = batch
        N = inp.shape[0]
        ts_all += [dataset.target_ts(i) for i in range(idx, idx + N)]
        idx += N
        target_all.append(target)

    output = {'data': np.concatenate(target_all, axis=0).astype(np.float16), 'ts': ts_all}

    assert output['data'].shape[0] == len(output['ts'])
    print(len(ts_all), 'entries added')
    return output


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
    parser.add_argument('--prediction_type', type=PredictionType.from_name, default=PredictionType.Prediction)
    parser.add_argument('--checkpoint_fpath', type=str)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--worst_k', type=int, default=100)
    parser.add_argument('--test_data', action='store_true')
    sampling_rate = 6
    args = parser.parse_args()
    batch_size = args.batch_size
    is_test = args.test_data
    is_validation = not is_test
    if args.prediction_type == PredictionType.Baseline:
        data = get_baseline(
            args.start, args.end, sampling_rate=sampling_rate, is_validation=is_validation, is_test=is_test)
        # Size
        data = data.astype(np.float16)

        fname = f'baseline_1hr_{args.start.strftime("%Y%m%d")}_{args.end.strftime("%Y%m%d")}_TEST-{int(is_test)}.pkl'
    else:

        assert args.checkpoint_fpath != '', 'Please provide a valid checkpoint path'
        ckp_kwargs = checkpoint_parser(args.checkpoint_fpath)
        data_kwargs = {
            'data_type': int(ckp_kwargs.get('dt', DataType.Radar + DataType.Rain + DataType.Altitude)),
            'residual': bool(int(ckp_kwargs.get('res', 0))),
            'target_offset': int(ckp_kwargs.get('toff', 0)),
            'target_len': int(ckp_kwargs.get('tlen', 3)),
            'input_len': int(ckp_kwargs.get('ilen', 6)),
            'hourly_data': bool(int(ckp_kwargs.get('hrly', 0))),
            # 'sampling_rate': int(ckp_kwargs.get('sampl', 5)),
            'prior_dtype': int(ckp_kwargs.get('ptyp', DataType.NoneAtAll)),
            'random_std': int(ckp_kwargs.get('r_std', 0)),
        }
        data_kwargs['sampling_rate'] = sampling_rate
        loss_kwargs = {
            'type': int(ckp_kwargs['lt']),
            'aggregation_mode': int(ckp_kwargs.get('la', -1)),
            'kernel_size': int(ckp_kwargs.get('lsz', -1)),
            'w': float(ckp_kwargs.get('lw', 1)),
        }
        model_kwargs = {
            'adv_w': float(ckp_kwargs.get('AdvW', 0.1)),
            'model_type': int(ckp_kwargs.get('mt', ModelType.TrajGRU)),
            'dis_d': int(ckp_kwargs.get('DisD', 5)),
        }

        if args.prediction_type == PredictionType.Target:
            data = get_target(
                args.start,
                args.end,
                data_kwargs['input_len'],
                data_kwargs['target_len'],
                data_kwargs['target_offset'],
                data_kwargs['sampling_rate'],
                batch_size,
                is_validation=is_validation,
                is_test=is_test)

            fname = (f"target_tlen:{data_kwargs['target_len']}_ilen:{data_kwargs['input_len']}_"
                     f"{args.start.strftime('%Y%m%d')}_{args.end.strftime('%Y%m%d')}_TEST-{int(is_test)}.pkl")
        elif args.prediction_type == PredictionType.Prediction:
            data = get_prediction(
                args.start,
                args.end,
                model_kwargs,
                data_kwargs,
                loss_kwargs,
                args.checkpoint_fpath,
                batch_size,
                is_validation=is_validation,
                is_test=is_test,
            )
            # Size
            data = data.astype(np.float16)

            fname = os.path.basename(args.checkpoint_fpath)
            assert fname[-5:] == '.ckpt'
            fname = fname[:-5] + f'_TEST-{int(is_test)}.pkl'
        elif args.prediction_type in [PredictionType.WorstKPrediction, PredictionType.WorstKPredictionLean]:
            print('Loss', loss_kwargs)
            data = get_worstK_prediction(
                args.worst_k,
                int(ckp_kwargs['mt']),
                args.start,
                args.end,
                data_kwargs,
                loss_kwargs,
                args.checkpoint_fpath,
                batch_size,
                lean=args.prediction_type == PredictionType.WorstKPredictionLean,
            )
            print('Average loss', np.mean([row[0] for row in data]))
            fname = os.path.basename(args.checkpoint_fpath)
            assert fname[-5:] == '.ckpt'
            s = args.start.strftime("%Y%m%d")
            e = args.end.strftime("%Y%m%d")
            fname = f'Worst{args.worst_k}_{s}_{e}_' + fname[:-5] + '.pkl'

    fname = os.path.join(args.output_dir, fname)
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
    print(f'[CsvPrediction]: Prediction written to {fname}')
