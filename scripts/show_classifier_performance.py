# from data_loader import DataLoader
import argparse
import copy
import os
import pickle
import socket
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.data import DataLoader
# from sklearn.metrics import precision_score, recall_score
# from utils.analysis_utils import compute_precision_recall
from tqdm import tqdm

from core.base_enum import DataType
from core.file_utils import RadarFileManager
from core.loss_type import BlockAggregationMode, LossType
from core.model_type import ModelType
from data_loaders.cur_value_baseline import CurrentValueBaseLine
from models.loss import WeightedMaeLoss, get_criterion
from models.simpleGRU_model_params import forecaster_params_GRU, get_encoder_params_GRU
from utils.log_utils import get_configuration
from utils.paper_utils import csv_style_print
from utils.performance_diagram import PerformanceDiagramStable
from utils.run_utils import checkpoint_parser, get_model, parse_date_end, parse_date_start, parse_dict


def get_kwargs(checkpoint_fname):
    kwargs = checkpoint_parser(checkpoint_fname)
    for k in kwargs.keys():
        if k in ['lw', 'adv_w', 'AdvW', 'auc', 'D_pos_acc', 'D_neg_acc', 'D_auc']:
            kwargs[k] = float(kwargs[k])
        elif k not in ['start', 'end', 'val_loss', 'pdsr']:
            kwargs[k] = int(kwargs[k])
    return kwargs


def main(s, e, checkpoint_model, checkpoint_classifier):
    # s = datetime(2018,8,1)
    # e = datetime(2018,8,31,23,50)
    input_shape = (540, 420)
    sampling_rate = 5
    is_test = True

    # checkpoint_classifier='/home/u4421059/checkpoints/RF_10101_31231_mt-15_dt-17_lt-12_tlen-3_la-0_lsz-5_res-0_ilen-6_sampl-20_v-15-_epoch=5_val_loss=0.058_pdsr=0.00.ckpt'
    # checkpoint_model=     '/home/u4421059/checkpoints/RF_10101_31231_mt-10_dt-17_lt-0_tlen-3_ilen-6_AdvW-0.01_v-7-_epoch=9_val_loss=0.750_pdsr=0.31_D_auc=0.62_D_pos_acc=0.47_D_neg_acc=0.68.ckpt'
    kwargs = get_kwargs(checkpoint_model)

    model_type = int(kwargs['mt'])
    target_len = int(kwargs['tlen'])
    residual = bool(kwargs.get('res', 0))
    hourly_data = bool(kwargs.get('hrly', 0))
    input_len = int(kwargs.get('ilen', 5))
    target_offset = int(kwargs.get('toff', 0))
    random_std = int(kwargs.get('r_std', 0))

    data_type = int(kwargs['dt'])

    data_kwargs = {
        'data_type': data_type,
        'residual': residual,
        'target_offset': target_offset,
        'target_len': target_len,
        'input_len': input_len,
        'hourly_data': hourly_data,
        'sampling_rate': sampling_rate,
        'prior_dtype': DataType.NoneAtAll,
        'random_std': random_std,
    }
    model_kwargs = {
        'adv_w': float(kwargs.get('adv_w', 0.1)),
        'model_type': model_type,
        'dis_d': int(kwargs.get('dis_d', 5)),
    }

    loss_kwargs = {
        'type': kwargs['lt'],
        'aggregation_mode': kwargs.get('la'),
        'kernel_size': kwargs.get('lsz'),
        'w': float(kwargs.get('lw', 1)),
        'residual_loss': None,
    }

    dataset = CurrentValueBaseLine(
        s,
        e,
        input_len,
        target_len,
        workers=0,
        target_offset=int(kwargs.get('toff', 0)),
        data_type=data_type,
        is_validation=not is_test,
        is_test=is_test,
        img_size=input_shape,
        residual=residual,
        hourly_data=hourly_data,
        sampling_rate=sampling_rate,
        pred_avg_length=1,
    )

    model = get_model(
        s,
        e,
        model_kwargs,
        loss_kwargs,
        data_kwargs,
        '',
        '',
        data_loader_info=dataset.get_info_for_model(),
    )

    checkpoint = torch.load(checkpoint_model)
    _ = model.load_state_dict(checkpoint['state_dict'])
    model = torch.nn.DataParallel(model).cuda()

    classifier_model = copy.deepcopy(
        get_model(
            s,
            e,
            {'model_type': ModelType.ClassifierGRU},
            loss_kwargs,
            data_kwargs,
            '',
            '',
            data_loader_info=dataset.get_info_for_model(),
        ))

    checkpoint2 = torch.load(checkpoint_classifier)
    _ = classifier_model.load_state_dict(checkpoint2['state_dict'])
    classifier_model = torch.nn.DataParallel(classifier_model).cuda()

    batch_size = 32
    num_workers = 4
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    criterion = get_criterion(loss_kwargs)
    criterion_base = WeightedMaeLoss()
    criterion

    loss_dict = {i: [] for i in range(target_len)}
    loss_base_dict = {i: [] for i in range(target_len)}
    loss_base_maskless_dict = {i: [] for i in range(target_len)}
    pdsr_criterias = {i: PerformanceDiagramStable() for i in range(target_len)}
    pdsr_global_criterion = PerformanceDiagramStable()

    model.eval()
    classifier_model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            inp, target, mask, baseline = batch
            inp = inp.cuda()
            target = target.cuda()
            mask = mask.cuda()
            baseline = baseline.cuda()
            baseline = baseline.permute(1, 0, 2, 3)
            baseline = baseline.repeat((3, 1, 1, 1))

            no_mask = torch.ones_like(target)

            output = model(inp)
            weights = classifier_model(inp)
            weights[2] = weights[2] / 0.71
            weights[1] = weights[1] / 0.84
            weights[weights > 1] = 1
            output = output * weights + (1 - weights) * baseline

            output[output < 0] = 0
            pdsr_global_criterion.compute(output.permute(1, 0, 2, 3), target)
            for t in range(target_len):
                # Output is different from target in terms of dimensions. 1st dim is time
                output_t = output[t:t + 1, ...].contiguous()
                target_t = target[:, t:t + 1, :, :].contiguous()
                mask_t = mask[:, t:t + 1, :, :].contiguous()
                no_mask_t = no_mask[:, t:t + 1, :, :].contiguous()

                loss_dict[t].append(criterion(output_t, target_t, mask_t).item())
                loss_base_dict[t].append(criterion_base(output_t, target_t, mask_t).item())
                loss_base_maskless_dict[t].append(criterion_base(output_t, target_t, no_mask_t).item())
                pdsr_criterias[t].compute(output_t[0], target_t[:, 0])

    print('')
    csv_style_print(checkpoint_model, kwargs, loss_base_dict, loss_base_maskless_dict, pdsr_criterias,
                    pdsr_global_criterion)


if __name__ == '__main__':

    print(socket.gethostname(), datetime.now().strftime("%y-%m-%d-%H:%M:%S"))
    print('Python version', sys.version)
    print('CUDA_HOME', CUDA_HOME)
    print('CudaToolKit Version', torch.version.cuda)
    print('torch Version', torch.__version__)
    print('torchvision Version', torchvision.__version__)

    input_shape = (540, 420)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--start', type=parse_date_start, default=datetime(2018, 1, 1))
    parser.add_argument('--end', type=parse_date_end, default=datetime(2018, 12, 31, 23, 50))
    # parser.add_argument('--loss_kwargs', type=parse_dict, default={})
    # parser.add_argument('--log_dir', type=str, default='logs')
    # parser.add_argument('--data_kwargs', type=parse_dict, default={})
    # parser.add_argument('--model_kwargs', type=parse_dict, default={})
    parser.add_argument('--model_ckpt', type=str)
    parser.add_argument('--classifier_ckpt', type=str)
    args = parser.parse_args()
    main(args.start, args.end, args.model_ckpt, args.classifier_ckpt)
