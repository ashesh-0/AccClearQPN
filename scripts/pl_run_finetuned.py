import argparse
import os
import socket
import sys
from datetime import datetime

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.cpp_extension import CUDA_HOME

from core.data_loader_type import DataLoaderType
from core.enum import DataType
from core.loss_type import BlockAggregationMode, LossType
from core.model_type import ModelType
from models.adverserial_model_finetuned import AdvMode
from pl_data_loader_module import PLDataLoader
from utils.run_utils import get_model, parse_date_end, parse_date_start, parse_dict

workers = 4

if __name__ == '__main__':
    # python scripts/pl_run.py --train_start=20150101 --train_end=20150131 --val_start=20150201 --val_end=20150331 --gpus=1 --batch_size=2 --loss_kwargs=type:1,kernel:10,aggregation_mode:1 --data_kwargs=residual:1 --precision=16 --model_type=BaselineCNN

    print(socket.gethostname(), datetime.now().strftime("%y-%m-%d-%H:%M:%S"))
    print('Python version', sys.version)
    print('CUDA_HOME', CUDA_HOME)
    print('CudaToolKit Version', torch.version.cuda)
    print('torch Version', torch.__version__)
    print('torchvision Version', torchvision.__version__)

    input_shape = (540, 420)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dloader_type', type=DataLoaderType.from_name, default=DataLoaderType.Native)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_start', type=parse_date_start, default=datetime(2015, 1, 1))
    parser.add_argument('--train_end', type=parse_date_end, default=datetime(2017, 12, 31, 23, 50))
    parser.add_argument('--val_start', type=parse_date_start, default=datetime(2018, 1, 1))
    parser.add_argument('--val_end', type=parse_date_end, default=datetime(2018, 12, 31, 23, 50))
    parser.add_argument('--loss_kwargs', type=parse_dict, default={})
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--data_kwargs', type=parse_dict, default={})
    parser.add_argument('--model_kwargs', type=parse_dict, default={})
    parser.add_argument('--init_mode', type=AdvMode.from_name, default=AdvMode.AdvDisabled)
    parser.add_argument(
        '--checkpoints_path',
        type=str,
        default=os.path.expanduser('~/checkpoints/'),
        help='Full path to the directory where model checkpoints are [to be] saved')
    parser.add_argument('--evaluate_ckp_path', type=str, default='')
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = 10

    loss_type = int(args.loss_kwargs.get('type', LossType.WeightedMAE))
    loss_aggregation_mode = int(args.loss_kwargs.get('aggregation_mode', BlockAggregationMode.MAX))
    loss_kernel_size = int(args.loss_kwargs.get('kernel_size', 5))
    residual_loss = bool(int(args.loss_kwargs.get('residual_loss', 0)))
    mixing_weight = float(args.loss_kwargs.get('w', 1))
    loss_kwargs = {
        'type': loss_type,
        'aggregation_mode': loss_aggregation_mode,
        'kernel_size': loss_kernel_size,
        'residual_loss': residual_loss,
        'w': mixing_weight,
        'smoothing': float(args.loss_kwargs.get('smoothing', 0.001)),
    }

    data_kwargs = {
        'data_type': int(args.data_kwargs.get('type', DataType.Rain)),
        'residual': bool(int(args.data_kwargs.get('residual', 0))),
        'target_offset': int(args.data_kwargs.get('target_offset', 0)),
        'target_len': int(args.data_kwargs.get('target_len', 3)),
        'input_len': int(args.data_kwargs.get('input_len', 6)),
        'hourly_data': bool(int(args.data_kwargs.get('hourly_data', 0))),
        'sampling_rate': int(args.data_kwargs.get('sampling_rate', 5)),
        'prior_dtype': int(args.data_kwargs.get('prior', DataType.NoneAtAll)),
        'random_std': int(args.data_kwargs.get('random_std', 0)),
        'ith_grid': int(args.data_kwargs.get('ith_grid', -1)),
        'pad_grid': int(args.data_kwargs.get('pad_grid', 10)),
    }
    dm = PLDataLoader(
        args.train_start,
        args.train_end,
        args.val_start,
        args.val_end,
        img_size=input_shape,
        dloader_type=args.dloader_type,
        **data_kwargs,
        batch_size=args.batch_size,
        num_workers=workers,
    )
    model_kwargs = {
        'adv_w': float(args.model_kwargs.get('adv_w', 0.1)),
        'model_type': ModelType.from_name(args.model_kwargs.get('type', 'TrajGRU')),
        'dis_d': int(args.model_kwargs.get('dis_d', 5)),
        'mode': args.init_mode,
    }

    model = get_model(
        args.train_start,
        args.train_end,
        model_kwargs,
        loss_kwargs,
        data_kwargs,
        args.checkpoints_path,
        args.log_dir,
        data_loader_info=dm.model_related_info,
    )

    logger = TestTubeLogger(save_dir='logs', name=ModelType.name(model_kwargs['model_type']))
    logger.experiment.argparse(args)
    logger.experiment.tag({'input_len': data_kwargs['input_len'], 'target_len': data_kwargs['target_len']})
    checkpoint_callback = model.get_checkpoint_callback()
    trainer = Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, logger=logger)
    trainer.fit(model, dm)
    print('')
    print('')
    if args.init_mode == AdvMode.AdvDisabled:
        print('ENABLING ADVERSERIAL TRAINING')
    else:
        print('DISABLING ADVERSERIAL TRAINING')

    model_kwargs['mode'] = 1 - args.init_mode
    model = get_model(
        args.train_start,
        args.train_end,
        model_kwargs,
        loss_kwargs,
        data_kwargs,
        args.checkpoints_path,
        args.log_dir,
        data_loader_info=dm.model_related_info,
    )

    checkpoint = torch.load(checkpoint_callback.best_model_path)
    _ = model.load_state_dict(checkpoint['state_dict'])
    trainer = Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, logger=logger)
    trainer.fit(model, dm)
