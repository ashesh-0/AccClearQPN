import os
import re
import shutil
from collections import OrderedDict
from datetime import datetime, timedelta

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from core.enum import DataType
from core.loss_type import BlockAggregationMode, LossType
from core.model_type import ModelType
from models.adverserial_model import AdvModel
from models.adverserial_model_constrained import AdvConsModel
from models.adverserial_model_finetuned import AdvFinetunableModel
from models.adverserial_model_with_attention import (BalAdvAttention3OptModel, BalAdvAttentionModel,
                                                     BalAdvAttentionZeroLossModel)
from models.adverserial_model_with_prior import AdvModelWPrior, BalAdvModelWPrior
from models.balanced_adverserial_model import BalAdvModel
from models.classifier.classifier_model import ClassifierModel
from models.cnn_baseline_model import BaseLineModel
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from models.model_params import forecaster_params, get_encoder_params
from models.model_with_attention import EFAttentionModel
from models.model_with_prior import ModelWithPrior
from models.simpleGRU_model_params import forecaster_params_GRU, get_encoder_params_GRU
from models.ssim_model import SSIMModel


def infer_version(log_dir, exp_name):
    """
    Infers the version which the log file will write into.
    """
    present_versions = []
    directory = os.path.join(log_dir, exp_name)
    if not os.path.exists(directory):
        return 0

    for fpath in os.listdir(directory):
        tokens = fpath.split('_')
        assert len(tokens) == 2 and tokens[0] == 'version', fpath
        present_versions.append(int(tokens[1]))

    present_versions = sorted(present_versions)
    assert set(present_versions) == set([i for i in range(len(present_versions))])
    return len(present_versions)


def parse_dict(dict_str):
    """
    , => delimiter
    : => key value separator.

    """
    tokens = dict_str.split(',')
    output = {}
    for token in tokens:
        k, v = token.split(':')
        output[k] = v
    return output


def parse_date_start(date_str):
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except Exception as e:
        print(f'Issue in {date_str}')
        raise e


def parse_date_end(date_str):
    dt = parse_date_start(date_str)
    dt += timedelta(seconds=(60 * 23 + 50) * 60)
    return dt


def save_checkpoint(state, is_best, fpath):
    torch.save(state, fpath)
    if is_best:
        dirname = os.path.dirname(fpath)
        fname = os.path.basename(fpath)
        best_fpath = os.path.join(dirname, 'model_best_' + fname)
        shutil.copyfile(fpath, best_fpath)


def checkpoint_file_prefix(start_dt, end_dt, kwargs={}):
    y = start_dt.year - 2014
    m = start_dt.month
    d = start_dt.day
    start = '{y}{m:02d}{d:02d}'.format(y=y, m=m, d=d)

    y = end_dt.year - 2014
    m = end_dt.month
    d = end_dt.day
    end = '{y}{m:02d}{d:02d}'.format(y=y, m=m, d=d)
    prefix = f'RF_{start}_{end}'
    keys = list(kwargs.keys())
    for key in keys:
        prefix += f'_{key}-{kwargs[key]}'
    return prefix


def checkpoint_parser(fpath):
    # RF_10101_31231_la-0_lsz-20_lt-0_mt-1_res-1_toff-18_v-0-_epoch=9_val_loss=0.802.ckpt
    basename = os.path.basename(fpath)
    assert basename[-5:] == '.ckpt'
    basename = basename[:-5]
    tokens = basename.split('_')
    start = str(20140000 + int(tokens[1]))
    end = str(20140000 + int(tokens[2]))
    start = datetime.strptime(start, '%Y%m%d')
    end = datetime.strptime(end, '%Y%m%d') + timedelta(seconds=(23 * 60 + 50) * 60)
    tokens = tokens[3:]
    output = {'start': start, 'end': end}
    unused_tokens = []
    for token in tokens:
        kvpair = re.split('[-=]', token)
        if len(kvpair) == 1:
            unused_tokens += kvpair
            continue
        elif len(kvpair) == 3:
            assert kvpair[2] == ''
            k, v = kvpair[:2]
            output[k] = v
        else:
            k, v = kvpair
            # Handling jst_rain-1
            if len(unused_tokens) > 0:
                k = '_'.join(unused_tokens + [k])
                unused_tokens = []
            output[k] = v

    return output


def checkpoint_file(start_dt, end_dt, directory, prefix_kwargs={}):
    return os.path.join(directory, f'{checkpoint_file_prefix(start_dt, end_dt, kwargs=prefix_kwargs)}.pth.tar')


def get_model(
    train_start,
    train_end,
    model_kwargs,
    loss_kwargs,
    data_kwargs,
    checkpoint_dir,
    log_dir,
    input_shape=None,
    data_loader_info=None,
):
    model_type = model_kwargs['model_type']
    if input_shape is None:
        assert isinstance(data_loader_info, dict)
        input_shape = data_loader_info['input_shape']

    if loss_kwargs['type'] == LossType.WeightedMAEWithBuffer:
        loss_kwargs['padding_bbox'] = data_loader_info['padding_bbox']

    if model_type == ModelType.BaselineCNN:
        print('Using Baseline model')
        nc = DataType.count(data_kwargs['data_type'])
        model = BaseLineModel(
            loss_kwargs,
            checkpoint_file_prefix(train_start,
                                   train_end,
                                   kwargs=OrderedDict({
                                       'mt': model_type,
                                       'lt': loss_kwargs['type'],
                                       'la': loss_kwargs['aggregation_mode'],
                                       'lsz': loss_kwargs['kernel_size'],
                                       'res': int(data_kwargs['residual']),
                                       'v': infer_version(log_dir, ModelType.name(model_type)),
                                   })),
            in_channels=nc,
            checkpoint_directory=checkpoint_dir,
        )
    elif model_type in [
            ModelType.GRU,
            ModelType.TrajGRU,
            ModelType.TrajGRUWithPrior,
            ModelType.TrajGRUAdverserial,
            ModelType.GRUAdverserial,
            ModelType.BalancedGRUAdverserial,
            ModelType.GRUAdverserialRadarPrior,
            ModelType.BalancedGRUAdverserialRadarPrior,
            ModelType.BalancedGRUAdverserialFinetuned,
            ModelType.BalancedGRUAdverserialAttention,
            ModelType.BalancedGRUAdverserialAttentionZeroLoss,
            ModelType.BalancedGRUAdverserialAttention3Opt,
            ModelType.BalancedGRUAdverserialConstrained,
            ModelType.GRUAttention,
            ModelType.ClassifierGRU,
            ModelType.SSIMGRUModel,
    ]:
        print(f'Using {ModelType.name(model_type)} model')
        assert DataType.count1D(data_kwargs['data_type']) == 0
        nc = DataType.count(data_kwargs['data_type'])
        if model_type in [
                ModelType.GRU, ModelType.GRUAdverserial, ModelType.BalancedGRUAdverserial,
                ModelType.GRUAdverserialRadarPrior, ModelType.BalancedGRUAdverserialRadarPrior,
                ModelType.BalancedGRUAdverserialFinetuned, ModelType.BalancedGRUAdverserialAttention,
                ModelType.BalancedGRUAdverserialAttentionZeroLoss, ModelType.BalancedGRUAdverserialAttention3Opt,
                ModelType.BalancedGRUAdverserialConstrained, ModelType.GRUAttention, ModelType.ClassifierGRU,
                ModelType.SSIMGRUModel
        ]:
            encoder_params = get_encoder_params_GRU(nc + 2 * int(data_kwargs['hourly_data']), input_shape)
            encoder = Encoder(encoder_params[0], encoder_params[1])
            forecaster = Forecaster(forecaster_params_GRU[0], forecaster_params_GRU[1], data_kwargs['target_len'])
        else:
            encoder_params = get_encoder_params(nc + 2 * int(data_kwargs['hourly_data']), input_shape)
            encoder = Encoder(encoder_params[0], encoder_params[1])
            forecaster = Forecaster(forecaster_params[0], forecaster_params[1], data_kwargs['target_len'])

        checkpoint_dict = OrderedDict({
            'mt': model_type,
            'dt': data_kwargs['data_type'],
            'lt': loss_kwargs['type'],
            'tlen': data_kwargs['target_len'],
        })
        if loss_kwargs['type'] != LossType.WeightedMAE:
            checkpoint_dict['la'] = loss_kwargs['aggregation_mode']
            checkpoint_dict['lsz'] = loss_kwargs['kernel_size']
            checkpoint_dict['res'] = int(data_kwargs['residual'])

        if data_kwargs['threshold'] != 0.5:
            checkpoint_dict['dth'] = data_kwargs['threshold']
        if data_kwargs['input_len'] != 5:
            checkpoint_dict['ilen'] = data_kwargs['input_len']

        if loss_kwargs.get('residual_loss'):
            checkpoint_dict['lres'] = int(loss_kwargs['residual_loss'])

        if loss_kwargs['w'] != 1:
            checkpoint_dict['lw'] = loss_kwargs['w']
        if 'smoothing' in loss_kwargs and loss_kwargs['smoothing'] != 0.001:
            checkpoint_dict['lsm'] = loss_kwargs['smoothing']

        if data_kwargs['hourly_data'] is True:
            checkpoint_dict['hrly'] = 1

        if data_kwargs['sampling_rate'] != 5:
            checkpoint_dict['sampl'] = data_kwargs['sampling_rate']

        if data_kwargs['target_offset'] > 0:
            checkpoint_dict['toff'] = data_kwargs['target_offset']

        if data_kwargs['prior_dtype'] != DataType.NoneAtAll:
            checkpoint_dict['ptyp'] = data_kwargs['prior_dtype']

        if data_kwargs['random_std'] > 0:
            checkpoint_dict['r_std'] = data_kwargs['random_std']

        if model_kwargs.get('adv_w', 0.1) != 0.1:
            checkpoint_dict['AdvW'] = model_kwargs['adv_w']

        if model_kwargs.get('dis_d', 5) != 5:
            checkpoint_dict['DisD'] = model_kwargs['dis_d']

        if data_kwargs.get('ith_grid', -1) != -1:
            checkpoint_dict['Igrid'] = data_kwargs['ith_grid']
            if data_kwargs.get('pad_grid', 10) != 10:
                checkpoint_dict['Pgrid'] = data_kwargs['pad_grid']

        checkpoint_dict['v'] = infer_version(log_dir, ModelType.name(model_type))
        if model_type == ModelType.TrajGRUWithPrior:
            assert data_kwargs['prior_dtype'] != DataType.NoneAtAll
            model = ModelWithPrior(
                encoder,
                forecaster,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                loss_kwargs,
                input_shape,
                data_kwargs['prior_dtype'],
                residual=data_kwargs['residual'],
                checkpoint_directory=checkpoint_dir,
            )
        elif model_type in [ModelType.TrajGRUAdverserial, ModelType.GRUAdverserial]:
            model = AdvModel(
                model_kwargs['adv_w'],
                model_kwargs['dis_d'],
                encoder,
                forecaster,
                data_kwargs['target_len'],
                loss_kwargs,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_dir,
            )
        elif model_type == ModelType.BalancedGRUAdverserial:
            model = BalAdvModel(
                model_kwargs['adv_w'],
                model_kwargs['dis_d'],
                encoder,
                forecaster,
                data_kwargs['target_len'],
                loss_kwargs,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_dir,
            )
        elif model_type == ModelType.GRUAdverserialRadarPrior:
            nc = DataType.count(data_kwargs['data_type'])
            model = AdvModelWPrior(
                model_kwargs['adv_w'],
                model_kwargs['dis_d'],
                encoder,
                forecaster,
                data_kwargs['target_len'],
                loss_kwargs,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_dir,
                in_channels=nc,
            )
        elif model_type == ModelType.BalancedGRUAdverserialRadarPrior:
            nc = DataType.count(data_kwargs['data_type'])
            model = BalAdvModelWPrior(
                model_kwargs['adv_w'],
                model_kwargs['dis_d'],
                encoder,
                forecaster,
                data_kwargs['target_len'],
                loss_kwargs,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_dir,
                in_channels=nc,
            )
        elif model_type == ModelType.BalancedGRUAdverserialFinetuned:
            model = AdvFinetunableModel(
                model_kwargs['mode'],
                model_kwargs['adv_w'],
                model_kwargs['dis_d'],
                encoder,
                forecaster,
                data_kwargs['target_len'],
                loss_kwargs,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_dir,
            )
        elif model_type == ModelType.BalancedGRUAdverserialAttention:
            assert data_kwargs['data_type'] == DataType.Radar + DataType.Rain
            model = BalAdvAttentionModel(
                model_kwargs['adv_w'],
                model_kwargs['dis_d'],
                encoder,
                forecaster,
                data_kwargs['target_len'],
                loss_kwargs,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_dir,
            )
        elif model_type == ModelType.BalancedGRUAdverserialAttentionZeroLoss:
            model = BalAdvAttentionZeroLossModel(
                model_kwargs['adv_w'],
                model_kwargs['dis_d'],
                encoder,
                forecaster,
                data_kwargs['target_len'],
                loss_kwargs,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_dir,
            )
        elif model_type == ModelType.BalancedGRUAdverserialAttention3Opt:
            model = BalAdvAttention3OptModel(
                model_kwargs['adv_w'],
                model_kwargs['dis_d'],
                encoder,
                forecaster,
                data_kwargs['target_len'],
                loss_kwargs,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_dir,
            )
        elif model_type == ModelType.BalancedGRUAdverserialConstrained:
            model = AdvConsModel(
                model_kwargs['adv_w'],
                model_kwargs['dis_d'],
                encoder,
                forecaster,
                data_kwargs['target_len'],
                loss_kwargs,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_dir,
            )
        elif model_type == ModelType.GRUAttention:
            model = EFAttentionModel(
                encoder,
                forecaster,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_directory=checkpoint_dir,
                loss_kwargs=loss_kwargs,
                residual=data_kwargs['residual'],
            )
        elif model_type == ModelType.ClassifierGRU:
            model = ClassifierModel(
                encoder,
                forecaster,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_directory=checkpoint_dir,
                loss_kwargs=loss_kwargs,
                residual=data_kwargs['residual'],
            )
        elif model_type == ModelType.SSIMGRUModel:
            if loss_kwargs['mae_w'] != 0.1:
                checkpoint_dict['lmae'] = loss_kwargs['mae_w']
            if loss_kwargs['ssim_w'] != 0.01:
                checkpoint_dict['lssim'] = loss_kwargs['ssim_w']

            model = SSIMModel(
                encoder,
                forecaster,
                data_kwargs['target_len'],
                loss_kwargs,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_dir,
            )
        else:
            if loss_kwargs.get('mae_w', 0.1) != 0.1:
                checkpoint_dict['lmae'] = loss_kwargs['mae_w']
            if loss_kwargs.get('ssim_w', 0.01) != 0.01:
                checkpoint_dict['lssim'] = loss_kwargs['ssim_w']

            model = EF(
                encoder,
                forecaster,
                checkpoint_file_prefix(train_start, train_end, kwargs=checkpoint_dict),
                checkpoint_directory=checkpoint_dir,
                loss_kwargs=loss_kwargs,
                residual=data_kwargs['residual'],
            )

    return model
