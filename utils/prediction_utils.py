import pickle

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.selector import Selector
from data_loaders.data_loader_all_loaded import DataLoaderAllLoaded
from models.loss import get_criterion
from utils.run_utils import get_model


def get_model_and_dataset(
        start_date,
        end_date,
        model_kwargs,
        data_kwargs,
        checkpoint_fpath,
        input_shape=(540, 420),
        is_validation=True,
        is_test=True,
):
    loss_kwargs = {'type': 0, 'aggregation_mode': 0, 'kernel_size': None, 'residual_loss': 0, 'w': None}
    model = get_model(start_date, end_date, model_kwargs, loss_kwargs, data_kwargs, '', '', input_shape=input_shape)
    checkpoint = torch.load(checkpoint_fpath)
    _ = model.load_state_dict(checkpoint['state_dict'])
    model = torch.nn.DataParallel(model).cuda()

    dataset = DataLoaderAllLoaded(
        start_date,
        end_date,
        data_kwargs['input_len'],
        data_kwargs['target_len'],
        target_offset=data_kwargs['target_offset'],
        data_type=data_kwargs['data_type'],
        hourly_data=data_kwargs['hourly_data'],
        residual=data_kwargs['residual'],
        img_size=input_shape,
        sampling_rate=data_kwargs['sampling_rate'],
        random_std=data_kwargs['random_std'],
        is_validation=is_validation,
        is_test=is_test,
        workers=0,
    )
    return (model, dataset)


def get_worstK_prediction(k,
                          start_date,
                          end_date,
                          model_kwargs,
                          data_kwargs,
                          loss_kwargs,
                          checkpoint_fpath,
                          batch_size,
                          lean=False,
                          input_shape=(540, 420),
                          num_workers=4):

    selector = Selector(k, 'max')
    model, dataset = get_model_and_dataset(
        start_date,
        end_date,
        model_kwargs,
        data_kwargs,
        checkpoint_fpath,
        input_shape=(540, 420),
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    start_index = 0
    criterion = get_criterion(loss_kwargs)
    model.eval()
    recent_target = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            if data_kwargs['residual']:
                inp, target, mask, recent_target = batch
                recent_target = np.swapaxes(recent_target.numpy(), 0, 1)
            else:
                inp, target, mask = batch

            N = target.shape[0]
            inp = inp.cuda()
            target = target.cuda()
            mask = mask.cuda()

            prediction = model(inp) + recent_target
            assert target.shape[0] == prediction.shape[1]
            loss_list = []
            for b in range(target.shape[0]):
                loss_list.append(criterion(prediction[:, b:b + 1], target[b:b + 1], mask[b:b + 1]).item())

            prediction = prediction.permute(1, 0, 2, 3)
            prediction = prediction.cpu().numpy()
            target = target.cpu().numpy()
            ts = [dataset.target_ts(k) for k in range(start_index, start_index + N)]
            if lean:
                selector.add_batch(loss_list, ts)
            else:
                selector.add_batch(loss_list, prediction, target, ts)
            start_index += N

    return selector.all()


def load_worstK_predictions(fpath):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)

    loss_list = []
    target_list = []
    prediction_list = []
    ts_list = []
    lean = len(data[0]) == 2
    target, prediction = None, None
    while len(data) > 0:
        row = Selector.pop(data)
        assert len(row) == 4 or len(row) == 2
        if lean:
            loss, ts = row
        else:
            loss, prediction, target, ts = row
        loss_list.append(loss)
        target_list.append(target)
        prediction_list.append(prediction)
        ts_list.append(ts)

    if lean:
        return {'loss': loss_list, 'prediction': None, 'target': None, 'ts': ts_list}

    return {'loss': loss_list, 'prediction': prediction_list, 'target': target_list, 'ts': ts_list}


def get_prediction(start_date,
                   end_date,
                   model_kwargs,
                   data_kwargs,
                   loss_kwargs,
                   checkpoint_fpath,
                   batch_size,
                   is_validation=True,
                   is_test=False,
                   input_shape=(540, 420),
                   num_workers=4):
    model, dataset = get_model_and_dataset(
        start_date,
        end_date,
        model_kwargs,
        data_kwargs,
        checkpoint_fpath,
        input_shape=(540, 420),
        is_validation=is_validation,
        is_test=is_test,
    )
    criterion = get_criterion(loss_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    all_prediction = []
    loss_list = []
    model.eval()
    recent_target = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            if data_kwargs['residual']:
                inp, target, mask, recent_target = batch
                recent_target = np.swapaxes(recent_target.numpy(), 0, 1)
            else:
                inp, target, mask = batch

            N = target.shape[0]
            inp = inp.cuda()
            target = target.cuda()
            mask = mask.cuda()

            prediction = model(inp) + recent_target
            assert target.shape[0] == prediction.shape[1]
            loss_list.append(N * criterion(prediction, target, mask).item())

            prediction = prediction.cpu().numpy()
            all_prediction.append(np.swapaxes(prediction, 0, 1))

    print('[Loss]', round(np.sum(loss_list) / len(dataset), 3))
    return np.concatenate(all_prediction, axis=0)
