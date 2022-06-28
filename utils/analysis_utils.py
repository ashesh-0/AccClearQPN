import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.metrics.functional.classification import auroc
from torch.utils.data import DataLoader
from tqdm import tqdm


class UnderOverPredictionAnalysis:
    """
    Pixelwise analysis of how much is under prediction, how much is over prediction.
    """
    def __init__(self):
        self._Usz = None
        self._Osz = None
        self._Uerr = None
        self._Oerr = None
        self._N = 0
        self._target = None

    def update(self, output, target, mask):
        assert output.shape == target.shape
        assert target.shape == mask.shape
        if self._Usz is None:
            self._Usz = torch.zeros_like(target[0]).double()
            self._Osz = torch.zeros_like(target[0]).double()
            self._Uerr = torch.zeros_like(target[0]).double()
            self._Oerr = torch.zeros_like(target[0]).double()
            self._target = torch.zeros_like(target[0]).double()

        self._target += torch.sum(target, dim=0).double()

        diff = output - target
        under_mask = output <= target
        over_mask = output > target
        under_mask = under_mask * mask
        over_mask = over_mask * mask
        self._Uerr = torch.sum(-diff * under_mask, dim=0).double()
        self._Oerr = torch.sum(diff * over_mask, dim=0).double()

        self._Usz += torch.sum(under_mask, dim=0).double()
        self._Osz += torch.sum(over_mask, dim=0).double()
        self._N += target.shape[0]

    def get(self):
        # return {
        #     'UnderPredictionError': self._Uerr / self._Usz,
        #     'OverPredictionError': self._Oerr / self._Osz,
        #     'UnderPredictionFraction': self._Usz / self._N,
        #     'OverPredictionFraction': self._Osz / self._N,
        #     'target': self._target / self._N,
        #     'N': self._N,
        # }
        return {
            'UnderPredictionError': self._Uerr,
            'OverPredictionError': self._Oerr,
            'UnderPredictionFraction': self._Usz,
            'OverPredictionFraction': self._Osz,
            'target': self._target,
            'N': self._N,
        }

    def plot(self, vmax=None, sz=(3, 2)):
        import seaborn as sns
        import matplotlib.pyplot as plt
        nrows = 5
        ncols = 3
        if not isinstance(vmax, list):
            vmax = [vmax] * nrows

        _, ax = plt.subplots(figsize=(ncols * sz[0], nrows * sz[1]), ncols=ncols, nrows=nrows)
        hdata = self.get()
        print('N', hdata['N'])

        avg_t = hdata['target'].cpu().numpy()
        sns.heatmap(avg_t[0], ax=ax[0, 0], vmax=vmax[0])
        sns.heatmap(avg_t[1], ax=ax[0, 1], vmax=vmax[0])
        sns.heatmap(avg_t[2], ax=ax[0, 2], vmax=vmax[0])
        ax[0, 0].set_ylabel('Target')

        under_prediction = hdata['UnderPredictionError'].cpu().numpy()
        sns.heatmap(under_prediction[0], ax=ax[1, 0], vmax=vmax[1])
        sns.heatmap(under_prediction[1], ax=ax[1, 1], vmax=vmax[1])
        sns.heatmap(under_prediction[2], ax=ax[1, 2], vmax=vmax[1])
        ax[1, 0].set_ylabel('UnderPError')

        over_prediction = hdata['OverPredictionError'].cpu().numpy()
        sns.heatmap(over_prediction[0], ax=ax[2, 0], vmax=vmax[2])
        sns.heatmap(over_prediction[1], ax=ax[2, 1], vmax=vmax[2])
        sns.heatmap(over_prediction[2], ax=ax[2, 2], vmax=vmax[2])
        ax[2, 0].set_ylabel('OverPError')

        under_prediction_fraction = hdata['UnderPredictionFraction'].cpu().numpy()
        sns.heatmap(under_prediction_fraction[0], ax=ax[3, 0], vmax=vmax[3])
        sns.heatmap(under_prediction_fraction[1], ax=ax[3, 1], vmax=vmax[3])
        sns.heatmap(under_prediction_fraction[2], ax=ax[3, 2], vmax=vmax[3])
        ax[3, 0].set_ylabel('UnderPFraction')

        over_prediction_fraction = hdata['OverPredictionFraction'].cpu().numpy()
        sns.heatmap(over_prediction_fraction[0], ax=ax[4, 0], vmax=vmax[4])
        sns.heatmap(over_prediction_fraction[1], ax=ax[4, 1], vmax=vmax[4])
        sns.heatmap(over_prediction_fraction[2], ax=ax[4, 2], vmax=vmax[4])
        ax[4, 0].set_ylabel('OverPFraction')

        for row_idx in range(5):
            for col_idx in range(3):
                ax[row_idx, col_idx].tick_params(left=False, right=False, top=False, bottom=False)
                ax[row_idx, col_idx].get_xaxis().set_visible(False)  #axis('off')
                ax[row_idx, col_idx].get_yaxis().set_visible(col_idx == 0)

        ax[0, 0].set_title('First Hour')
        ax[0, 1].set_title('Second Hour')
        ax[0, 2].set_title('Third Hour')


class SpatialHeatmap:
    def __init__(self):
        self._target = None
        self._masked_map = None
        self._unmasked_map = None
        self._N = 0

    def update(self, output, target, mask):
        assert output.shape == target.shape
        assert target.shape == mask.shape
        diff = torch.abs(output - target)
        if self._masked_map is None:
            self._masked_map = torch.sum(diff * mask, dim=0).double()
            self._unmasked_map = torch.sum(diff, dim=0).double()
            self._target = torch.sum(target, dim=0).double()
            self._N = target.shape[0]
        else:
            self._masked_map += torch.sum(diff * mask, dim=0).double()
            self._unmasked_map += torch.sum(diff, dim=0).double()
            self._target += torch.sum(target, dim=0).double()
            self._N += target.shape[0]

    def get(self):
        return {
            'masked': self._masked_map / self._N,
            'unmasked': self._unmasked_map / self._N,
            'target': self._target / self._N,
            'N': self._N
        }

    def plot(self):
        import seaborn as sns
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(4 * 3, 3 * 3), ncols=3, nrows=3)
        hdata = self.get()
        print('N', hdata['N'])
        avg_t = hdata['target'].cpu().numpy()
        avg_masked = hdata['masked'].cpu().numpy()
        avg_unmasked = hdata['unmasked'].cpu().numpy()
        sns.heatmap(avg_t[0], ax=ax[0, 0])
        sns.heatmap(avg_t[1], ax=ax[0, 1])
        sns.heatmap(avg_t[2], ax=ax[0, 2])

        sns.heatmap(avg_masked[0], ax=ax[1, 0])
        sns.heatmap(avg_masked[1], ax=ax[1, 1])
        sns.heatmap(avg_masked[2], ax=ax[1, 2])

        sns.heatmap(avg_unmasked[0], ax=ax[2, 0])
        sns.heatmap(avg_unmasked[1], ax=ax[2, 1])
        sns.heatmap(avg_unmasked[2], ax=ax[2, 2])

        for row_idx in range(3):
            for col_idx in range(3):
                ax[row_idx, col_idx].tick_params(left=False, right=False, top=False, bottom=False)
                ax[row_idx, col_idx].get_xaxis().set_visible(False)  #axis('off')
                ax[row_idx, col_idx].get_yaxis().set_visible(col_idx == 0)

        ax[0, 0].set_title('First Hour')
        ax[0, 1].set_title('Second Hour')
        ax[0, 2].set_title('Third Hour')

        ax[0, 0].set_ylabel('Target')
        ax[1, 0].set_ylabel('Masked')
        ax[2, 0].set_ylabel('Unmasked')


def _compute_custom_metric(prediction, target, func):
    # batch,pixels
    assert len(prediction.shape) == 2
    tp = torch.logical_and(target, target == prediction)
    fp = torch.logical_and(~target, prediction)
    tn = torch.logical_and(~target, target == prediction)
    fn = torch.logical_and(target, ~prediction)
    # 1 value per entry in batch
    tp = torch.sum(tp, axis=1).float()
    fp = torch.sum(fp, axis=1).float()
    tn = torch.sum(tn, axis=1).float()
    fn = torch.sum(fn, axis=1).float()

    return func(tp, tn, fp, fn)


def compute_CSI(tp, tn, fp, fn):
    return tp / (tp + fn + fp)


def compute_HSS(tp, tn, fp, fn):
    num = 2 * (tp * tn - fn * fp)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return num / den


def batch_CSI(prediction, target):
    return _compute_custom_metric(prediction, target, compute_CSI)


def batch_HSS(prediction, target):
    return _compute_custom_metric(prediction, target, compute_HSS)


def _get_tp_tn_fp_fn(tp, tn, fp, fn):
    return torch.sum(tp).item(), torch.sum(tn).item(), torch.sum(fp).item(), torch.sum(fn).item()


def get_tp_tn_fp_fn(prediction, target):
    return _compute_custom_metric(prediction, target, _get_tp_tn_fp_fn)


def batch_precision(prediction, target):
    """
    -1e6 is set to those entries where not a single target is 1.
    """
    tp = torch.logical_and(target, target == prediction)
    fp = torch.logical_and(~target, prediction)
    tp = torch.sum(tp, axis=1)
    fp = torch.sum(fp, axis=1)
    precision = -1e6 * tp.new_ones(tp.shape[0])
    N = torch.sum(target, axis=1)
    # NOTE: that tp + fp can still be zero. so there can be some nan entries. It is kept this way to ensure that
    # output of batch_precision and batch_recall are aligned and therefore also of same dimension
    invalid_mask = N == 0
    precision[~invalid_mask] = torch.true_divide(tp[~invalid_mask], (tp + fp)[~invalid_mask])
    precision[torch.isnan(precision)] = 0

    return precision


def batch_recall(prediction, target):
    """
    -1e6 is set to those entries where not a single target is 1.
    """
    tp = torch.logical_and(target, target == prediction)
    tp = torch.sum(tp, axis=1)
    N = torch.sum(target, axis=1)
    invalid_mask = N == 0
    recall = -1e6 * tp.new_ones(tp.shape[0])
    recall[~invalid_mask] = torch.true_divide(tp[~invalid_mask], N[~invalid_mask])
    return recall


def compute_precision_recall(prediction, target, thresholds):
    N = target.shape[0]
    tar = target.view(N, -1).cpu().numpy()
    pred = prediction.view(N, -1).cpu().numpy()
    precision = np.zeros((len(target), len(thresholds)))
    recall = np.zeros((len(target), len(thresholds)))

    for i, th in enumerate(thresholds):
        tar_b = tar > th
        pred_b = pred > th
        precision[:, i] = batch_precision(pred_b, tar_b)
        recall[:, i] = batch_recall(pred_b, tar_b)

    return precision, recall


def compute_precision_recall_from_img_loader(
    img_loader,
    model,
    thresholds=None,
    batch_size=16,
    num_workers=4,
    max_pool_kernel_size=None,
):
    if thresholds is None:
        thresholds = [1, 5, 10, 15, 20]

    precision_data = np.zeros((len(img_loader), len(thresholds)))
    recall_data = np.zeros((len(img_loader), len(thresholds)))
    data_loader = DataLoader(img_loader, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    c_idx = 0
    pool = None
    if max_pool_kernel_size:
        pool = nn.MaxPool2d(max_pool_kernel_size, stride=1, padding=max_pool_kernel_size // 2)

    with torch.no_grad():
        for inp, target, _ in tqdm(data_loader):
            N = inp.shape[0]
            inp = inp.cuda()
            target = target.cuda()
            prediction = model(inp)
            if pool is not None:
                target = pool(target)
                prediction = pool(prediction)
            precision, recall = compute_precision_recall(prediction, target, thresholds)
            precision_data[c_idx:c_idx + N] = precision
            recall_data[c_idx:c_idx + N] = recall
            c_idx += N
    assert c_idx == len(img_loader)

    return precision_data, recall_data


def compute_auroc(generated_data_probab, actual_data_probab):
    pred = torch.cat([generated_data_probab, actual_data_probab])
    tar = torch.zeros_like(pred)
    tar[len(generated_data_probab):] = 1
    return auroc(pred, tar)


class DiscriminatorStats:
    def __init__(self):
        self._genP = None
        self._actP = None
        self.reset()

    def reset(self):
        self._genP = []
        self._actP = []

    def update(self, generated_data_probablity, actual_data_probablity):
        self._genP.append(generated_data_probablity.cpu().view(-1, ))
        self._actP.append(actual_data_probablity.cpu().view(-1, ))

    def get(self, threshold=0.5):
        neg = torch.cat(self._genP)
        pos = torch.cat(self._actP)
        assert len(neg) == len(pos)
        return {
            'auc': compute_auroc(neg, pos),
            'pos_accuracy': torch.mean((pos >= threshold).double()),
            'neg_accuracy': torch.mean((neg < threshold).double()),
            'N': len(pos),
        }

    def raw_data(self):
        return {'actual': torch.cat(self._actP), 'generated': torch.cat(self._genP)}

    def __len__(self):
        return sum([len(x) for x in self._genP])
