import os
import pickle
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from models.loss import WeightedMaeLossPixel
from utils.run_utils import checkpoint_file_prefix


def csv_style_print(checkpoint_test, kwargs, loss_base_dict, loss_base_maskless_dict, pdsr_criterias,
                    pdsr_global_criterion):
    target_len = kwargs['tlen']
    print(os.path.basename(checkpoint_test))
    print(kwargs['mt'])
    print(kwargs['v'])
    print('')
    # WMAE
    for t in range(target_len):
        print(np.mean(loss_base_dict[t]))
    print(np.mean([np.mean(v) for _, v in loss_base_dict.items()]))
    print('')

    # Maskless WMAE
    for t in range(target_len):
        print(np.mean(loss_base_maskless_dict[t]))
    print(np.mean([np.mean(v) for _, v in loss_base_maskless_dict.items()]))
    print('')

    # H0,H1
    for t in range(min(2, target_len)):
        pdsr_tmp = pdsr_criterias[t].get(verbose=False)
        for elem in pdsr_tmp['F1']:
            print(elem)
        print('')

    # HSS
    pdsr_tmp = pdsr_criterias[0].get(verbose=False)
    for elem in pdsr_tmp['global_HSS']:
        print(elem)

    print('')
    for elem in pdsr_tmp['global_CSI']:
        print(elem)


def save_pdsr(kwargs, is_test, start_date, end_date, pdsr_criterias, pdsr_global_criterion):
    dic = kwargs.copy()
    dic.pop('start')
    dic.pop('end')
    dic['TEST'] = int(is_test)
    pdsr_fpath = os.path.join('pdsr', checkpoint_file_prefix(start_date, end_date, kwargs=dic) + '.pkl')
    with open(pdsr_fpath, 'wb') as f:
        pdsr_dic = deepcopy(pdsr_criterias)
        pdsr_dic['global'] = pdsr_global_criterion
        pickle.dump(pdsr_dic, f)
    return pdsr_fpath


class QuantileLoss(nn.Module):
    def __init__(self, q, target_len=3):
        super().__init__()
        self._loss = WeightedMaeLossPixel()
        self._tlen = target_len
        self._q = q
        self.loss_data = {t: [] for t in range(self._tlen)}
        self.wmae_data = {t: [] for t in range(self._tlen)}
        self._finalized = False

    def forward(self, predicted, target, mask):
        pixel_loss = self._loss(predicted, target, mask)
        return self.percentile(pixel_loss), torch.mean(pixel_loss)

    def percentile(self, t: torch.tensor):
        assert t.shape[1] == 1
        t = t.view(t.shape[0], -1)
        k = 1 + round(.01 * float(self._q) * (t.shape[-1] - 1))
        result = t.kthvalue(k, dim=-1).values
        return result

    def update(self, predicted, target, mask, seq_idx):
        assert self._finalized is False
        q_loss, wmae = self.forward(predicted, target, mask)
        self.loss_data[seq_idx].append(q_loss.cpu().numpy())
        self.wmae_data[seq_idx].append(wmae.item())

    def finalize(self):
        for t in range(self._tlen):
            self.loss_data[t] = np.concatenate(self.loss_data[t])
        self._finalized = True
