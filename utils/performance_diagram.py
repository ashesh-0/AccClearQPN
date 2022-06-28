import numpy as np
import torch

from utils.analysis_utils import batch_CSI, batch_HSS, batch_precision, batch_recall, get_tp_tn_fp_fn, compute_HSS, compute_CSI


class PerformanceDiagram:
    """
    Computes the success ratio and probablity of detection for different thresholds.
    """
    def __init__(self, thresholds=None, weights=None):
        self._tlist = thresholds
        if thresholds is None:
            self._tlist = [1, 3, 5, 10, 15, 20, 30, 40, 50]

        # Weights for different thresholds
        self._wlist = weights
        if self._wlist is None:
            self._wlist = [1] * len(self._tlist)

    def compute_PD(self, prediction, target):
        """
        Probablity of detection == RECALL
        """
        return batch_recall(prediction, target)

    def compute_SR(self, prediction, target):
        return batch_precision(prediction, target)

    def compute_PD_SR(self, prediction, target, threshold):
        N = target.shape[0]
        target = target >= threshold
        prediction = prediction >= threshold
        target = target.view(N, -1)
        prediction = prediction.view(N, -1)
        pd = self.compute_PD(prediction, target)
        sr = self.compute_SR(prediction, target)
        return (pd, sr)

    def compute_overall_metric(self, PD_SR_list, verbose=False):
        # Component along (1,1) vector
        mlist = [sum(list(d)) / np.sqrt(2) for d in PD_SR_list]
        if verbose:
            for i, t in enumerate(self._tlist):
                print(f'Threshold:{t} Score:{mlist[i]:.3f} PD:{PD_SR_list[i][0]:.3f} SR:{PD_SR_list[i][1]:.3f}')

        metric = 0
        wsum = 0
        for i, val in enumerate(mlist):
            # if no entry in the batch has a valid target pixel for this threshold, then one gets nan
            if np.isnan(val):
                continue

            metric += self._wlist[i] * val
            wsum += self._wlist[i]

        if wsum > 0:
            return metric / wsum
        else:
            return float('nan')

    def compute(self, prediction, target):
        assert prediction.shape == target.shape
        data = [self.compute_PD_SR(prediction, target, t) for t in self._tlist]
        proc_data = []
        # ignore negative entries
        for PD, SR in data:
            mask = PD >= 0
            elem = (torch.mean(PD[mask]).item(), torch.mean(SR[mask]).item())
            proc_data.append(elem)
        return self.compute_overall_metric(proc_data)


class PerformanceDiagramStable(PerformanceDiagram):
    """
    Here, we aggregate the Probablity of detection and Success ratio over all batches. This was needed as there are a
    lot of frames for which the target has no significant non-zero entry. So, computing the metric for every batch and
     then averaging it over all batches results in a very unstable metric.
    """
    def __init__(self, thresholds=None, weights=None):
        super().__init__(thresholds=thresholds, weights=weights)
        self._pd = None
        self._tr = None
        self._csi = None
        self._hss = None
        self._tp_tn_fp_fn = None
        self.reset()

    def compute_CSI(self, prediction, target):
        return batch_CSI(prediction, target)

    def compute_HSS(self, prediction, target):
        return batch_HSS(prediction, target)

    def binarize(self, prediction, target, threshold):
        target = target >= threshold
        prediction = prediction >= threshold
        return (prediction, target)

    def compute_metrics(self, prediction, target, threshold):
        assert prediction.shape == target.shape
        N = target.shape[0]
        prediction, target = self.binarize(prediction, target, threshold)

        target = target.view(N, -1)
        prediction = prediction.view(N, -1)
        tp_tn_fp_fn = get_tp_tn_fp_fn(prediction, target)

        pd = self.compute_PD(prediction, target)
        sr = self.compute_SR(prediction, target)
        csi = self.compute_CSI(prediction, target)
        hss = self.compute_HSS(prediction, target)
        return (pd, sr, csi, hss, tp_tn_fp_fn)

    def compute(self, prediction, target):
        data = [self.compute_metrics(prediction, target, t) for t in self._tlist]

        # ignore negative entries
        for i, elem in enumerate(data):
            PD, SR, CSI, HSS, tp_tn_fp_fn = elem
            mask = SR >= 0
            t = self._tlist[i]
            self._pd[t] += list(PD[mask].cpu().numpy())
            self._sr[t] += list(SR[mask].cpu().numpy())
            self._hss[t] += list(HSS[mask].cpu().numpy())
            self._csi[t] += list(CSI[mask].cpu().numpy())
            self._tp_tn_fp_fn[t] += np.array(tp_tn_fp_fn)

    def F1_scores(self, pdsr_list):
        output = []
        for pdsr in pdsr_list:
            pd, sr = pdsr
            output.append(2 * pd * sr / (pd + sr))
        return output

    def mean(self, arr):
        mask = ~np.isnan(arr)
        return np.array(arr)[mask].mean()

    def get(self, verbose=False):
        pdsr = []
        hss = []
        csi = []
        global_hss = []
        global_csi = []
        for t in self._tlist:
            elem = (self.mean(self._pd[t]), self.mean(self._sr[t]))
            hss.append(self.mean(self._hss[t]))
            csi.append(self.mean(self._csi[t]))
            pdsr.append(elem)
            tp, tn, fp, fn = self._tp_tn_fp_fn[t]
            global_csi.append(compute_CSI(tp, tn, fp, fn))
            global_hss.append(compute_HSS(tp, tn, fp, fn))
        # pdsr = [elem[:2] for elem in proc_data]
        return {
            'metrics': pdsr,
            'Th': self._tlist,
            'F1': self.F1_scores(pdsr),
            'HSS': hss,
            'CSI': csi,
            'Dotmetric': self.compute_overall_metric(pdsr, verbose=verbose),
            'global_HSS': global_hss,
            'global_CSI': global_csi,
        }

    def reset(self):
        self._pd = {t: [] for t in self._tlist}
        self._sr = {t: [] for t in self._tlist}
        self._csi = {t: [] for t in self._tlist}
        self._hss = {t: [] for t in self._tlist}
        self._tp_tn_fp_fn = {t: np.array([0.0] * 4) for t in self._tlist}
