import numpy as np
from tqdm import tqdm


class RainStats:
    """
    1. Distribu. of rain rate (eg:90 percentile=>10mm/hr)
    2. Pixel variations
    3. Rainfall variation (rainy days vs)
    """

    def __init__(self, thresholds, input_shape):
        assert 0 not in thresholds
        assert 1000 not in thresholds
        self._T = thresholds + [max(thresholds) + 1000]
        self._shape = input_shape
        # per pixel variation, how many pixels have
        self._th_data = None
        self._rr_sum = None

        self.reset()

    def reset(self):
        self._th_data = [np.zeros(self._shape) for _ in self._T]
        self._rr_sum = []

    def update_rr_sum(self, rain):
        assert rain.shape == self._shape
        self._rr_sum.append(np.sum(rain))

    def update_th_data(self, rain):
        low = 0
        masks = []
        for i, high in enumerate(self._T):
            mask = np.logical_and(rain >= low, rain < high)
            masks.append(mask)
            low = high

        net = 0
        for i, mask in enumerate(masks):
            self._th_data[i] += mask
            net += mask

        net = np.unique(net)
        assert len(net) == 1 and net[0] == 1

    def update(self, rain):
        self.update_rr_sum(rain)
        self.update_th_data(rain)

    def run(self, dataset):
        self.reset()
        for i in tqdm(range(len(dataset._time))):
            rain = dataset._get_raw_rain_data(i)
            self.update(rain)
        self.sanity(len(dataset._time))

    def sanity(self, N):
        net = 0
        for frame in self._th_data:
            net += frame
        net = np.unique(net)
        assert len(net) == 1
        assert net[0] == N, f'N:{N} sum:{net[0]}'

    def get_rain_distribution(self):
        frac = []
        for frame in self._th_data:
            frac.append(frame.sum())
        N = sum(frac)
        frac = [x / N for x in frac]
        return frac
