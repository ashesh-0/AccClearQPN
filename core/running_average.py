class RunningAverage:
    def __init__(self):
        self._N = 0
        self._sum = 0

    def get(self):
        if self._N > 0:
            return self._sum / self._N

        return None

    def reset(self):
        self._N = 0
        self._sum = 0

    def add(self, val, n):
        self._N += n
        self._sum += val
