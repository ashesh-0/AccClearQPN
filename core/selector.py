from copy import deepcopy
from heapq import heappop, heappush


class Element:
    def __init__(self, metric, *args):
        self.metric = metric
        self.data = args

    def __lt__(self, other):
        return self.metric < other.metric


class Selector:
    def __init__(self, k, selection_type):
        assert selection_type in ['min', 'max']
        self._multiplier = -1 if selection_type == 'min' else 1

        self._k = k
        self._data = []

    def add(self, metric, data):
        heappush(self._data, Element(metric * self._multiplier, *data))

        if len(self._data) > self._k:
            _ = heappop(self._data)

    def add_batch(self, metric_list, *data):
        for i in range(len(metric_list)):
            self.add(metric_list[i], tuple(one_data[i] for one_data in data))

    @staticmethod
    def pop(data):
        if len(data):
            elem = heappop(data)
            return (elem.metric, *elem.data)
        return None

    def all(self):
        orig_data = deepcopy(self._data)
        output = []
        last_elem = Selector.pop(orig_data)
        while last_elem is not None:
            output.append(last_elem)
            last_elem = Selector.pop(orig_data)

        return output
