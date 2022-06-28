from core.enum import Enum


class DataLoaderType(Enum):
    Native = 0
    WithPrior = 1
    Grid = 2
    Classifier = 3
