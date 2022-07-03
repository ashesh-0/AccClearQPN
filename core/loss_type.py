from core.base_enum import Enum


class LossType(Enum):
    WeightedMAE = 0
    BlockWeightedMAE = 1
    BlockWeightedAvgMAE = 2
    BlockWeightedMAEDiversify = 3
    WeightedAbsoluteMAE = 4
    MAE = 5
    ReluWeightedMaeLoss = 6
    BalancedWeightedMaeLoss = 7
    WeightedMaeLossDiversify = 8
    WeightedMAEWithBuffer = 9
    KernelWeightedMAE = 10
    WeightedMAEandMSE = 11
    ClassificationBCE = 12
    SSIMBasedLoss = 13
    NormalizedSSIMBasedLoss = 14
    WeightedMSE = 15


class BlockAggregationMode(Enum):
    MAX = 0
    MEAN = 1
