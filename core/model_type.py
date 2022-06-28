from core.enum import Enum


class ModelType(Enum):
    BaselineCNN = 0
    TrajGRU = 1
    TrajGRUWithPrior = 2
    GRU = 3
    TrajGRUAdverserial = 4
    GRUAdverserial = 5
    BalancedGRUAdverserial = 6
    GRUAdverserialRadarPrior = 7
    BalancedGRUAdverserialRadarPrior = 8
    BalancedGRUAdverserialFinetuned = 9
    BalancedGRUAdverserialAttention = 10
    BalancedGRUAdverserialAttentionZeroLoss = 11
    BalancedGRUAdverserialAttention3Opt = 12
    BalancedGRUAdverserialConstrained = 13
    GRUAttention = 14
    ClassifierGRU = 15
    SSIMGRUModel = 16
