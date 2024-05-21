import enum


class BackboneArch(str, enum.Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    SWIN_V2 = "swinv2_base_window12to16_192to256_22kft1k"

    def __str__(self) -> str:
        return self.value

class OptimizerType(str, enum.Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"

    def __str__(self) -> str:
        return self.value
    
class LossName(str, enum.Enum):
    MULTI_SIMILARITY_LOSS = "MultiSimilarityLoss"
    TRIPLET_MARGIN_LOSS = "TripletMarginLoss"
    CONTRASTIVE_LOSS = "ContrastiveLoss"

    def __str__(self) -> str:
        return self.value
    
import enum

class DatasetOptions(str, enum.Enum):
    GSV_XS = "GSV-XS"
    SF_XL = "SF-XL"
    TOKYO_XL = "TOKYO-XL"

    def __str__(self) -> str:
        return self.value

class ExperimentPhase(str, enum.Enum):
    ALL = "all"
    TRAIN = "train"
    TEST = "test"
    METRIC = "metric"

    def __str__(self) -> str:
        return self.value
