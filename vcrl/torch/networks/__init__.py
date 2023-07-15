"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
from vcrl.torch.networks.basic import (
    Clamp, ConcatTuple, Detach, Flatten, FlattenEach, Split, Reshape,
)
from vcrl.torch.networks.cnn import BasicCNN, CNN, MergedCNN, CNNPolicy
from vcrl.torch.networks.dcnn import DCNN, TwoHeadDCNN
from vcrl.torch.networks.feat_point_mlp import FeatPointMlp
from vcrl.torch.networks.image_state import ImageStatePolicy, ImageStateQ
from vcrl.torch.networks.linear_transform import LinearTransform
from vcrl.torch.networks.normalization import LayerNorm
from vcrl.torch.networks.mlp import (
    Mlp, ConcatMlp, MlpPolicy, TanhMlpPolicy,
    MlpVf,
    MlpQf,
    MlpQfWithObsProcessor,
    ConcatMultiHeadedMlp,
)
from vcrl.torch.networks.pretrained_cnn import PretrainedCNN
from vcrl.torch.networks.two_headed_mlp import TwoHeadMlp

__all__ = [
    'Clamp',
    'ConcatMlp',
    'ConcatMultiHeadedMlp',
    'ConcatTuple',
    'BasicCNN',
    'CNN',
    'CNNPolicy',
    'DCNN',
    'Detach',
    'FeatPointMlp',
    'Flatten',
    'FlattenEach',
    'LayerNorm',
    'LinearTransform',
    'ImageStatePolicy',
    'ImageStateQ',
    'MergedCNN',
    'Mlp',
    'PretrainedCNN',
    'Reshape',
    'Split',
    'TwoHeadDCNN',
    'TwoHeadMlp',
]

