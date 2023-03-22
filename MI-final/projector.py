from torch.nn import functional as F
from torch import nn

from utils import filter_encodernames, filter_decodernames
from models import UNet

class Flatten(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features):
        b, *_ = features.shape
        return features.view(b, -1)

class SoftmaxWithT(nn.Softmax):

    def __init__(self, dim, T: float = 0.1) -> None:
        super().__init__(dim)
        self._T = T

    def forward(self, input):
        input /= self._T
        return super().forward(input)

class Normalize(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return nn.functional.normalize(input, p=2, dim=1)

class Identical(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return input


class EncoderClusterHead(nn.Module):
    def __init__(self, input_dim, num_clusters=5, num_subheads=10, head_type="linear", T=1, normalize=False) -> None:
        super().__init__()
        assert head_type in ("linear", "mlp"), head_type
        self._input_dim = input_dim
        self._num_clusters = num_clusters
        self._num_subheads = num_subheads
        self._T = T
        self._normalize = normalize

        def init_sub_header(head_type):
            if head_type == "linear":
                return nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(self._input_dim, self._num_clusters),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )
            else:
                return nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(self._input_dim, 128),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Linear(128, num_clusters),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )

        headers = [
            init_sub_header(head_type)
            for _ in range(self._num_subheads)
        ]

        self._headers = nn.ModuleList(headers)

    def forward(self, features):
        return [x(features) for x in self._headers]

class LocalClusterHead(nn.Module):
    """
    this classification head uses the loss for IIC segmentation, which consists of multiple heads
    """

    def __init__(self, input_dim, head_type="linear", num_clusters=10, num_subheads=10, T=1, interm_dim=64, normalize=False) -> None:
        super().__init__()
        assert head_type in ("linear", "mlp"), head_type
        self._T = T
        self._normalize = normalize

        def init_sub_header(head_type):
            if head_type == "linear":
                return nn.Sequential(
                    nn.Conv2d(input_dim, num_clusters, 1, 1, 0),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(input_dim, interm_dim, 1, 1, 0),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Conv2d(interm_dim, num_clusters, 1, 1, 0),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )

        headers = [init_sub_header(head_type) for _ in range(num_subheads)]
        self._headers = nn.ModuleList(headers)

    def forward(self, features):
        return [x(features) for x in self._headers]

class LocalClusterWrappaer(nn.Module):
    def __init__(
        self,
        feature_names,
        head_types = "linear",
        num_subheads = 5,
        num_clusters= 10,
        normalize = False
    ) -> None:
        super(LocalClusterWrappaer, self).__init__()
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names

        self._clusters = nn.ModuleDict()

        for f in self._feature_names:
            self._clusters[f] = self._create_clusterheads(
                input_dim=UNet.dimension_dict[f],
                head_type=head_types,
                num_clusters=num_clusters,
                num_subheads=num_subheads,
                normalize=normalize
            )

    def __len__(self):
        return len(self._feature_names)

    def __iter__(self):
        for k, v in self._clusters.items():
            yield v

    def __getitem__(self, item):
        if item in self._clusters.keys():
            return self._clusters[item]
        return super().__getitem__(item)

    @staticmethod
    def _create_clusterheads(*args, **kwargs):
        return LocalClusterHead(*args, **kwargs)

class EncoderClusterWrapper(LocalClusterWrappaer):

    @staticmethod
    def _create_clusterheads(*args, **kwargs):
        return EncoderClusterHead(*args, **kwargs)

class ProjectorWrapper(nn.Module):
    ENCODER_INITIALIZED = False
    DECODER_INITIALIZED = False

    def init_encoder(
        self,
        feature_names,
        head_types= "linear",
        num_subheads = 5,
        num_clusters = 10,
        normalize = False
    ):
        self._encoder_names = filter_encodernames(feature_names)
        self._encoder_projectors = EncoderClusterWrapper(
            self._encoder_names, head_types, num_subheads,
            num_clusters, normalize)
        self.ENCODER_INITIALIZED = True

    def init_decoder(self,
                     feature_names,
                     head_types = "linear",
                     num_subheads = 5,
                     num_clusters = 10,
                     normalize = False
                     ):
        self._decoder_names = filter_decodernames(feature_names)
        self._decoder_projectors = LocalClusterWrappaer(
            self._decoder_names, head_types, num_subheads,
            num_clusters, normalize)
        self.DECODER_INITIALIZED = True

    @property
    def feature_names(self):
        return self._encoder_names + self._decoder_names

    def __getitem__(self, item):
        if self.ENCODER_INITIALIZED and item in self._encoder_projectors._feature_names:
            return self._encoder_projectors[item]
        elif self.DECODER_INITIALIZED and item in self._decoder_projectors._feature_names:
            return self._decoder_projectors[item]
        raise IndexError(item)

    def __iter__(self):
        if (self.ENCODER_INITIALIZED and self.DECODER_INITIALIZED) is not True:
            raise RuntimeError(f"Encoder_projectors or Decoder_projectors are not initialized "
                               f"in {self.__class__.__name__}.")
        if self.ENCODER_INITIALIZED:
            yield from self._encoder_projectors
        if self.DECODER_INITIALIZED:
            yield from self._decoder_projectors
