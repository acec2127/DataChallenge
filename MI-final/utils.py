import numpy as np
import random
import torch 
from torch import nn

from models import UNet

def filter_encodernames(feature_list):
    encoder_list = UNet().encoder_names
    return list(filter(lambda x: x in encoder_list, feature_list))

def filter_decodernames(feature_list):
    decoder_list = UNet().decoder_names
    return list(filter(lambda x: x in decoder_list, feature_list))

def average_iter(a_list):
    return sum(a_list) / float(len(a_list))

def multiply_iter(iter_a, iter_b):
    return [x * y for x, y in zip(iter_a, iter_b)]

def weighted_average_iter(a_list, weight_list):
    sum_weight = sum(weight_list)+1e-16
    return sum(multiply_iter(a_list, weight_list)) / sum_weight


class FixRandomSeed:
    """
    This class fixes the seeds for numpy and random pkgs.
    """

    def __init__(self, random_seed: int = 0):
        self.random_seed = random_seed
        self.randombackup = random.getstate()
        self.npbackup = np.random.get_state()
        self.torchbackup = torch.get_rng_state()

    def __enter__(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def __exit__(self, *_):
        np.random.set_state(self.npbackup)
        random.setstate(self.randombackup)
        torch.set_rng_state(self.torchbackup)

class FeatureExtractor(nn.Module):
    class _FeatureExtractor:
        def __call__(self, _, input, result):
            self.feature = result

    def __init__(self, net, feature_names) -> None:
        super().__init__()
        self._net = net
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names
        for f in self._feature_names:
            assert f in UNet().component_names, f

    def __enter__(self):
        self._feature_exactors = {}
        self._hook_handlers = {}
        for f in self._feature_names:
            extractor = self._FeatureExtractor()
            handler = getattr(self._net, f).register_forward_hook(extractor)
            self._feature_exactors[f] = extractor
            self._hook_handlers[f] = handler
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self._hook_handlers.items():
            v.remove()
        del self._feature_exactors, self._hook_handlers

    def __getitem__(self, item):
        if item in self._feature_exactors:
            return self._feature_exactors[item].feature
        return super().__getitem__(item)

    def get_feature_from_num(self, num):
        feature = self._feature_names[num]
        return self[feature]

    def __iter__(self):
        for k, v in self._feature_exactors.items():
            yield v.feature

