import sys
import numpy as np
import torch
from torch import nn

from itertools import repeat

import torch.nn.functional as F

from utils import filter_encodernames, filter_decodernames, average_iter


def simplex(t, axis=1) -> bool:
    """
    check if the matrix is the probability distribution
    :param t:
    :param axis:
    :return:
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, rtol=1e-4, atol=1e-4)

def patch_generator(feature_map, patch_size=(32, 32), step_size=(16, 16)):
    b, c, h, w = feature_map.shape
    hs = np.arange(0, h - patch_size[0], step_size[0])
    hs = np.append(hs, max(h - patch_size[0], 0))
    ws = np.arange(0, w - patch_size[1], step_size[1])
    ws = np.append(ws, max(w - patch_size[1], 0))
    for _h in hs:
        for _w in ws:
            yield feature_map[:, :, _h:min(_h + patch_size[0], h), _w:min(_w + patch_size[1], w)]
            # yield [_h, min(_h + patch_size[0], h), _w, min(_w + patch_size[1], w)]

def compute_joint(x_out, x_tf_out, symmetric=True) :
    r"""
    return joint probability
    :param x_out: p1, simplex
    :param x_tf_out: p2, simplex
    :return: joint probability
    """
    # produces variable that requires grad (since args require grad)
    assert simplex(x_out), f"x_out not normalized."
    assert simplex(x_tf_out), f"x_tf_out not normalized."

    bn, k = x_out.shape
    assert x_tf_out.size(0) == bn and x_tf_out.size(1) == k

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k aggregated over one batch
    if symmetric:
        p_i_j = (p_i_j + p_i_j.t()) / 2.0  # symmetric
    p_i_j /= p_i_j.sum()  # normalise

    return p_i_j

class _IIDLoss(nn.Module):
    def __init__(self, lamb: float = 1.0, eps: float = sys.float_info.epsilon):
        """
        :param lamb:
        :param eps:
        """
        super().__init__()
        print(f"Initialize {self.__class__.__name__}.")
        self.lamb = float(lamb)
        self.eps = float(eps)
        self.torch_vision = torch.__version__

    def forward(self, x_out, x_tf_out):
        """
        return the inverse of the MI. if the x_out == y_out, return the inverse of Entropy
        :param x_out:
        :param x_tf_out:
        :return:
        """
        assert simplex(x_out), f"x_out not normalized."
        assert simplex(x_tf_out), f"x_tf_out not normalized."
        _, k = x_out.size()
        p_i_j = compute_joint(x_out, x_tf_out)
        assert p_i_j.size() == (k, k)

        p_i = (
            p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        )  # p_i should be the mean of the x_out
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

        # p_i = x_out.mean(0).view(k, 1).expand(k, k)
        # p_j = x_tf_out.mean(0).view(1, k).expand(k, k)
        #

        loss = -p_i_j * (
            torch.log(p_i_j + 1e-10) - self.lamb * torch.log(p_j + 1e-10) - self.lamb * torch.log(p_i + 1e-10)
        )
        loss = loss.sum()
        loss_no_lamb = -p_i_j * (torch.log(p_i_j + 1e-10) - torch.log(p_j + 1e-10) - torch.log(p_i + 1e-10))
        loss_no_lamb = loss_no_lamb.sum()
        return loss, loss_no_lamb, p_i_j

class IIDLoss(_IIDLoss):

    def forward(self, x_out, x_tf_out):
        return super().forward(x_out, x_tf_out)[0]

class IIDSegmentationLoss(nn.Module):
    def __init__(
        self, lamda=1.0, padding=7, eps: float = sys.float_info.epsilon
    ) :
        super(IIDSegmentationLoss, self).__init__()
        print(f"Initialize {self.__class__.__name__}.")
        self.lamda = lamda
        self.padding = padding
        self.eps = eps

    def __call__(
        self, x_out, x_tf_out, mask = None
    ) :
        assert simplex(x_out)
        assert x_out.shape == x_tf_out.shape
        bn, k, h, w = x_tf_out.shape
        if mask is not None:
            x_out = x_out * mask
            x_tf_out = x_tf_out * mask

        x_out = x_out.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
        x_tf_out = x_tf_out.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
        # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
        p_i_j = F.conv2d(x_out, weight=x_tf_out, padding=self.padding)
        p_i_j = p_i_j - p_i_j.min().detach() + 1e-16
        T_side_dense = self.padding * 2 + 1

        # T x T x k x k
        p_i_j = p_i_j.permute(2, 3, 0, 1)
        p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)  # norm

        # symmetrise, transpose the k x k part
        p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0

        # T x T x k x k
        p_i_mat = p_i_j.sum(dim=2, keepdim=True).repeat(1, 1, k, 1)
        p_j_mat = p_i_j.sum(dim=3, keepdim=True).repeat(1, 1, 1, k)

        # maximise information
        loss = (
                   -p_i_j
                   * (
                       torch.log(p_i_j + 1e-16)
                       - self.lamda * torch.log(p_i_mat + 1e-16)
                       - self.lamda * torch.log(p_j_mat + 1e-16)
                   )
               ).sum() / (T_side_dense * T_side_dense)
        if torch.isnan(loss):
            raise RuntimeError(loss)
        return loss

class IIDSegmentationSmallPathLoss(IIDSegmentationLoss):

    def __init__(self, lamda=1.0, padding=7, eps: float = sys.float_info.epsilon, patch_size=32) -> None:
        super().__init__(lamda, padding, eps)
        self._patch_size = tuple(repeat(patch_size, 2))
        self._step_size = tuple(repeat(patch_size // 2, 2))

    def __call__(self, x_out, x_tf_out, mask = None):
        assert x_out.shape == x_tf_out.shape, (x_out.shape, x_tf_out.shape)
        if mask is None:
            iic_patch_list = [super(IIDSegmentationSmallPathLoss, self).__call__(x, y) for x, y in zip(
                patch_generator(x_out, self._patch_size, self._step_size),
                patch_generator(x_tf_out, self._patch_size, self._step_size)
            )]
        else:
            iic_patch_list = [super(IIDSegmentationSmallPathLoss, self).__call__(x, y, m) for x, y, m in zip(
                patch_generator(x_out, self._patch_size, self._step_size),
                patch_generator(x_tf_out, self._patch_size, self._step_size),
                patch_generator(mask, self._patch_size, self._step_size)
            )]
        if any([torch.isnan(x) for x in iic_patch_list]):
            raise RuntimeError(iic_patch_list)
        return average_iter(iic_patch_list)

    def __repr__(self):
        return f"{self.__class__.__name__} with patch_size={self._patch_size} and padding={self.padding}."

class IICLossWrapper(nn.Module):

    def __init__(self,
                 feature_names,
                 paddings,
                 patch_sizes) -> None:
        super().__init__()
        self._encoder_features = filter_encodernames(feature_names)
        self._decoder_features = filter_decodernames(feature_names)
        assert len(feature_names) == len(self._encoder_features) + len(self._decoder_features)
        self._LossModuleDict = nn.ModuleDict()

        if len(self._encoder_features) > 0:
            for f in self._encoder_features:
                self._LossModuleDict[f] = IIDLoss()
        if len(self._decoder_features) > 0:
            paddings = paddings
            patch_sizes = list(repeat(patch_sizes, len(self._decoder_features)))
            for f, p, size in zip(self._decoder_features, paddings, patch_sizes):
                self._LossModuleDict[f] = IIDSegmentationSmallPathLoss(padding=p, patch_size=size)

    def __getitem__(self, item):
        if item in self._LossModuleDict.keys():
            return self._LossModuleDict[item]
        raise IndexError(item)

    def __iter__(self):
        for k, v in self._LossModuleDict.items():
            yield v

    def items(self):
        return self._LossModuleDict.items()

    @property
    def feature_names(self):
        return self._encoder_features + self._decoder_features