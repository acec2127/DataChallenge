import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def huber_loss(y_pred, theta=0.1):
    """
    Huber Loss from DVIS paper
    https://github.com/jia2lin3yuan1/2020-instanceSeg/blob/main/layers/modules/instance_loss.py
    """
    less_grad_factor = 1. / (2 * theta)
    less_loss_bias = less_grad_factor * theta ** 2
    less_than_theta = (y_pred < theta).float()

    loss = less_than_theta * (y_pred ** 2) * less_grad_factor + \
           (1 - less_than_theta) * (y_pred - theta + less_loss_bias)

    return loss


class BinaryLoss(nn.Module):
    """Compute binary loss to force background pixels close to 0 and foreground pixels far from 0"""
    def __init__(self, margin=2.0):
        super(BinaryLoss, self).__init__()

        self.margin = margin

    def forward(self, y_pred, y_true):
        isFG = (y_true > 0.5).float()
        loss_0 = huber_loss(F.relu(y_pred))
        loss_1 = huber_loss(F.relu(self.margin - y_pred))

        loss = (1 - isFG)*loss_0 + isFG*loss_1

        return loss.mean().float()


class PermuInvLoss(nn.Module):
    """
    This class compute the permutation-invariant loss between the targets and the predictions.
    The pixels used to compute pi_loss is sampled on each object
    It encourages pixels in same instance close to each other,
                  pixels from different instances far away from each other.
    """
    def __init__(self, margin=1.0, pi_pairs=2048, pos_wgt=3.0):
        super(PermuInvLoss, self).__init__()

        self.margin = margin
        self.pi_pairs = pi_pairs
        self.pos_wgt = pos_wgt

    def sampling_over_objects(self, true_1D):
        """
        We do random sampling on pi_pairs number of pixels only on foreground pixels
        The binary loss already enables to distinguish background from foreground
        true_1D: One ground truth segmentation [N, ch], no batch_size
        """
        # First we select the indexes for foreground pixels
        idxs_FG = torch.nonzero(true_1D.squeeze()).squeeze()

        # Create permutation to get random selection
        perm = torch.randperm(idxs_FG.size(0))

        # Taking only pi_pairs elements
        idx = perm[:self.pi_pairs]

        return idxs_FG[idx]

    def forward(self, y_pred, y_true):
        """
        Compute the permutation invariant loss on pixel pairs if both pixels are in the instances.
        """
        bs, ch, ht, wd = y_pred.size()
        preds_1D = y_pred.view(bs, ch, ht*wd).permute(0, 2, 1)  # [bs, N, ch] with N = ht*wd
        true_1D = y_true.view(bs, ch, ht*wd).permute(0, 2, 1)  # [bs, N, ch] with N = ht*wd

        all_losses = []

        for el in range(bs):
            # We get the indexes on which we want to compute the loss randomly
            with torch.no_grad():
                smpl_idxs = self.sampling_over_objects(true_1D[el])

            # Getting corresponding predictions and true labels
            smpl_pred = preds_1D[el][smpl_idxs].squeeze()
            smpl_true = true_1D[el][smpl_idxs].squeeze()

            # We then want to compute distances between all possible pairs
            smpl_pred_combs = torch.combinations(smpl_pred, r=2)
            smpl_true_combs = torch.combinations(smpl_true, r=2)

            # Computing all distances and
            pi_pred = torch.abs(F.relu(smpl_pred_combs[:, 0] - F.relu(smpl_pred_combs[:, 1])))
            pi_target = (smpl_true_combs[:, 0] == smpl_true_combs[:, 1]).float()

            loss_1 = huber_loss(pi_pred)
            loss_0 = huber_loss(self.margin - pi_pred)

            loss = (3.0*pi_target*loss_1 + (1. - pi_target)*loss_0).mean()  # We penalize more when gt is the same
            all_losses.append(loss)

        all_losses = torch.stack(all_losses).mean()

        return all_losses


class QuantizationLoss(nn.Module):
    """
    This class computes the quantization distance on predictions of the network.
    It only works for the regression model that network directly output 1 channel instance label
    """
    def __init__(self):
        super(QuantizationLoss, self).__init__()

    def forward(self, y_pred):
        """
        @Param: y_pred -- instance map after relu. size [bs, 1, ht, wd], only on foreground
        """
        loss = torch.abs(y_pred[y_pred > 0.] - torch.round(y_pred[y_pred > 0.])).mean()
        loss = loss if not loss.isnan() else torch.Tensor([0.]).to(loss.device)

        return loss


class RegularizationLoss(nn.Module):
    """
    This class computes the Mumford-Shah regularization loss on the prediction of the network.
    The way to approximate it is introduced in https://arxiv.org/abs/2007.11576
    """
    def __init__(self):
        super(RegularizationLoss, self).__init__()

        # create kernel_h/v in size [1, 1, 3, 3]
        kernel_h = np.zeros([1, 1, 3, 3], dtype=np.float32)
        kernel_h[:, :, 1, 0] = 0.0
        kernel_h[:, :, 1, 1] = 1.0
        kernel_h[:, :, 1, 2] = 1.0
        self.kernel_h = torch.FloatTensor(kernel_h).cuda()

        kernel_v = np.zeros([1, 1, 3, 3], dtype=np.float32)
        kernel_v[:, :, 0, 1] = 0.0
        kernel_v[:, :, 1, 1] = 1.0
        kernel_v[:, :, 2, 1] = -1.0
        self.kernel_v = torch.FloatTensor(kernel_v).cuda()

    def forward(self, y_pred):
        """
        @Param: preds -- instance map after relu. size [bs, ch, ht, wd]
        """
        ch = y_pred.size(1)
        loss = []
        for k in range(ch):
            loss_h = F.conv2d(y_pred[:, k: k+1, :, :], self.kernel_h)
            loss_v = F.conv2d(y_pred[:, k: k+1, :, :], self.kernel_v)

            tmp = torch.log(loss_h**2 + loss_v**2 + 1)
            loss.extend([tmp])

        loss = torch.stack(loss)
        if loss.max() > 0:
            return loss[loss > 0].mean()
        else:
            return None


class GlobalLoss(nn.Module):
    """Sum all previous losses to get final loss"""
    def __init__(self):
        super(GlobalLoss, self).__init__()

        self.BinaryLoss = BinaryLoss()
        self.QuantizationLoss = QuantizationLoss()
        self.RegularizationLoss = RegularizationLoss()
        self.PermuInvLoss = PermuInvLoss()

    def forward(self, y_pred, y_true):
        lb = self.BinaryLoss(y_pred, y_true)
        lq = self.QuantizationLoss(y_pred)
        lr = self.RegularizationLoss(y_pred)
        lpi = self.PermuInvLoss(y_pred, y_true)

        # print(lb.item(), lq.item(), lr.item(), lpi.item())

        return 10.*lb + .5*lq + .1*lr + 1.*lpi

