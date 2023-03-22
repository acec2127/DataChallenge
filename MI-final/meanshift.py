from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


class MeanShiftCluster(nn.Module):
    """
    GPU meanshift and labeling, implemented by yjl, at 20201106
    """
    @torch.no_grad()
    def __init__(self, spatial_radius=9, range_radius=0.5, num_iteration=4, cuda=True, use_spatial=False):
        super(MeanShiftCluster, self).__init__()
        self.sradius = spatial_radius
        self.sdiameter = spatial_radius * 2 + 1

        self.rradius = range_radius
        self.num_iter = num_iteration
        self.cuda = cuda
        self.use_spatial = use_spatial

        # here, computed sigma is tested method, for center weights is 2 to 3 times over edge weights
        self.ssigma = np.sqrt(2*self.sradius**2)/1.5

        self.sqrt_pi = np.sqrt(2*np.pi)
        self.rsigma = self.rradius/3.0

        self.nei_kernel, self.rng_kernel, self.spt_wght = self.create_mf_kernels()

    @torch.no_grad()
    def gaussian(self, x, sigma=1.0):
        expVal = torch.mul(-0.5, torch.pow(torch.mul(x, 1.0/sigma), 2))
        return torch.mul(torch.exp(expVal), 1.0/(sigma*self.sqrt_pi))

    @torch.no_grad()
    def create_mf_kernels(self):
        """
        @Output: two conv kernel ([out, in ,kh, kw]) to compute neighbour info.
                 spatial gaussian weights (1, out, 1, 1) for each neighbour
        """
        axis_x, axis_y = np.meshgrid(range(self.sdiameter), range(self.sdiameter))
        cy, cx = self.sradius, self.sradius
        spt_size = self.sdiameter*self.sdiameter

        spt_kernel = torch.sqrt(torch.FloatTensor((axis_x-cx)**2 + (axis_y-cy)**2))
        idxM = torch.FloatTensor(axis_x + axis_y*self.sdiameter)
        if self.cuda:
            idxM, spt_kernel = idxM.cuda(), spt_kernel.cuda()

        # range_kernel for conv to compute rng_dist
        nei_kernel = torch.eye(spt_size).cuda()[idxM.long()].permute(2, 0, 1)  # [K*K, K, K]
        rng_kernel = torch.eye(spt_size).cuda()[idxM.long()].permute(2, 0, 1)  # [K*K, K, K]
        rng_kernel[rng_kernel > 0] = -1
        rng_kernel[:, cy, cx] += 1

        # pre-computed spatial weights
        spt_wght = None

        if self.use_spatial:
            spt_kernel = spt_kernel.reshape(-1)  # [K*K]
            spt_wght = self.gaussian(spt_kernel, sigma=self.ssigma)
            spt_wght = spt_wght[None, :, None, None]

        nei_kernel, rng_kernel = nei_kernel[:, None, :, :], rng_kernel[:, None, :, :]
        return nei_kernel, rng_kernel, spt_wght

    @torch.no_grad()
    def compute_pairwise_conv(self, tensor, kernel):
        """
        @func: compute pairwise relationship
        @param: tensor -- size [bs, 1, ht, wd]
                kernel -- size [N, 1, kht, kwd]
        @output: result in size [bs, N, ht, wd]
        """
        kht, kwd = kernel.shape[2], kernel.shape[3]
        pad_ht, pad_wd = kht//2, kwd//2
        padT = torch.nn.ReplicationPad2d([pad_wd, pad_wd, pad_ht, pad_ht])(tensor)

        out = F.conv2d(padT, kernel)  # [bs, N, ht, wd]
        assert(out.shape[2:] == tensor.shape[2:])
        return out

    @torch.no_grad()
    def meanshift(self, features):
        """
        @func: meanshift segmentation algorithm:
             https://www.cnblogs.com/ariel-dreamland/p/9419154.html#:~:text=Mean%20Shift%E7%AE%97%E6%B3%95%EF%BC%8C%E4%B8%80%E8%88%AC%E6%98%AF,%E7%9B%AE%E6%A0%87%E8%B7%9F%E8%B8%AA%E5%92%8C%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E3%80%82
        @Param: input - [bs, 1, ht, wd]
        """
        x = features
        for _ in range(self.num_iter):
            nei_x = self.compute_pairwise_conv(x, self.nei_kernel)
            rng_diff = self.compute_pairwise_conv(x, self.rng_kernel)
            rng_wght = self.gaussian(rng_diff, sigma=self.rsigma) # [bs, nei-size, ht, wd]
            rng_wght = rng_wght - (rng_wght<self.rradius).float()*rng_wght*.9

            if self.use_spatial:
                rng_wght = rng_wght * self.spt_wght

            x = ((rng_wght*nei_x).sum(axis=1)/rng_wght.sum(axis=1))[:, None, :, :]
        return x

    @torch.no_grad()
    def forward(self, features):
        """
        Params:
            features: tensor in size [bs, 1, ht, wd].
        Outputs:
            list of onehot label, each in size [N, ht, wd]
            list of bboxes, each in size [N, 7], represents
                    [bs_idx, x0,y0,x1,y1,integer_label, real_label]
        """
        bs, _, ht, wd = features.size()

        # Meanshift smoothing
        relu_features = nn.ReLU()(features)
        mf_features = self.meanshift(relu_features)
        mf_features = mf_features[:, 0, :, :]

        # Discretized label
        labelImgs = torch.round(mf_features/(self.rradius+1e-2)).int()
        labelImgs = torch.round((labelImgs - labelImgs.min()) / (labelImgs.max() - labelImgs.min()) * 254).int()

        return labelImgs