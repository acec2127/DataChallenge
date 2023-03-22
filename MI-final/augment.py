import random
import torch
from typing import Callable, List

from PIL import Image
from torch import Tensor

import torchvision.transforms as T

from utils import FixRandomSeed


class TransformLabeled:
    def __init__(self, n_geo, n_color, magnitude, mode):
        self.n_geo = n_geo 
        self.n_color = n_color
        self.magnitude = magnitude 
        self.mode = mode

        self.geo_list = [
            T.RandomRotation(int(self.magnitude * 180)),
            T.RandomResizedCrop(512, scale = (1 - self.magnitude*0.4, 1.0) ),
            T.RandomAffine(0, translate = (self.magnitude, self.magnitude)),
            T.RandomAffine(0, scale = ((1 - self.magnitude)*0.65, (1 - self.magnitude)*1.5)),
            T.RandomAffine(0, shear = (-int(self.magnitude*0.5 * 180), int(self.magnitude * 180))), 
            T.RandomAffine(0, shear = (0, 0, -int(self.magnitude *0.5* 180), int(self.magnitude *0.5* 180)))
        ]

        self.color_list = [
            T.RandomAdjustSharpness(1 + 2*self.magnitude),
            T.RandomAdjustSharpness(1 - 0.5*self.magnitude),
            T.ColorJitter(brightness = self.magnitude),
            T.ColorJitter(contrast = self.magnitude),
        ]


    def __call__(self, img, label, seed):
        if self.mode == 'train':
            geo_transform = T.Compose([
                    random.choice(self.geo_list) for _ in range(self.n_geo)
                ])
            color_transform = T.Compose([
                    random.choice(self.color_list) for _ in range(self.n_color)
                ])
            with FixRandomSeed(seed):
                img = geo_transform(img)

            with FixRandomSeed(seed):
                label = geo_transform(label)

            img = color_transform(img)

        return T.functional.normalize(T.functional.pil_to_tensor(img).to(torch.float), 0, 0.5), T.functional.pil_to_tensor(label).squeeze(0)
    
class TransformUnlabeled:
    def __init__(self, n_geo, n_color, magnitude):
        self.n_geo = n_geo 
        self.n_color = n_color
        self.magnitude = magnitude 

        self.geo_list = [
            T.RandomRotation(int(self.magnitude * 180)),
            #T.RandomResizedCrop(512, scale = (1 - self.magnitude*0.4, 1.0) ),
            T.RandomAffine(0, translate = (self.magnitude, self.magnitude)),
            T.RandomAffine(0, scale = ((1 - self.magnitude)*0.65, (1 - self.magnitude)*1.5)),
            T.RandomAffine(0, shear = (-int(self.magnitude*0.5 * 180), int(self.magnitude * 180))), 
            T.RandomAffine(0, shear = (0, 0, -int(self.magnitude *0.5* 180), int(self.magnitude *0.5* 180)))
        ]
        '''
        self.color_list = [
            T.RandomAdjustSharpness(1 + 2*self.magnitude),
            T.RandomAdjustSharpness(1 - 0.5*self.magnitude),
            T.ColorJitter(brightness = self.magnitude),
            T.ColorJitter(contrast = self.magnitude),
        ]
        '''


    def __call__(self, img, seed):
        with FixRandomSeed(seed):
            geo_transform = T.Compose([
                    random.choice(self.geo_list) for _ in range(self.n_geo)
                ])
            '''
            color_transform = T.Compose([
                    random.choice(self.color_list) for _ in range(self.n_color)
                ])
            ''' 

            img_t = geo_transform(img)
            #img_t = color_transform(img_t)

        return T.functional.normalize(T.functional.pil_to_tensor(img).to(torch.float), 0, 0.5),\
              T.functional.normalize(T.functional.pil_to_tensor(img).to(torch.float), 0, 0.5)

class TransformPostUnlabeled:
    def __init__(self, n_geo, n_color, magnitude):
        self.n_geo = n_geo 
        self.n_color = n_color
        self.magnitude = magnitude 
        
        self.geo_list = [
            T.RandomRotation(int(self.magnitude * 180)),
            #T.RandomResizedCrop(512, scale = (1 - self.magnitude*0.4, 1.0) ),
            T.RandomAffine(0, translate = (self.magnitude, self.magnitude)),
            T.RandomAffine(0, scale = ((1 - self.magnitude)*0.65, (1 - self.magnitude)*1.5)),
            T.RandomAffine(0, shear = (-int(self.magnitude*0.5 * 180), int(self.magnitude * 180))), 
            T.RandomAffine(0, shear = (0, 0, -int(self.magnitude *0.5* 180), int(self.magnitude *0.5* 180)))
        ]
        '''
        self.color_list = [
            T.RandomAdjustSharpness(1 + 2*self.magnitude),
            T.RandomAdjustSharpness(1 - 0.5*self.magnitude),
            T.ColorJitter(brightness = self.magnitude),
            T.ColorJitter(contrast = self.magnitude),
        ]
        '''


    def __call__(self, imgs, seeds):
        assert imgs.shape[0] == seeds.shape[0]
        imgs_t = []
        for i in range(seeds.shape[0]) :
            with FixRandomSeed(seeds[i]):
                geo_transform = T.Compose([
                        random.choice(self.geo_list) for _ in range(self.n_geo)
                    ])
                '''
                color_transform = T.Compose([
                        random.choice(self.color_list) for _ in range(self.n_color)
                    ])
                '''

                img_t = geo_transform(imgs[i])
                #print("shape img_t : ", img_t.shape)
                #img_t = color_transform(img_t)
                imgs_t.append(img_t)

        return torch.stack(imgs_t, dim=0)
    
