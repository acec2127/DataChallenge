from models import UNet
from meters import RandScore
from dataloader import get_dataloaders
from train import TrainerMutualInfo

from loss_perm_inv import GlobalLoss

import torch

config = {
    'model':{
        'input_dim' : 1,
        'num_class' : 1
    },
    'optimizer':{
        'max_lr' : 5e-4,
        'pct_start' : 0.1,
        'div_factor' : 25,
        'final_div_factor' : 1e2
    },

    'dataloader':{
        'val_ratio' : 0.05,
        'batch_size' : 8,
        'n_geo' : 1,
        'n_color' : 1,
        'magnitude' : 0.4,
        'num_workers' : 0  
    },

    'Trainer':{
        'max_epoch' : 6,
        'num_epoch_record' : 1,
        'feature_names' : ["Conv4", "Up_conv3", "Up_conv2"],
        #need to sum to 1
        'feature_importance' : [0.5, 0.25, 0.25],
        #regularizer weight
        'reg_weight' : 0.4,
        'num_clusters' : 10,
        'num_subheads' : 1,
        'patch_sizes' : 1024,
        'paddings': [1, 3], 
        'name_experiment' : '1_suite',
        #'device' : 'cuda',
        'load' : True
    }
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(config['model']['input_dim'], config['model']['num_class']).to(device)

labeled_train_loader, unlabeled_train_loader,\
      labeled_val_loader, unlabeled_val_loader,\
         transform_unlabeled = get_dataloaders(
            config['dataloader']['val_ratio'], 
            config['dataloader']['batch_size'], 
            config['dataloader']['n_geo'], 
            config['dataloader']['n_color'],
            config['dataloader']['magnitude'],
            config['dataloader']['num_workers'], 
            #device
          )

sup_criterion = GlobalLoss().to(device)
#torch.nn.CrossEntropyLoss()
metric = RandScore()

trainer = TrainerMutualInfo(
    config,
    model,
    labeled_train_loader, 
    unlabeled_train_loader, 
    labeled_val_loader, 
    unlabeled_val_loader, 
    sup_criterion, 
    metric, 
    transform_unlabeled,
    device, 
    config['Trainer']['load'],
    config['model']['num_class'],
    max_epoch=config['Trainer']['max_epoch'],
    num_epoch_record=config['Trainer']['num_epoch_record'], 
    feature_names=config['Trainer']['feature_names'], 
    feature_importance=config['Trainer']['feature_importance'], 
    reg_weight=config['Trainer']['reg_weight'], 
    num_clusters=config['Trainer']['num_clusters'], 
    num_subheads=config['Trainer']['num_subheads'],
    patch_sizes=config['Trainer']['patch_sizes'], 
    paddings=config['Trainer']['paddings'], 
    name_experiment=config['Trainer']['name_experiment']
)

trainer.inference()


