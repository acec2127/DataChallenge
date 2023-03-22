import json 
import os.path
from itertools import chain
import random 
import torch
from torch import optim
from einops import rearrange
import pandas as pd
import numpy as np

from utils import FeatureExtractor, average_iter, weighted_average_iter
from projector import ProjectorWrapper, EncoderClusterHead
from loss import IICLossWrapper
from meanshift import MeanShiftCluster
from dataset import DatasetTest



def save_dict_to_json(dico, filepath) :
    with open(filepath, 'w') as fp:
        json.dump(dico, fp, indent=4)

class TrainerMutualInfo():
    def __init__(self, 
                 settings_dict, 
                 model, 
                 labeled_train_loader, 
                 unlabeled_train_loader, 
                 labeled_val_loader, 
                 unlabeled_val_loader,
                 sup_criterion, 
                 metric, 
                 transform_unlabeled, 
                 device, 
                 num_class,
                 load=False,
                 max_epoch=100, 
                 num_epoch_record=10, 
                 feature_names=["Conv5", "Up_conv3", "Up_conv2"], 
                 feature_importance=[0.5, 0.25, 0.25], 
                 reg_weight=0.1, 
                 num_clusters=10, 
                 num_subheads=5,
                 patch_sizes=1024, 
                 paddings=[1,3], 
                 name_experiment='1'
                 ):
        self.dict_record = {
            'settings' : settings_dict, 
            'epoch': {},
        }
        self.name_experiment = name_experiment

        self.model = model

        self.labeled_train_loader = labeled_train_loader
        self.unlabeled_train_loader = unlabeled_train_loader
        self.labeled_val_loader = labeled_val_loader
        self.unlabeled_val_loader = unlabeled_val_loader

        self.sup_criterion = sup_criterion
        self.metric = metric
        self.reg_weight = reg_weight

        self.num_class = num_class
        self.max_epoch = max_epoch
        self.num_epoch_record = num_epoch_record

        self.transform_unlabeled = transform_unlabeled

        self.feature_importance = feature_importance
        self.feature_positions = feature_names
        assert len(self.feature_importance) == len(self.feature_positions)
        
        self.device = device

        self.meanshift = MeanShiftCluster()

        self.projector_wrappers = ProjectorWrapper()
        self.projector_wrappers.init_encoder(
            feature_names=self.feature_positions,
            num_subheads=num_subheads, 
            num_clusters=num_clusters,
        )
        self.projector_wrappers.init_decoder(
            feature_names=self.feature_positions,
            num_subheads=num_subheads, 
            num_clusters=num_clusters,
        )
        self.IIDSegCriterionWrapper = IICLossWrapper(
            feature_names=self.feature_positions,
            paddings=paddings,
            patch_sizes=patch_sizes,
        )
        self.projector_wrappers.to(self.device)

        self.load = load

        self.init_optim()

    def init_optim(self):
        optim_config = self.dict_record['settings']['optimizer']
        self.optimizer = optim.RAdam(params=chain(self.model.parameters(), self.projector_wrappers.parameters()))
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, optim_config['max_lr'],
                                                        epochs = self.max_epoch,
                                                        steps_per_epoch= 2 * len(self.labeled_train_loader),
                                                        pct_start = optim_config['pct_start'],
                                                        div_factor = optim_config['div_factor'],
                                                        final_div_factor = optim_config['final_div_factor'])                 

    def run_train_epoch(self):
        self.metric.reset_train()
        self.model.train()
        count = 0
        with FeatureExtractor(self.model, self.feature_positions) as self.fextractor:
            for (labeled_image, labeled_target), (unlabeled_image, unlabeled_image_tf, transform_seed) in zip(self.labeled_train_loader, self.unlabeled_train_loader):
                count += 1
                print('Training batch :', count)
                assert unlabeled_image_tf.shape == unlabeled_image.shape, \
                    (unlabeled_image_tf.shape, unlabeled_image.shape)
                predict_logits = self.model(torch.cat([labeled_image.to(self.device), unlabeled_image.to(self.device), unlabeled_image_tf.to(self.device)], dim=0))
                label_logits, unlabel_logits, _ = \
                    torch.split(
                        predict_logits,
                        [len(labeled_image), len(unlabeled_image), len(unlabeled_image_tf)],
                        dim=0
                    )
                #supervised
                #label_logits_sup = rearrange(label_logits, 'b c h w -> (b h w) c')
                #labeled_target_sup = rearrange(labeled_target.to(self.device), 'b h w -> (b h w)')
                labeled_target_sup = labeled_target.to(self.device).unsqueeze(1)
                sup_loss = self.sup_criterion(label_logits, labeled_target_sup)
                # regularized part
                reg_loss = self.regularization(
                        n_unlabeled=unlabel_logits.shape[0],
                        seed=transform_seed
                    )
                total_loss = sup_loss + self.reg_weight * reg_loss
                # gradient backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    seg = self.meanshift(label_logits)
                    seg = rearrange(seg, 'b h w -> b (h w)')
                    #seg = (seg - seg.min(dim=-1)) / (seg.max(dim=-1) - seg.min(dim=-1)) * 255
                    labeled_target_sup = rearrange(labeled_target, 'b h w -> b (h w)')
                    self.metric.add_train(seg.detach().cpu(), labeled_target_sup.detach().cpu())
                    #.max(1)[1]
            self.scheduler.step()
        return sup_loss, reg_loss, total_loss

    def run_val_epoch(self):
        self.metric.reset_val()
        self.model.eval()
        with torch.no_grad() :
            with FeatureExtractor(self.model, self.feature_positions) as self.fextractor:
                for (labeled_image, labeled_target), (unlabeled_image, unlabeled_image_tf, transform_seed) in zip(self.labeled_val_loader, self.unlabeled_val_loader):
                    assert unlabeled_image_tf.shape == unlabeled_image.shape, \
                    (unlabeled_image_tf.shape, unlabeled_image.shape)
                    predict_logits = self.model(torch.cat([labeled_image.to(self.device), unlabeled_image.to(self.device), unlabeled_image_tf.to(self.device)], dim=0))
                    label_logits, unlabel_logits, _ = \
                        torch.split(
                            predict_logits,
                            [len(labeled_image), len(unlabeled_image), len(unlabeled_image_tf)],
                            dim=0
                        )
                    # supervised part
                    labeled_target_sup = labeled_target.to(self.device).unsqueeze(1)
                    sup_loss = self.sup_criterion(label_logits, labeled_target_sup)
                    # regularized part
                    reg_loss = self.regularization(
                        n_unlabeled=unlabel_logits.shape[0],
                        seed=transform_seed
                    )   
                    total_loss = sup_loss + self.reg_weight * reg_loss

                    seg = self.meanshift(label_logits)
                    seg = rearrange(seg, 'b h w -> b (h w)')
                    #seg = (seg - seg.min(dim=-1)) / (seg.max(dim=-1) - seg.min(dim=-1)) * 255
                    labeled_target_sup = rearrange(labeled_target, 'b h w -> b (h w)')
                    self.metric.add_val(seg.detach().cpu(), labeled_target_sup.detach().cpu())
        return sup_loss, reg_loss, total_loss
    
    def regularization(self, n_unlabeled, seed):
        unlabeled_length = n_unlabeled * 2
        iic_losses_for_features = []

        for inter_feature, projector, criterion \
            in zip(self.fextractor, self.projector_wrappers, self.IIDSegCriterionWrapper):
            unlabeled_features = inter_feature[len(inter_feature) - unlabeled_length:]
            unlabeled_features, unlabeled_features_tf = torch.chunk(unlabeled_features, 2, dim=0)
            if isinstance(projector, EncoderClusterHead):  # features from encoder
                unlabeled_tf_features = unlabeled_features
            else:
                unlabeled_tf_features = self.transform_unlabeled(
                    unlabeled_features, seed)
            assert unlabeled_tf_features.shape == unlabeled_features_tf.shape, \
                (unlabeled_tf_features.shape, unlabeled_features_tf.shape)
            prob1, prob2 = list(
                zip(*[torch.chunk(x, 2, 0) for x in projector(
                    torch.cat([unlabeled_tf_features, unlabeled_features_tf], dim=0)
                )])
            )
            _iic_loss_list = [criterion(x, y) for x, y in zip(prob1, prob2)]
            _iic_loss = average_iter(_iic_loss_list)
            iic_losses_for_features.append(_iic_loss)
        reg_loss = weighted_average_iter(iic_losses_for_features, self.feature_importance)
        return reg_loss

    def training(self):
        if self.load :
            self.load_model(9)

        for epoch in range(self.max_epoch):
            train_sup_loss, train_reg_loss, train_total_loss = self.run_train_epoch()
            with torch.no_grad():
                val_sup_loss, val_reg_loss, val_total_loss = self.run_val_epoch()

            print(f"Epoch {epoch} : ")
            print("Train : ")
            print(f"Labeled Loss : {train_sup_loss}, Unlabeled loss : {train_reg_loss}, Total : {train_total_loss}")
            print("Evaluation :")
            print(f"Labeled Loss : {val_sup_loss}, Unlabeled loss : {val_reg_loss}, Total : {val_total_loss}")
            print(repr(self.metric))
            if epoch % self.num_epoch_record == 0 :
                self.dict_record['epoch'][epoch] = {
                    'Train':{
                        'Labeled Loss' : train_sup_loss, 
                        'Unlabeled loss' : train_reg_loss, 
                        'Total' : train_total_loss
                    }, 
                    'Evaluation':{
                        'Labeled Loss' : train_sup_loss, 
                        'Unlabeled loss' : train_reg_loss, 
                        'Total' : train_total_loss
                    },
                    'Metric': self.metric.summary()
                }
                try:
                    os.makedirs(f"checkpoints/model_exp_{self.name_experiment}")
                except FileExistsError:
                    pass
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'stats' : self.dict_record['epoch'][epoch],
                }, f'checkpoints/model_exp_{self.name_experiment}/epoch_{epoch}.pt')
    
        #create output path
        try:
            os.makedirs("results")
        except FileExistsError:
            pass
        save_dict_to_json(self.dict_record, f'results/state_dict_exp_{self.name_experiment}.json')
    
    def load_model(self, epoch) :
        print('loading')
        checkpoint = torch.load('epoch_9.pt')
        #torch.load(f'/checkpoints/model_exp_{self.name_experiment}/epoch_{epoch}.pt') 
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def inference(self) : 
        self.load_model(6)
        submissions = pd.DataFrame(columns=[f'Pixel {i}' for i in range(512*512)])
        
        dataset = DatasetTest()
        with torch.no_grad():
            for sample in dataset:
                slice, path = sample
                slice = slice[None, :, :, :].to(self.device)
                name_file = path.split('\\')[-1]
                # Creating prediction for unlabeled data
                y_pred = self.model(slice) 
                y_pred = self.meanshift(y_pred).cpu().numpy().flatten().astype(np.uint8)
                submissions.loc[name_file] = y_pred.tolist()

        submissions.transpose().to_csv('y_submit_3.csv')

    
    #def inference
 




        
