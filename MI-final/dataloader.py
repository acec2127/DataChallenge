import random

from torch.utils.data.dataloader import default_collate

from torch.utils.data import DataLoader, RandomSampler

from dataset import DatasetLabeled, DatasetUnlabeled
from sampler import InfiniteRandomSampler
from augment import TransformLabeled, TransformUnlabeled, TransformPostUnlabeled

def get_dataloaders(val_ratio, batch_size, n_geo, n_color, magnitude, num_workers):
    seed_labeled_train = int(random.randint(0, int(2 ** 32)))
    seed_labeled_val = int(random.randint(0, int(2 ** 32)))
    seed_train_val_split_unlabeled_train = int(random.randint(0, int(2 ** 32)))
    seed_train_val_split_unlabeled_val = int(random.randint(0, int(2 ** 32)))

    transform_labeled_train= TransformLabeled(n_geo, n_color, magnitude, 'train')
    transform_labeled_val= TransformLabeled(n_geo, n_color, magnitude, 'val')
    transform_unlabeled= TransformUnlabeled(n_geo, n_color, magnitude)
    transform_post_unlabeled = TransformPostUnlabeled(n_geo, n_color, magnitude)

    labeled_train = DatasetLabeled('train', val_ratio, seed_labeled_train, transform_labeled_train)
    labeled_val = DatasetLabeled('val', val_ratio, seed_labeled_val, transform_labeled_val)
    unlabeled_train = DatasetUnlabeled('train', val_ratio, seed_train_val_split_unlabeled_train, transform_unlabeled)
    unlabeled_val = DatasetUnlabeled('val', val_ratio, seed_train_val_split_unlabeled_val, transform_unlabeled)

    # labeled loader is with normal 2d slicing and InfiniteRandomSampler
    labeled_train_loader = DataLoader(
        labeled_train, sampler=RandomSampler(
            labeled_train,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        #collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)), 
        pin_memory=True
    )

    unlabeled_train_loader = DataLoader(
        unlabeled_train, sampler=InfiniteRandomSampler(
            unlabeled_train,
            shuffle=True
        ),
        batch_size=batch_size // 2,
        num_workers=num_workers,
        #collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)), 
        pin_memory=True
    )

    labeled_val_loader = DataLoader(
        labeled_val, sampler=RandomSampler(
            labeled_val,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        #collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)), 
        pin_memory=True
    )
    
    unlabeled_val_loader = DataLoader(
        unlabeled_val, sampler=InfiniteRandomSampler(
            unlabeled_val,
            shuffle=True
        ),
        batch_size=batch_size // 2,
        num_workers=num_workers,
        #collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)), 
        pin_memory=True
    )
    return labeled_train_loader, unlabeled_train_loader,\
          labeled_val_loader, unlabeled_val_loader, transform_post_unlabeled
