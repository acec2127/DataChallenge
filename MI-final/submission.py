from unet import UNet
from data import CTDataset

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_model(path):
    model = UNet()
    model.load_state_dict(torch.load(path))
    return model


def create_submission_file(model, dataset, path):
    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if cuda:
        model = model.cuda()

    submissions = pd.DataFrame(columns=[f'Pixel {i}' for i in range(512*512)])

    with torch.no_grad():
        for sample in tqdm(dataset):
            slice = sample['slice'].type(Tensor)[None, :, :, :]
            name_file = sample['data_path'].split('\\')[-1]

            # Creating prediction for unlabeled data
            y_pred = model(slice)['seg'].cpu().numpy().flatten().astype(np.uint8)
            submissions.loc[name_file] = y_pred.tolist()

    submissions.transpose().to_csv(path)


if __name__ == '__main__':
    # Loading trained model
    model = load_model('checkpoints/10_0.5_0.1_0.2/model.pt')
    dataset = CTDataset('test')
    path = 'submissions/y_submit_2.csv'

    create_submission_file(model, dataset, path)



