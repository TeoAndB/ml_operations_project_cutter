# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms
import numpy as np
import torch

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    images_train_l, labels_train_l = [], []

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    #testing set
    path_test = input_filepath + '/corruptmnist/test/'
    images_test_np = np.load(path_test + 'images.npy')
    images_test = transform(images_test_np)
    images_test = torch.reshape(images_test,
                                 shape=(images_test.shape[1], images_test.shape[0], images_test.shape[2]))

    labels_test_np = np.load(path_test + 'labels.npy')
    labels_test = torch.tensor(labels_test_np)

    # training set
    for i in range(5):
        path_train = input_filepath + '/corruptmnist/train_' + str(i) + '/'
        images_i =  np.load(path_train + 'images.npy')
        labels_i = np.load(path_train + 'labels.npy')
        images_train_l.append(images_i)
        labels_train_l.append(labels_i)

    images_train = transform(np.concatenate(images_train_l))
    images_train = torch.reshape(images_train,
                                 shape=(images_train.shape[1], images_train.shape[0], images_train.shape[2]))

    labels_train = torch.tensor(np.concatenate(labels_train_l))

    labels_train, labels_test = labels_train.type(torch.LongTensor), labels_test.type(torch.LongTensor)
    images_train, images_test= images_train.type(torch.FloatTensor), images_test.type(torch.FloatTensor)
    #saving the processsed tensors
    torch.save(images_train, output_filepath + '/images_train.pt')
    torch.save(labels_train, output_filepath + '/labels_train.pt')
    torch.save(images_test, output_filepath + '/images_test.pt')
    torch.save(labels_test, output_filepath + '/labels_test.pt')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
