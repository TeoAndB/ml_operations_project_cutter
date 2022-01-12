# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms
import numpy as np
import torch
import argparse
import sys
import logging
from pytorch_lightning import Trainer

from model import MyAwesomeModel
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import hydra
from variable import PROJECT_PATH
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# Note: Hydra is incompatible with @click
@hydra.main(config_path=PROJECT_PATH / "configs", config_name="/config.yaml")

def main(cfg):
    # def main(cfg):
    # logging
    log = logging.getLogger(__name__)

    input_size = 784
    output_size = 10

    # for hydra parameters
    input_filepath_data = str(PROJECT_PATH / "data" / "processed")
    output_filepath_model = str(PROJECT_PATH / "models")
    output_plot_model = str(PROJECT_PATH / "reports" / "figures")
    '''
    hidden_layers = [512, 256, 128]
    drop_p = 0.5
    batch_size = 64
    learning_rate = 0.003
    '''
    hidden_layers = cfg.hyperparameters.hidden_layers
    drop_p = cfg.hyperparameters.drop_p
    batch_size = cfg.hyperparameters.batch_size
    learning_rate = cfg.hyperparameters.learning_rate

    # training set
    train_set_images = torch.load(input_filepath_data + "/images_train.pt")
    train_set_labels = torch.load(input_filepath_data + "/labels_train.pt")
    train_set = TensorDataset(train_set_images, train_set_labels)

    # validation set
    validation_set_images = torch.load(input_filepath_data + "/images_test.pt")
    validation_set_labels = torch.load(input_filepath_data + "/labels_test.pt")

    validation_set = TensorDataset(validation_set_images, validation_set_labels)

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=True)

    # training ###########################################
    model = MyAwesomeModel()

    early_stopping_callback = EarlyStopping(
        monitor="loss", patience=3, verbose=True, mode="min"
    )
    trainer = Trainer(callbacks=[early_stopping_callback])

    trainer.fit(model, train_loader, val_loader)

    preds, target = [],[]

    for batch in train_loader:
        x, y = batch
        x = x.resize_(x.size()[0], 784)
        probs = model(x)
        preds.append(probs.argmax(dim=-1))
        target.append(y.detach())

    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)

    report = classification_report(target, preds)
    with open("classification_report.txt", 'w') as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confmat, )
    disp.plot()
    plt.savefig('confusion_matrix.png')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename="model_train.log", level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

