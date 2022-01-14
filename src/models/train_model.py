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

# Note: Hydra is incompatible with @click
@hydra.main(config_path= PROJECT_PATH / "configs",config_name="/config.yaml")

#
#@click.command()
#@click.argument('input_filepath_data', type=click.Path(exists=True))
#@click.argument('output_filepath_model', type=click.Path(exists=True))
#@click.argument('output_plot_model', type=click.Path())

def main(cfg):
#def main(cfg):
    #logging
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
    '''
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 13
    steps = 0
    running_loss = 0
    running_losses = []
    steps_list = []

    print_every = 100
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                # Model in inference mode, dropout is off

                running_losses.append(running_loss / print_every)
                mean_loss = running_loss / print_every
                log.info('Epoch: {}/{} - Training loss: {:.2f}'.format(e, epochs, mean_loss))

                steps_list.append(steps)

                running_loss = 0

    checkpoint = {'input_size': 784,
                  'output_size': 10,
                  'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, output_filepath_model + "/checkpoint.pth")


    plt.plot(steps_list, running_losses)
    plt.legend()
    plt.title("Training losses")
    plt.show()
    plt.savefig(output_plot_model + '/training_plot.png')
    plt.close()
    '''

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename="model_train.log", level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
    
