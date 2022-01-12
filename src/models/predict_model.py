import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms
import numpy as np
import torch
import argparse
import sys
import helper

from model import MyAwesomeModel
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
#from variable import PROJECT_PATH

@click.command()
@click.argument('input_filepath_model', type=click.Path(exists=True))
@click.argument('input_filepath_data', type=click.Path(exists=True))
@click.argument('output_predictions', type=click.Path())
def main(input_filepath_model, input_filepath_data, output_predictions):

    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = MyAwesomeModel(checkpoint['input_size'],
                               checkpoint['output_size'],
                               checkpoint['hidden_layers'])
        model.load_state_dict(checkpoint['state_dict'])

        return model

    print("Predicting... not like throwing arrows in the dark!")

    # TODO: Implement evaluation logic here
    criterion = nn.NLLLoss()
    model = load_checkpoint(input_filepath_model + '/checkpoint.pth')
    test_set_images = torch.load(input_filepath_data + "/images_test.pt")
    test_set_labels = torch.load(input_filepath_data + "/labels_test.pt")

    test_set = TensorDataset(test_set_images, test_set_labels)

    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    accuracy = 0

    test_loss = 0
    str_for_text_results = ''
    i = 0
    for images, labels in testloader:
        model.eval()
        i+=1
        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        print(ps.max(1)[1])
        #str_out = 'Image batch' + str(i) + 'was classified as ' + ps.max(1)[1]
        #str_for_text_results.append('\nstr_out')
        print('Image {} was classified as {}'.format(i, ps.max(1)[1]))

        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

        # Plotting and saving results:
        img = images[0]
        # Convert 2D image to 1D vector
        img = img.view(1, 784)

        # Calculate the class probabilities (softmax) for img
        with torch.no_grad():
            output = model.forward(img)

        ps = torch.exp(output)

        # Plot the image and probabilities
        helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')
        plt.savefig(output_predictions + '/prediction' + str(i))

    #text_file = open(output_predictions + "/Output.txt", "w")
    #text_file.write(str_for_text_results)
    #text_file.close()

    print('Test accuracy {:.2f}%'.format(accuracy * 100 / len(testloader)))
    # print(f'Accuracy: {accuracy * 100 / len(testloader)}%')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()