import argparse
import sys

import torch

from model import MyAwesomeModel
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from variable import PROJECT_PATH




class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()


    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])

        # TODO: Implement training loop here
        input_size = 784
        output_size = 10
        hidden_layers = [512, 256, 128]
        drop_p = 0.5
        model = MyAwesomeModel(input_size, output_size, hidden_layers, drop_p)
        path_train =PROJECT_PATH / "data" / "processed" / "images_train.pt"

        train_set_images = torch.load(PROJECT_PATH / "data" / "processed" / "images_train.pt")
        train_set_labels = torch.load(PROJECT_PATH / "data" / "processed" / "labels_train.pt")

        train_set = TensorDataset(train_set_images, train_set_labels)

        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
                    print('Epoch: {}/{} - Training loss: {:.2f}'.format(e,epochs,mean_loss))

                    steps_list.append(steps)

                    running_loss = 0

        checkpoint = {'input_size': 784,
                      'output_size': 10,
                      'hidden_layers': [each.out_features for each in model.hidden_layers],
                      'state_dict': model.state_dict()}

        torch.save(checkpoint, PROJECT_PATH / "data" / "models" / "checkpoint.pth")

        plt.plot(steps_list, running_losses)
        plt.legend()
        plt.title("Training losses")
        plt.show()
        plt.savefig('../reports/figures/' + 'training_plot.png')
        plt.close()

    def evaluate(self):

        def load_checkpoint(filepath):
            checkpoint = torch.load(filepath)
            model = MyAwesomeModel(checkpoint['input_size'],
                                     checkpoint['output_size'],
                                     checkpoint['hidden_layers'])
            model.load_state_dict(checkpoint['state_dict'])

            return model

        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])

        # TODO: Implement evaluation logic here
        criterion = nn.NLLLoss()
        model = load_checkpoint('../models/' + args.load_model_from)
        test_set_images = torch.load(path=PROJECT_PATH / "data" / "processed" / "images_test.pt")
        test_set_labels = torch.load(path=PROJECT_PATH / "data" / "processed" / "labels_test.pt")

        test_set = TensorDataset(test_set_images, test_set_labels)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
        accuracy = 0

        test_loss = 0
        for images, labels in testloader:
            images = images.resize_(images.size()[0], 784)

            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ## Calculating the accuracy
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()
        print('Test accuracy {:.2f}%'.format(accuracy*100/len(testloader)))
        #print(f'Accuracy: {accuracy * 100 / len(testloader)}%')



if __name__ == '__main__':
    TrainOREvaluate()
