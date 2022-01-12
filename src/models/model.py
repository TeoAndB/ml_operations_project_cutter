from torch import nn
import torch.nn.functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule

class MyAwesomeModel(LightningModule):

    def __init__(self):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers

        '''

        input_size = 784
        output_size = 10
        hidden_layers = [512, 256, 128]
        drop_p = 0.5

        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        data, target = batch
        data.resize_(data.size()[0], 784)
        preds = self(data)
        loss = self.criterium(preds, target)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        learning_rate = 1e-2
        return optim.Adam(self.parameters(), lr=learning_rate)
