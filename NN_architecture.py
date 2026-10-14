import torch 
import torch.nn as nn 
import torch.optim as optim

# This class works as 
class PINNModel(nn.Module) : 
    def __init__( self, input_dim, hidden_dim, num_hidden_layers, output_dim ) :
        super(PINNModel, self).__init__()

        """
        Function initializes the PINN class.
        The variables that are taken as input are the input dimension, which corresponds to 
        the number of input variables for the model (15 in our case) , the hidden dimension,
        which corresponds to the number of neurons per layer, the number of hidden layers and 
        the output dimension (which is one in our case)

        """

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # Activation function - we'll use tanh for smoothness
        self.activation = nn.Tanh()

    def forward(self, x):

        """
        Function to have the forward pass during training in the 
        Network.
        
        """
        # Forward pass through the input layer
        x = self.activation(self.input_layer(x))
        
        # Forward pass through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Output layer (without activation, typically regression doesn't need activation here)
        x = self.output_layer(x)
        
        return x
