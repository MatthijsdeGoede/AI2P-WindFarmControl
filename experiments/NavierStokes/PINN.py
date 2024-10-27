import torch

class PINN(torch.nn.Module):
    def __init__(self, in_dimensions=3, hidden_size=64, nr_hidden_layer=5, out_dimensions=5, relu_activation=False):
        super().__init__()

        self.input = torch.nn.Linear(in_dimensions, hidden_size)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(nr_hidden_layer)])
        self.output = torch.nn.Linear(hidden_size, out_dimensions)

        self.activation = torch.nn.ReLU() if relu_activation else torch.nn.SiLU()

    def forward(self, coords):
        # Input: x, y, z, t
        coords = coords.clone().detach().requires_grad_(True) 

        x = self.activation(self.input(coords))

        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        x = self.output(x)
        
        #Output (incompressible): u, v, w, p
        return x, coords