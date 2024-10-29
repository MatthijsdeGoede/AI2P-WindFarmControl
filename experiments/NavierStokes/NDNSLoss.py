import torch
from torch import autograd

"""
NON DIMESIONAL INCOMPRESSIBLE NAVIER STOKES EQUATIONS

EULER REGIME ASSUMED (Re -> infinity)

Quick Re calculations gives values in the order of 8-9. 

Here we do not assume that z-derivatives are zero. The model inputs should be: x, y, z, t
"""

class NSLoss(torch.nn.Module):
    def __init__(self, physics_coef=1, time_scaling_factor=1.0):
        """
        Time scaling factor should be the same value used for normalizing time, if normalized after non dimensionalizing.
        The gradients also need to be scaled by the same factor. If unnormalized use 1. The range of the parameter: (0, 1]
        """
        super().__init__()
        self.physics_coef = physics_coef
        self.time_scaling_factor = time_scaling_factor


    def forward(self, input, output, target):
        data = self.data_loss_func(output, target)
        physics = self.physics_loss_func(input, output) * self.physics_coef
        
        total_loss = data + physics
        
        return  total_loss, data, physics

    def momentum_x_component(self, output, grads):
        u_pred, v_pred, w_pred = output[:, 0], output[:, 1], output[:, 2]

        u_grad = grads["u_grad"]

        u_x = u_grad[:, 0]
        u_y = u_grad[:, 1]
        u_z = u_grad[:, 2]
        u_t = u_grad[:, 3] * self.time_scaling_factor

        p_x = grads["p_grad"][:, 0]

        momentum = u_t + u_pred*u_x + v_pred*u_y + w_pred * u_z + p_x

        momentum_loss = momentum.pow(2).mean()
        return momentum_loss

    def momentum_y_component(self, output, grads):
        u_pred, v_pred, w_pred = output[:, 0], output[:, 1], output[:, 2]

        v_grads = grads["v_grad"]
        v_x = v_grads[:, 0]
        v_y = v_grads[:, 1]
        v_z = v_grads[:, 2]
        v_t = v_grads[:, 3] * self.time_scaling_factor

        p_y = grads["p_grad"][:, 1]

        momentum = v_t + u_pred * v_x + v_pred * v_y + w_pred * v_z + p_y
        momentum_loss = momentum.pow(2).mean()
        return momentum_loss

    def momemntum_z_component(self, output, grads):
        u_pred, v_pred, w_pred = output[:, 0], output[:, 1], output[:, 2]

        w_grads = grads["w_grad"]
        w_x = w_grads[:, 0]
        w_y = w_grads[:, 1]
        w_z = w_grads[:, 2]
        w_t = w_grads[:, 3] * self.time_scaling_factor

        p_z = grads["p_grad"][:, 2]

        momentum = w_t + u_pred * w_x + v_pred * w_y + w_pred * w_z + p_z
        momentum_loss = momentum.pow(2).mean()
        return momentum_loss

    def momentum_loss(self, output, grads):
        x_loss = self.momentum_x_component(output, grads)
        y_loss = self.momentum_y_component(output, grads)
        z_loss = self.momemntum_z_component(output, grads)

        total_momentum_loss = x_loss + y_loss + z_loss
        return total_momentum_loss

    def contiunity_loss(self, grads):
        u_x = grads['u_grad'][:, 0]
        v_y = grads['v_grad'][:, 1]
        w_z = grads['w_grad'][:, 2]
        
        continuity = u_x + v_y + w_z
        return continuity.pow(2).mean()

    def physics_loss_func(self, input, output):
        grads = self.calculate_grads(input, output)

        continuity = self.contiunity_loss(grads)
        momentum = self.momentum_loss(output, grads)

        return continuity + momentum

    def calculate_grads(self, input, output):
        "Inputs: x, y, z, t, ..."
        "Output: u, v, w, pressure"
        u_pred, v_pred, w_pred, p_pred = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

        u_grad = autograd.grad(u_pred, input, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, :4]
        v_grad = autograd.grad(v_pred, input, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, :4]
        p_grad = autograd.grad(p_pred, input, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0][:, :4]
        w_grad = autograd.grad(w_pred, input, grad_outputs=torch.ones_like(w_pred), create_graph=True)[0][:, :4]

        grads = {"u_grad": u_grad, "v_grad": v_grad, "w_grad": w_grad, "p_grad": p_grad}

        return grads

    def data_loss_func(self, output, target):
        flow_velocity = output[:, [0, 1, 2]]
        return torch.nn.functional.mse_loss(flow_velocity, target)