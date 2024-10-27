import torch
from torch import autograd

"""
INCOMPRESSIBLE NAVIER-STOKES EQUATIONS

The following Navier-Stokes equations assume the following:
$$
\frac{\partial w}{\partial z} = 0
$$

This simplification assumes that the flow is essentially 2D at the slice level, 
with no vertical velocity gradient. This is the best estimate we can do as we 
use a 2D horizontal slice at only one level. Therefore, z gradients are omitted from calculations.

Equations taken from: https://texmex.mit.edu/pub/emanuel/CLASS/12.340/navier-stokes%282%29.pdf


Apparently slicing the input is a big no no for autograd.grad. Therefore, we 
calculate gradient of an output w.r.t. to all input, then slice the gradient 
tensor for the specific input.

Turns out if we rescale our inputs (time and meters), we also have to rescale our coefficients in our physics equations.
"""

MU = 1.7894e-5 #Dynamic viscosity of air (kg/m-s)
DENSITY = 1.2041 #Atmospheric air density at 20 celsius (kg/m3)

class NSLoss(torch.nn.Module):
    def __init__(self, physics_coef=1):
        super().__init__()
        self.physics_coef = physics_coef

    
    def forward(self, input, output, target):
        data = self.data_loss_func(output, target)
        physics = self.physics_loss_func(input, output) * self.physics_coef
        total_loss = data + physics
        
        return  total_loss, data, physics

    
    def momentum_x_component(self, output, grads):
        u_pred, v_pred = output[:, 0], output[:, 1]

        u_grad = grads["u_grad"]
        u_t = u_grad[:, 2]
        u_x = u_grad[:, 0]
        u_y = u_grad[:, 1]

        p_x = grads["p_grad"][:, 0]

        u_hessian = grads["u_hessian"]
        u_xx = u_hessian[:, 0]
        u_yy = u_hessian[:, 1]

        #rho * (u_t + u*u_x + v*u_y) + p_x - MU (u2_x2 + u2_y2) = 0
        momentum = DENSITY * (u_t + u_pred*u_x + v_pred*u_y) + p_x - MU * (u_xx + u_yy)
        momentum_loss = momentum.pow(2).mean()
        return momentum_loss

    def momentum_y_component(self, output, grads):
        u_pred, v_pred = output[:, 0], output[:, 1]

        v_grads = grads["v_grad"]
        v_t = v_grads[:, 2]
        v_x = v_grads[:, 0]
        v_y = v_grads[:, 1]

        p_y = grads["p_grad"][:, 1]

        v_hessian = grads["v_hessian"]
        v_xx = v_hessian[:, 0]
        v_yy = v_hessian[:, 1]

        momentum = DENSITY * (v_t + u_pred * v_x + v_pred * v_y) + p_y - MU * (v_xx + v_yy)
        momentum_loss = momentum.pow(2).mean()
        return momentum_loss

    def momemntum_z_component(self, output, grads):
        u_pred, v_pred = output[:, 0], output[:, 1]

        w_grads = grads["w_grad"]
        w_t = w_grads[:, 2]
        w_x = w_grads[:, 0]
        w_y = w_grads[:, 1]

        w_hessian = grads["w_hessian"]
        w_xx = w_hessian[:, 0]
        w_yy = w_hessian[:, 1]

        momentum = DENSITY * (w_t + u_pred * w_x + v_pred * w_y) - MU * (w_xx + w_yy)
        momentum_loss = momentum.pow(2).mean()
        return momentum_loss

    def momentum_loss(self, output, grads):
        x_loss = self.momentum_x_component(output, grads)
        y_loss = self.momentum_y_component(output, grads)
        z_loss = self.momemntum_z_component(output, grads)

        total_momentum_loss = x_loss + y_loss + z_loss
        return total_momentum_loss

    def physics_loss_func(self, input, output):
        grads = self.calculate_grads(input, output)
        return self.momentum_loss(output, grads)

    def data_loss_func(self, output, target):
        flow_velocity = output[:, [0, 1, 2]]
        return torch.nn.functional.mse_loss(flow_velocity, target)

    def calculate_grads(self, input, output):
        "Output: u, v, w, pressure"
        u_pred, v_pred, w_pred, p_pred = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

        u_grad = autograd.grad(u_pred, input, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, :3]
        v_grad = autograd.grad(v_pred, input, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, :3]
        w_grad = autograd.grad(w_pred, input, grad_outputs=torch.ones_like(w_pred), create_graph=True)[0][:, :3]
        p_grad = autograd.grad(p_pred, input, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0][:, :3]

        #These are technicaly not hessians but it's close
        u_hessian = autograd.grad(u_grad, input, grad_outputs=torch.ones_like(u_grad), create_graph=True)[0][:, :3]
        v_hessian = autograd.grad(v_grad, input, grad_outputs=torch.ones_like(v_grad), create_graph=True)[0][:, :3]
        w_hessian = autograd.grad(w_grad, input, grad_outputs=torch.ones_like(v_grad), create_graph=True)[0][:, :3]

        grads = {"u_grad": u_grad,
                "v_grad": v_grad,
                "w_grad": w_grad,
                "p_grad": p_grad,
                "u_hessian": u_hessian,
                "v_hessian": v_hessian,
                "w_hessian": w_hessian,
                }

        return grads