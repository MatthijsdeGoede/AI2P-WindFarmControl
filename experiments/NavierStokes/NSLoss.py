import torch
from torch import autograd

"""
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
tensor for the specific input
"""

MU = 1.7894e-5  # Dynamic viscosity of air


def get_tau_xx(input, output):
    u_pred, v_pred = output[:, 0], output[:, 1]

    u_x = autograd.grad(u_pred, input, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 0]
    v_y = autograd.grad(v_pred, input, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 1]

    tau = -MU * (2 * u_x - (2 / 3) * (u_x + v_y))
    return tau


def get_tau_yy(input, output):
    u_pred, v_pred = output[:, 0], output[:, 1]

    u_x = autograd.grad(u_pred, input, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 0]
    v_y = autograd.grad(v_pred, input, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 1]

    tau = -MU * (2 * v_y - (2 / 3) * (u_x + v_y))
    return tau


def get_tau_zz(input, output):
    u_pred, v_pred = output[:, 0], output[:, 1]

    u_x = autograd.grad(u_pred, input, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 0]
    v_y = autograd.grad(v_pred, input, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 1]

    tau = -MU * (-(2 / 3) * (u_x + v_y))
    return tau


def get_tau_xy(input, output):
    u_pred, v_pred = output[:, 0], output[:, 1]

    u_y = autograd.grad(u_pred, input, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 1]
    v_x = autograd.grad(v_pred, input, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 0]

    tau = -MU * (u_y + v_x)
    return tau


def get_tau_xz(input, output):
    w_pred = output[:, 2]

    w_x = autograd.grad(w_pred, input, grad_outputs=torch.ones_like(w_pred), create_graph=True)[0][:, 0]

    tau = -MU * w_x
    return tau


def get_tau_yz(input, output):
    w_pred = output[:, 2]

    w_y = autograd.grad(w_pred, input, grad_outputs=torch.ones_like(w_pred), create_graph=True)[0][:, 1]

    tau = -MU * w_y
    return tau


def momentum_x_component(input, output):
    u_pred, v_pred, rho_pred, p_pred = output[:, 0], output[:, 1], output[:, 3], output[:, 4]
    tau_xx, tau_xy = get_tau_xx(input, output), get_tau_xy(input, output)

    u_grad = autograd.grad(u_pred, input, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_t = u_grad[:, 2]
    u_x = u_grad[:, 0]
    u_y = u_grad[:, 1]

    p_x = autograd.grad(p_pred, input, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0][:, 0]
    tau_xx_x = autograd.grad(tau_xx, input, grad_outputs=torch.ones_like(tau_xx), create_graph=True)[0][:, 0]
    tau_xy_y = autograd.grad(tau_xy, input, grad_outputs=torch.ones_like(tau_xy), create_graph=True)[0][:, 1]

    # rho * (u_t + u*u_x + v*u_y) + p_x + tau_xx_x + tau_xy_y = 0
    momentum = rho_pred * (u_t + u_pred * u_x + v_pred * u_y) + p_x + tau_xx_x + tau_xy_y
    momentum_loss = momentum.pow(2).mean()
    return momentum_loss


def momentum_y_component(input, output):
    u_pred, v_pred, rho_pred, p_pred = output[:, 0], output[:, 1], output[:, 3], output[:, 4]
    tau_xy, tau_yy = get_tau_xy(input, output), get_tau_yy(input, output)

    v_grads = autograd.grad(v_pred, input, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
    v_t = v_grads[:, 2]
    v_x = v_grads[:, 0]
    v_y = v_grads[:, 1]

    p_y = autograd.grad(p_pred, input, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0][:, 1]
    tau_xy_x = autograd.grad(tau_xy, input, grad_outputs=torch.ones_like(tau_xy), create_graph=True)[0][:, 0]
    tau_yy_y = autograd.grad(tau_yy, input, grad_outputs=torch.ones_like(tau_yy), create_graph=True)[0][:, 1]

    momentum = rho_pred * (v_t + u_pred * v_x + v_pred * v_y) + p_y + tau_xy_x + tau_yy_y
    momentum_loss = momentum.pow(2).mean()
    return momentum_loss


def momemntum_z_component(input, output):
    u_pred, v_pred, w_pred, rho_pred = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
    tau_yz = get_tau_yz(input, output)

    w_grads = autograd.grad(w_pred, input, grad_outputs=torch.ones_like(w_pred), create_graph=True)[0]
    w_t = w_grads[:, 2]
    w_x = w_grads[:, 0]
    w_y = w_grads[:, 1]

    tau_yz_y = autograd.grad(tau_yz, input, grad_outputs=torch.ones_like(tau_yz), create_graph=True)[0][:, 1]

    momentum = rho_pred * (w_t + u_pred * w_x + v_pred * w_y) + tau_yz_y
    momentum_loss = momentum.pow(2).mean()
    return momentum_loss


def momentum_loss(input, output):
    x_loss = momentum_x_component(input, output)
    y_loss = momentum_y_component(input, output)
    z_loss = momemntum_z_component(input, output)

    total_momentum_loss = x_loss + y_loss + z_loss
    return total_momentum_loss


def continuity_loss(input, output):
    u_pred, v_pred, rho_pred = output[:, 0], output[:, 1], output[:, 3]

    rho_t = autograd.grad(rho_pred, input, grad_outputs=torch.ones_like(rho_pred), create_graph=True)[0][:, 2]
    rho_u_x = autograd.grad(rho_pred * u_pred, input, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 0]
    rho_v_y = autograd.grad(rho_pred * v_pred, input, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0][:, 1]

    # rho_t + (rho*u)_t + (rho*v)_t must equal to 0 according to the continuity equation
    continuity_residual = (rho_t + rho_u_x + rho_v_y).pow(2).mean()
    return continuity_residual


def physics_loss_func(input, output):
    return momentum_loss(input, output) + continuity_loss(input, output)


def data_loss_func(output, target):
    flow_velocity = output[:, [0, 1, 2]]
    return torch.nn.functional.mse_loss(flow_velocity, target)


class NSLoss(torch.nn.Module):
    def __init__(self, physics_coef=1):
        super().__init__()
        self.physics_coef = physics_coef

    def forward(self, input, output, target):
        data = data_loss_func(output, target)
        physics = physics_loss_func(input, output)
        total_loss = data + self.physics_coef * physics

        return total_loss, data, physics
