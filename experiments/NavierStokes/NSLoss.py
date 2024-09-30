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


MU = 1.7894e-5 #Dynamic viscosity of air

def get_tau_xx(grads):
    u_x = grads["u_grad"][:, 0]
    v_y = grads["v_grad"][:, 1]

    tau = -MU * (2*u_x - (2/3) * (u_x + v_y))
    return tau

def get_tau_yy(grads):
    u_x = grads["u_grad"][:, 0]
    v_y = grads["v_grad"][:, 1]

    tau = -MU * (2*v_y - (2/3) * (u_x + v_y))
    return tau

def get_tau_zz(grads):
    u_x = grads["u_grad"][:, 0]
    v_y = grads["v_grad"][:, 1]

    tau = -MU * (-(2/3) * (u_x + v_y))
    return tau

def get_tau_xy(grads):
        u_y = grads["u_grad"][:, 1]
        v_x = grads["v_grad"][:, 0]

        tau = -MU * (u_y + v_x)
        return tau

def get_tau_xz(grads):
        w_x = grads["w_grad"][:, 0]

        tau = -MU * w_x
        return tau

def get_tau_yz(grads):
        w_y = grads["w_grad"][:, 1]

        tau = -MU * w_y
        return tau

def momentum_x_component(output, grads):
    u_pred, v_pred, rho_pred = output[:, 0], output[:, 1], output[:, 3]

    u_grad = grads["u_grad"]
    u_t = u_grad[:, 2]
    u_x = u_grad[:, 0]
    u_y = u_grad[:, 1]

    p_x = grads["p_grad"][:, 0]
    tau_xx_x = grads["tau_xx_grad"][:, 0]
    tau_xy_y = grads["tau_xy_grad"][:, 1]

    #rho * (u_t + u*u_x + v*u_y) + p_x + tau_xx_x + tau_xy_y = 0
    momentum = rho_pred * (u_t + u_pred*u_x + v_pred*u_y) + p_x + tau_xx_x + tau_xy_y
    momentum_loss = momentum.pow(2).mean()
    return momentum_loss

def momentum_y_component(output, grads):
    u_pred, v_pred, rho_pred = output[:, 0], output[:, 1], output[:, 3]

    v_grads = grads["v_grad"]
    v_t = v_grads[:, 2]
    v_x = v_grads[:, 0]
    v_y = v_grads[:, 1]

    p_y = grads["p_grad"][:, 1]
    tau_xy_x = grads["tau_xy_grad"][:, 0]
    tau_yy_y = grads["tau_yy_grad"][:, 1]

    momentum = rho_pred * (v_t + u_pred * v_x + v_pred * v_y) + p_y + tau_xy_x + tau_yy_y
    momentum_loss = momentum.pow(2).mean()
    return momentum_loss

def momemntum_z_component(output, grads):
    u_pred, v_pred, rho_pred = output[:, 0], output[:, 1], output[:, 3]

    w_grads = grads["w_grad"]
    w_t = w_grads[:, 2]
    w_x = w_grads[:, 0]
    w_y = w_grads[:, 1]

    tau_yz_y = grads["tau_yz_grad"][:, 1]

    momentum = rho_pred * (w_t + u_pred * w_x + v_pred * w_y) + tau_yz_y
    momentum_loss = momentum.pow(2).mean()
    return momentum_loss

def momentum_loss(output, grads):
    x_loss = momentum_x_component(output, grads)
    y_loss = momentum_y_component(output, grads)
    z_loss = momemntum_z_component(output, grads)

    total_momentum_loss = x_loss + y_loss + z_loss
    return total_momentum_loss

def continuity_loss(grads):
    rho_t = grads["rho_grad"][:, 2]
    rho_u_x = grads["rho_u_x"]
    rho_v_y = grads["rho_v_y"]

    #rho_t + (rho*u)_t + (rho*v)_t must equal to 0 according to the continuity equation
    continuity_residual = (rho_t + rho_u_x + rho_v_y).pow(2).mean()
    return continuity_residual

def physics_loss_func(input, output):
    grads = calculate_grads(input, output)
    return momentum_loss(output, grads) + continuity_loss(grads)

def data_loss_func(output, target):
    flow_velocity = output[:, [0, 1, 2]]
    return torch.nn.functional.mse_loss(flow_velocity, target)

def calculate_grads(input, output):
    u_pred, v_pred, w_pred, rho_pred, p_pred = output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4]

    u_grad = autograd.grad(u_pred, input, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    v_grad = autograd.grad(v_pred, input, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
    w_grad = autograd.grad(w_pred, input, grad_outputs=torch.ones_like(w_pred), create_graph=True)[0]
    rho_grad = autograd.grad(rho_pred, input, grad_outputs=torch.ones_like(rho_pred), create_graph=True)[0]
    p_grad = autograd.grad(p_pred, input, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0]

    rho_u_x = autograd.grad(rho_pred * u_pred, input, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 0]
    rho_v_y = autograd.grad(rho_pred * v_pred, input, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 1]

    grads = {"u_grad": u_grad,
             "v_grad": v_grad,
             "w_grad": w_grad,
             "rho_grad": rho_grad,
             "p_grad": p_grad,
             "rho_u_x": rho_u_x,
             "rho_v_y": rho_v_y
             }
    
    tau_xx = get_tau_xx(grads)
    tau_xy = get_tau_xy(grads)
    tau_yy = get_tau_yy(grads)
    tau_yz = get_tau_yz(grads)

    grads["tau_xx_grad"] = autograd.grad(tau_xx, input, grad_outputs=torch.ones_like(tau_xx), create_graph=True)[0]
    grads["tau_xy_grad"] = autograd.grad(tau_xy, input, grad_outputs=torch.ones_like(tau_xy), create_graph=True)[0]
    grads["tau_yy_grad"] = autograd.grad(tau_yy, input, grad_outputs=torch.ones_like(tau_yy), create_graph=True)[0]
    grads["tau_yz_grad"] = autograd.grad(tau_yz, input, grad_outputs=torch.ones_like(tau_yz), create_graph=True)[0]

    return grads
     

class NSLoss(torch.nn.Module):
    def __init__(self, physics_coef=1):
        super().__init__()
        self.physics_coef = physics_coef
    
    def forward(self, input, output, target):
        data = data_loss_func(output, target)
        physics = physics_loss_func(input, output)
        total_loss = data + self.physics_coef * physics
        
        return  total_loss, data, physics