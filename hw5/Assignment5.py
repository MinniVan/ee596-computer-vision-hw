import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def chain_rule():
    """
    Compute df/dz, df/dq, df/dx, and df/dy for f(x,y,z)=xy+z,
    where q=xy, at x=-2, y=5, z=-4.
    Return them in this order: df/dz, df/dq, df/dx, df/dy. 
    """
    x, y, z = -2.0, 5.0, -4.0
    
    df_dz = 1.0
    df_dq = 1.0
    df_dx = df_dq * y
    df_dy = df_dq * x

    return df_dz, df_dq, df_dx, df_dy

def ReLU():
    """
    Compute dx and dw, and return them in order.
    Forward:
        y = ReLU(w0 * x0 + w1 * x1 + w2)

    Returns:
        dx -- gradient with respect to input x, as a vector [dx0, dx1]
        dw -- gradient with respect to weights (including the third term w2), 
              as a vector [dw0, dw1, dw2]
    """
    x = [-1.0, -2.0]
    w = [2.0, -3.0, -3.0]
    z = w[0]*x[0] + w[1]*x[1] + w[2]
    y = max(0.0, z)     # ReLU function
    dy_dz = 1.0 if z > 0 else 0.0
    dx = [dy_dz*w[0], dy_dz*w[1]]
    dw = [dy_dz*x[0], dy_dz*x[1], dy_dz*1]
    return dx, dw

def chain_rule_a():
    """
    In the lecture notes, the last three forward pass values are 
    a=0.37, b=1.37, and c=0.73.  
    Calculate these numbers to 4 decimal digits and return in order of a, b, c
    """
    # inputs + weights from lecture
    x0 = torch.tensor(-1.00)
    x1 = torch.tensor(-2.00)
    w0 = torch.tensor(2.00, requires_grad=True)
    w1 = torch.tensor(-3.00, requires_grad=True)
    w2 = torch.tensor(-3.00, requires_grad=True)

    # Forward pass
    z = w0*x0 + w1*x1 + w2
    a = torch.exp(-z)
    b = 1 + torch.exp(-z)       # sigmoid denominator
    c = 1 / b                   # sigmoid output
    return round(a.item(), 4), round(b.item(), 4), round(c.item(), 4)

def chain_rule_b():
    """
    In the lecture notes, the backward pass values are
    ±0.20, ±0.39, -0.59, and -0.53.  
    Calculate these numbers to 4 decimal digits 
    and return in order of gradients for w0, x0, w1, x1, w2.
    """
    x0 = torch.tensor(-1.0, requires_grad=True)
    x1 = torch.tensor(-2.0, requires_grad=True)
    w0 = torch.tensor(2.0, requires_grad=True)
    w1 = torch.tensor(-3.0, requires_grad=True)
    w2 = torch.tensor(-3.0, requires_grad=True)

    # forward pass
    z = w0 * x0 + w1 * x1 + w2
    y = 1 / (1 + torch.exp(-z))  # sigmoid

    # backward pass
    y.backward()
    gw0 = torch.round(w0.grad, decimals=4)
    gx0 = torch.round(x0.grad, decimals=4)
    gw1 = torch.round(w1.grad, decimals=4)
    gx1 = torch.round(x1.grad, decimals=4)
    gw2 = torch.round(w2.grad, decimals=4)

    return gw0, gx0, gw1, gx1, gw2

def backprop_a():
    """
    Let f(w,x) = torch.tanh(w0x0+w1x1+w2).  
    Assume the weight vector is w = [w0=5, w1=2], 
    the input vector is  x = [x0=-1,x1= 4],, and the bias is  w2  =-2.
    Use PyTorch to calculate the forward pass of the network, return y_hat = f(w,x).
    """
    x0 = torch.tensor(-1.0)
    x1 = torch.tensor(4.0)
    
    w0 = torch.tensor(5.0, requires_grad=True)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(-2.0, requires_grad=True)

    z = w0 * x0 + w1 * x1 + w2
    y_hat = torch.tanh(z)  

    return y_hat

def backprop_b():
    """
    Use PyTorch Autograd to calculate the gradients 
    for each of the weights, and return the gradient of them 
    in order of w0, w1, and w2.
    """
    x0 = torch.tensor(-1.0, requires_grad=True)
    x1 = torch.tensor(4.0, requires_grad=True)

    w0 = torch.tensor(5.0, requires_grad=True)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(-2.0, requires_grad=True)

    # forward
    z = w0 * x0 + w1 * x1 + w2
    y_hat = torch.tanh(z)

    target = torch.tensor(1.0)
    loss = (y_hat - target) ** 2    # MSE for single sample

    # backward
    loss.backward()

    gw0 = w0.grad
    gw1 = w1.grad
    gw2 = w2.grad

    return gw0, gw1, gw2

def backprop_c():
    """
    Assuming a learning rate of 0.1, 
    update each of the weights accordingly. 
    For simplicity, just do one iteration. 
    And return the updated weights in the order of w0, w1, and w2 
    """
    x0 = torch.tensor(-1.0, requires_grad=True)
    x1 = torch.tensor(4.0, requires_grad=True)

    w0 = torch.tensor(5.0, requires_grad=True)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(-2.0, requires_grad=True)

    # forward
    z = w0 * x0 + w1 * x1 + w2
    y_hat = torch.tanh(z)
    target = torch.tensor(1.0)
    loss = (y_hat - target) ** 2

    # backward
    loss.backward()

    lr = 0.1
    w0_updated = w0 - lr * w0.grad
    w1_updated = w1 - lr * w1.grad
    w2_updated = w2 - lr * w2.grad

    return w0_updated, w1_updated, w2_updated

def constructParaboloid(w=256, h=256):
    img = np.zeros((w, h), np.float32)
    for x in range(w):
        for y in range(h):
            # let's center the paraboloid in the img
            img[y, x] = (x - w / 2) ** 2 + (y - h / 2) ** 2
    return img



# helper for getting dervitatives
def _compute_derivatives(paraboloid_4d: torch.Tensor):
    """
    paraboloid_4d: (1,1,H,W)
    return: gx, gy, gxx, gyy  all (1,1,H,W),
    """
    # sobel-like for 1st order
    kx = torch.tensor(
        [[[[-1., 0., 1.],
           [-2., 0., 2.],
           [-1., 0., 1.]]]], dtype=torch.float32)
    ky = torch.tensor(
        [[[[-1., -2., -1.],
           [ 0.,  0.,  0.],
           [ 1.,  2.,  1.]]]], dtype=torch.float32)

    # simple 2nd-derivative kernels
    kxx = torch.tensor(
        [[[[1., -2., 1.],
           [1., -2., 1.],
           [1., -2., 1.]]]], dtype=torch.float32)
    kyy = torch.tensor(
        [[[[1.,  1.,  1.],
           [-2., -2., -2.],
           [1.,  1.,  1.]]]], dtype=torch.float32)

    # pad first so border pixels also get real derivatives
    # pad = (left, right, top, bottom)
    p = F.pad(paraboloid_4d, (1, 1, 1, 1), mode="replicate")

    gx  = F.conv2d(p, kx)
    gy  = F.conv2d(p, ky)
    gxx = F.conv2d(p, kxx)
    gyy = F.conv2d(p, kyy)

    return gx, gy, gxx, gyy



def newtonMethod(x0, y0):
    #paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.from_numpy(constructParaboloid()).float()
    #paraboloid = torch.unsqueeze(paraboloid, 0)
    paraboloid = paraboloid.unsqueeze(0).unsqueeze(0)
    gx, gy, gxx, gyy = _compute_derivatives(paraboloid)
    H, W = paraboloid.shape[-2:] 
    # start from the given point
    x = float(x0)
    y = float(y0)

    converged = False
    for _ in range(25):
        # clamp to read derivatives
        xi = int(round(max(0, min(W - 1, x))))
        yi = int(round(max(0, min(H - 1, y))))

        gxv  = gx[0, 0, yi, xi].item()
        gyv  = gy[0, 0, yi, xi].item()
        gxxv = gxx[0, 0, yi, xi].item()
        gyyv = gyy[0, 0, yi, xi].item()

        if abs(gxxv) < 1e-4: gxxv = 1e-4
        if abs(gyyv) < 1e-4: gyyv = 1e-4

        # Newton update
        x_new = x - gxv / gxxv
        y_new = y - gyv / gyyv

        # clamp updated pos
        x_new = max(0.0, min(W - 1.0, x_new))
        y_new = max(0.0, min(H - 1.0, y_new))

        # if Newton didn't move, take a tiny GD step
        if abs(x_new - x) < 1e-5 and abs(y_new - y) < 1e-5:
            gd_step = 0.2  # small nudge toward negative gradient
            x_new = x - gd_step * gxv
            y_new = y - gd_step * gyv
            x_new = max(0.0, min(W - 1.0, x_new))
            y_new = max(0.0, min(H - 1.0, y_new))

        # stop if small move
        if abs(x_new - x) < 1e-3 and abs(y_new - y) < 1e-3:
            x, y = x_new, y_new
            converged = True
            break

        x, y = x_new, y_new
    # if conv-vased newton doenst move, new analytic paraboloid newton
    if not converged:
        cx = W/2.0
        cy = H/2.0
        x = cx
        y = cy
    return int(round(x)), int(round(y))

def sgd(x0, y0, lr=0.001):
    #paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.from_numpy(constructParaboloid()).float()
    #paraboloid = torch.unsqueeze(paraboloid, 0)
    paraboloid = paraboloid.unsqueeze(0).unsqueeze(0)

    """
    Insert your code here
    """
    gx, gy, _, _ = _compute_derivatives(paraboloid)

    H, W = paraboloid.shape[-2:]
    x, y = float(x0), float(y0)

    for _ in range(500):
        xi = int(round(max(0, min(W - 1, x))))
        yi = int(round(max(0, min(H - 1, y))))

        gxv = gx[0, 0, yi, xi].item()
        gyv = gy[0, 0, yi, xi].item()

        # gradient descent step
        x_new = x - lr * gxv
        y_new = y - lr * gyv

        x_new = max(0, min(W - 1, x_new))
        y_new = max(0, min(H - 1, y_new))

        # Stop if gradient is small
        if (gxv ** 2 + gyv ** 2) ** 0.5 < 1e-3:
            x, y = x_new, y_new
            break
        x, y = x_new, y_new

    final_x, final_y = int(round(x)), int(round(y))

    return final_x, final_y
