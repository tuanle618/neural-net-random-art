# -*- coding: utf-8 -*-
"""
@title: random_art.py
@author: Tuan Le
@email: tuanle@hotmail.de

Minimal image pre-processing script
"""

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Pool


def process_xy_el_meshgrid(xv: np.ndarray,
                           yv: np.ndarray,
                           symmetry: bool,
                           trig: bool,
                           z1: float,
                           z2: float) -> torch.Tensor:
    """
    Transforms the input tuple (x,y)
    Args:
        xv: numpy array (width, height)
        yv: (width, height)
        symmetry: boolean, if x and y should be symmetric, i.e. squaring the tuple elements
        trig: if the input tuples should be transformed using the cosine and sine function. Note this returns more
              variability, for the z1 and z2 values, respectively.
        z1: deterministic factor z1, if trig is set to False.
        z2: deterministic factor z2, if trig is set to False.

    Returns: np.array with 5 columns, for the values x,y,r,z1,z2
    """
    # radius part r
    if symmetry:
        xv = xv**2
        yv = yv**2
    r_ = np.sqrt(xv**2 + yv**2)
    # z1, z2 part
    if trig:
        z1_ = np.cos(z1*xv)
        z2_ = np.sin(z2*yv)
    else:
        z1_ = np.empty_like(xv, dtype=np.float32)
        z1_.fill(z1)
        z2_ = np.empty_like(yv, dtype=np.float32)
        z2_.fill(z2)

    x_ = np.expand_dims(xv.T.flatten(), axis=1)
    y_ = np.expand_dims(yv.T.flatten(), axis=1)
    r_ = np.expand_dims(r_.T.flatten(), axis=1)
    z1_ = np.expand_dims(z1_.T.flatten(), axis=1)
    z2_ = np.expand_dims(z2_.T.flatten(), axis=1)

    # create flattened image
    res = np.concatenate([x_, y_, r_, z1_, z2_], axis=1)
    return torch.from_numpy(res).float()


def init_data(img_height: int = 500,
              img_width: int = 700,
              symmetry: bool = False,
              trig: bool = True,
              z1: float = -0.618, z2: float = 0.618,
              noise: float = False,
              noise_std: float = 0.01):
    factor = min(img_height, img_width)
    # get input: x,y,r,z_1,z_2
    x = [(i/factor-0.5)*2 for i in range(img_height)]
    y = [(j/factor-0.5)*2 for j in range(img_width)]

    xv, yv = np.meshgrid(x, y)
    in_data = process_xy_el_meshgrid(xv, yv, symmetry, trig, z1, z2)
    if noise:
        in_data += torch.zeros_like(in_data).normal_(mean=0, std=noise_std)
    return in_data


def hsv_to_rgb(h, s, v):
    # hsw are between 0 and 1
    # returns rgb between 0 and 1
    # from: https://bgrins.github.io/TinyColor/docs/tinycolor.html
    h *= 6
    i = np.floor(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    mod = int(i % 6)
    r = [v, q, p, p, t, v][mod]
    g = [t, v, v, q, p, p][mod]
    b = [p, p, t, v, v, q][mod]
    return r, g, b


def hsv_to_rgb_torch(img: torch.Tensor) -> torch.Tensor:
    # hsw are between 0 and 1
    # returns rgb between 0 and 1
    # from: https://bgrins.github.io/TinyColor/docs/tinycolor.html
    h_ = img[:, :, 0].view(img.size(0) * img.size(1)).detach().data.numpy()
    s_ = img[:, :, 1].view(img.size(0) * img.size(1)).detach().data.numpy()
    v_ = img[:, :, 2].view(img.size(0) * img.size(1)).detach().data.numpy()
    h = 6 * h_
    i = np.floor(h)
    f = h - i
    p = (v_ * (1 - s_))
    q = (v_ * (1 - f * s_))
    t = (v_ * (1 - (1 - f) * s_))
    mod = [int(a % 6) for a in i]
    r_select = torch.Tensor([[v_[i], q[i], p[i], p[i], t[i], v_[i]][m] for i, m in enumerate(mod)]).view(img.size(0),
                                                                                                         img.size(
                                                                                                             1)).unsqueeze(
        -1)
    g_select = torch.Tensor([[t[i], v_[i], v_[i], q[i], p[i], p[i]][m] for i, m in enumerate(mod)]).view(img.size(0),
                                                                                                         img.size(
                                                                                                             1)).unsqueeze(
        -1)
    b_select = torch.Tensor([[p[i], p[i], t[i], v_[i], v_[i], q[i]][m] for i, m in enumerate(mod)]).view(img.size(0),
                                                                                                         img.size(
                                                                                                             1)).unsqueeze(
        -1)
    img = torch.cat([r_select, g_select, b_select], dim=-1)

    return img


def hue_to_rgb(p, q, t):
    if t < 0 or t > 1:
        return p
    if t < 1 / 6:
        return p + (q - p) * 6 * t
    if t < 1 / 2:
        return q
    if t < 2 / 3:
        return p + (q - p) * (2 / 3 - t) * 6
    else:
        return p


def hsl_to_rgb(h, s, l):
    # hsl are between 0 and 1
    # returns rgb between 0 and 1
    if s == 0:
        r = g = b = l  # achromatic
    else:
        if l < 0.5:
            q = l * (1 + s)
        else:
            q = l + s - l * s

        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1 / 3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1 / 3)

    return r, g, b


def hsl_to_rgb_torch(h: torch.Tensor, s: torch.Tensor, l: torch.Tensor):
    h = h.cpu().data.numpy()
    s = s.cpu().data.numpy()
    l = l.cpu().data.numpy()
    with Pool(processes=mp.cpu_count() - 2) as pool:
        proc_img = pool.starmap(hsl_to_rgb, zip(h, s, l))
    proc_img = torch.Tensor(proc_img)
    return proc_img


def transform_colors(img: torch.Tensor, colormode: str, alpha: bool):
    if alpha:
        alpha_tensor = img[:, :, -1]
        # Since non blackmode [rgb, cmyk, hsv, hsl] values are mapped onto [0,1] the alpha channel is also between [0,1].
        # 0=transparency, 1=opaque wrt. to overlaying
        a = 1 - torch.abs(2 * alpha_tensor - 1)
        a = (0.25 + 0.75 * a).unsqueeze(-1)
    else:
        a = torch.ones(size=(img.size(0), img.size(1))).unsqueeze(-1)

    if colormode == "rgb":  # Output via sigmoid activation mapped into range [0,1]
        proc_img = img[:, :, 0:2]
    elif colormode == "bw":
        proc_img = torch.cat([img[:, :, 0].unsqueeze(-1)] * 3, dim=-1)
    elif colormode == "cmyk":
        r = ((1 - img[:, :, 0]) * img[:, :, 3]).unsqueeze(-1)
        g = ((1 - img[:, :, 1]) * img[:, :, 3]).unsqueeze(-1)
        b = ((1 - img[:, :, 2]) * img[:, :, 3]).unsqueeze(-1)
        proc_img = torch.cat([r, g, b], dim=-1).to(img.device)
    elif colormode == "hsv":
        proc_img = hsv_to_rgb_torch(img)
    elif colormode == "hsl":
        h = img[:, :, 0].view(img.size(0) * img.size(1))
        s = img[:, :, 1].view(img.size(0) * img.size(1))
        l = img[:, :, 2].view(img.size(0) * img.size(1))
        proc_img = hsl_to_rgb_torch(h, s, l).to(img.device)
        proc_img = proc_img.view(img.size(0), img.size(1), 3)
    else:
        print("Inserted colormode '{}' is not part ob supported ones: [rgb, bw, cmyk, hsv, hsl]".format(colormode))
        raise Exception("Non-supported colormode {}".format(colormode))

    res = torch.cat([proc_img, a], dim=-1)

    return res