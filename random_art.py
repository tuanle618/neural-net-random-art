# -*- coding: utf-8 -*-
"""
@title: random_art.py
@author: Tuan Le
@email: tuanle@hotmail.de
"""

import warnings
warnings.filterwarnings("ignore")
import argparse
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from multiprocessing import Pool

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 1.0)


class TorchNeuralNet(nn.Module):

    def __init__(self, layers_dims: list = [10, 10, 10, 10, 10], activation_fnc: str = "tanh",
                 colormode="rgb", alpha=True):
        if colormode in ["rgb", "hsv", "hsl"]:  # RGB
            if not alpha:
                out_nodes = 3
            else:
                out_nodes = 4
        elif colormode == "cmyk":
            if not alpha:
                out_nodes = 4
            else:
                out_nodes = 5
        elif colormode == "bw":
            if not alpha:
                out_nodes = 1
            else:
                out_nodes = 2
        super(TorchNeuralNet, self).__init__()
        self.input_layer= nn.Linear(in_features=5, out_features=layers_dims[0])
        self.output_layer = nn.Linear(in_features=layers_dims[-1], out_features=out_nodes)
        self.layers = nn.ModuleList([self.input_layer] +
                                    [nn.Linear(in_features=layers_dims[i],
                                               out_features=layers_dims[i+1])
                                     for i in range(len(layers_dims)-1)] +
                                    [self.output_layer]
                                    )

        self.activation = self.init_activation_fnc(activation_fnc.lower())

    def init_activation_fnc(self, a):
        if a == "tanh":
            return nn.Tanh()
        elif a == "sigmoid":
            return nn.Sigmoid()
        elif a == "relu":
            return nn.ReLU(inplace=True)
        elif a == "softsign":
            return nn.Softsign()
        elif a == "sin":
            return torch.sin
        elif a == "cos":
            return torch.cos
        else:
            print(f"Inserted activation function {a} not compatible. Using tanh.")
            return nn.Tanh()

    def forward(self, x):
        out = x
        for i, m in enumerate(self.layers):
            out = m(out)
            if i < len(self.layers)-1:
                out = self.activation(out)
            else:
                out = F.sigmoid(out)
        return out

class NeuralNet:
    def __init__(self, layers_dims, activations_fnc,  type_="classif"):
        self.type_ = type_
        self.W = [None] * (len(layers_dims)-1)
        self.b = [None] * (len(layers_dims)-1)
        self.out = [None] * (len(layers_dims)-1)
        self.layers_dims = layers_dims
        self.activations_fnc = activations_fnc
        for i in range(len(layers_dims)-1):
            self.b[i] = np.random.randn(layers_dims[i+1]).reshape(1, layers_dims[i+1])
            self.W[i] = np.random.randn(layers_dims[i], layers_dims[i+1])
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
        
    def tanh(self, x):
        return np.tanh(x)
        
    def relu(self, x):
        return np.maximum(x, 0)
    
    def identity(self, x):
        return x
    
    def softsign(self,x):
        return x / (1 + np.abs(x))
        
    def sin(self, x):
        return np.sin(x)
    
    def cos(self, x):
        return np.cos(x)
        
    def softmax(self, x):
        exp_scores = np.exp(x)
        out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return out
        
    def activation(self, x, t):
        t = t.lower()
        if t == "sigmoid":
            return self.sigmoid(x)
        elif t == "tanh":
            return self.tanh(x)
        elif t == "relu":
            return self.relu(x)
        elif t == "identity":
            return self.identity(x)
        elif t == "softsign":
            return self.softsign(x)
        elif t == "sin":
            return self.sin(x)
        elif t == "cos":
            return self.cos(x)
        elif t == "softmax":
            return self.softmax(x)
        else:
            raise Exception("Non-supported activation function {}".format(t))
            
    def multiply(self, x, W):
        m = np.dot(x, W)
        return m
    
    def add(self, m, b):
        return m + b
            
    def forward(self, x):
        input_ = x
        for i, activation in enumerate(self.activations_fnc) :
            weighted_sum = self.multiply(x=input_, W=self.W[i])
            weighted_sum = self.add(m=weighted_sum, b=self.b[i])
            input_ = self.out[i] = self.activation(x=weighted_sum, t=activation)
            

def hsv_to_rgb(h, s, v):
    ## hsw are between 0 and 1
    ## returns rgb between 0 and 1
    ## from: https://bgrins.github.io/TinyColor/docs/tinycolor.html
    h *= 6
    i = np.floor(h)
    f = h-i 
    p = v*(1-s)
    q = v*(1-f*s)
    t = v*(1-(1-f)*s)
    mod = int(i % 6)
    r = [v, q, p, p, t, v][mod]
    g = [t, v, v, q, p, p][mod]
    b = [p, p, t, v, v, q][mod]
    
    return r,g,b


def hsv_to_rgb_torch(img: torch.Tensor) -> torch.Tensor:
    ## hsw are between 0 and 1
    ## returns rgb between 0 and 1
    ## from: https://bgrins.github.io/TinyColor/docs/tinycolor.html
    h_ = img[:, :, 0].view(img.size(0)*img.size(1)).data.numpy()
    s_ = img[:, :, 1].view(img.size(0)*img.size(1)).data.numpy()
    v_ = img[:, :, 2].view(img.size(0)*img.size(1)).data.numpy()
    h = 6 * h_
    i = np.floor(h)
    f = h - i
    p = (v_ * (1 - s_))
    q = (v_ * (1 - f * s_))
    t = (v_ * (1 - (1 - f) * s_))
    mod = [int(a % 6) for a in i]
    r_select = torch.Tensor([[v_[i], q[i], p[i], p[i], t[i], v_[i]][m] for i,m in enumerate(mod)]).view(img.size(0),
                                                                                                        img.size(1)).unsqueeze(-1)
    g_select = torch.Tensor([[t[i], v_[i], v_[i], q[i], p[i], p[i]][m] for i, m in enumerate(mod)]).view(img.size(0),
                                                                                                        img.size(1)).unsqueeze(-1)
    b_select = torch.Tensor([[p[i], p[i], t[i], v_[i], v_[i], q[i]][m] for i, m in enumerate(mod)]).view(img.size(0),
                                                                                                        img.size(1)).unsqueeze(-1)
    img = torch.cat([r_select, g_select, b_select], dim=-1)
    return img


def hue_to_rgb(p, q, t):
    if t < 0 or t > 1:
        return p
    if t < 1/6:
        return p+(q-p)*6*t
    if t < 1/2:
        return q
    if t < 2/3:
        return p+(q-p)*(2/3-t)*6
    else:
        return p

def hsl_to_rgb(h, s, l):
    ## hsl are between 0 and 1
    ## returns rgb between 0 and 1
    if s==0:
        r = g = b = l #achromatic
    else:
        if l < 0.5:
            q = l*(1+s)
        else:
            q = l+s-l*s

        p = 2*l-q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

    return r, g, b

def hsl_to_rgb_torch(h,s,l):
    h = h.data.numpy()
    s = s.data.numpy()
    l = l.data.numpy()
    with Pool(processes=mp.cpu_count()-2) as pool:
        proc_img = pool.starmap(hsl_to_rgb, zip(h, s, l))
    proc_img = torch.Tensor(proc_img)
    return proc_img

def get_color_at(nnet, x, y, r, z1, z2, colormode, alpha):
    input_ = np.array([x,y, r, z1, z2], dtype=np.float32).reshape(1, 5)
    nnet.forward(input_)
    
    colormode = colormode.lower()
    
    if colormode == "rgb": ## Output via sigmoid activation mapped into range [0,1]
        r = nnet.out[len(nnet.out)-1][0][0]
        g = nnet.out[len(nnet.out)-1][0][1]
        b = nnet.out[len(nnet.out)-1][0][2]
        a_index = 3
    elif colormode == "bw":
        r=g=b = nnet.out[len(nnet.out)-1][0][0]
        a_index = 1
    elif colormode == "cmyk":
        c = nnet.out[len(nnet.out)-1][0][0]
        m = nnet.out[len(nnet.out)-1][0][1]
        y = nnet.out[len(nnet.out)-1][0][2]
        k = nnet.out[len(nnet.out)-1][0][3]
        r = (1-c)*k
        g = (1-m)*k
        b = (1-y)*k
        a_index = 4
    elif colormode == "hsv":
        h = nnet.out[len(nnet.out)-1][0][0]
        s = nnet.out[len(nnet.out)-1][0][1]
        v = nnet.out[len(nnet.out)-1][0][2]        
        r, g, b = hsv_to_rgb(h, s, v)
        a_index = 3
    elif colormode == "hsl":
        h = nnet.out[len(nnet.out)-1][0][0]
        s = nnet.out[len(nnet.out)-1][0][1]
        l = nnet.out[len(nnet.out)-1][0][2]  
        r, g, b = hsl_to_rgb(h, s, l)
        a_index = 3
    else:
        print("Inserted colormode '{}' is not part ob supported ones: [rgb, bw, cmyk, hsv, hsl]".format(colormode))
        raise Exception("Non-supported colormode {}".format(colormode))
    if alpha: 
        # Since non blackmode [rgb, cmyk, hsv, hsl] values are mapped onto [0,1] the alpha channel is also between [0,1].
        #0=transparency, 1=opaque wrt. to overlaying
        a = 1-abs(2*nnet.out[len(nnet.out)-1][0][a_index]-1)
        a = 0.25 + 0.75*a
    else:
        a = 1.0
    
    return r, g, b, a


def transform_colors(img, colormode, alpha):

    if alpha:
        alpha_tensor = img[:, :, -1]
        # Since non blackmode [rgb, cmyk, hsv, hsl] values are mapped onto [0,1] the alpha channel is also between [0,1].
        # 0=transparency, 1=opaque wrt. to overlaying
        a = 1 - torch.abs(2 * alpha_tensor - 1)
        a = (0.25 + 0.75 * a).unsqueeze(-1)
    else:
        a = torch.ones(size=(img.size(0), img.size(1))).unsqueeze(-1)

    if colormode == "rgb":  ## Output via sigmoid activation mapped into range [0,1]
        proc_img = img[:, :, 0:2]
    elif colormode == "bw":
        proc_img = torch.cat([img[:,:,0].unsqueeze(-1)]*3, dim=-1)
    elif colormode == "cmyk":
        r = ((1 - img[:, :, 0]) * img[:, :, 3]).unsqueeze(-1)
        g = ((1 - img[:, :, 1]) * img[:, :, 3]).unsqueeze(-1)
        b = ((1 - img[:, :, 2]) * img[:, :, 3]).unsqueeze(-1)
        proc_img = torch.cat([r, g, b], dim = -1)
    elif colormode == "hsv":
        proc_img = hsv_to_rgb_torch(img)
    elif colormode == "hsl":
        h = (img[:, :, 0]).view(img.size(0)*img.size(1))
        s = (img[:, :, 1]).view(img.size(0)*img.size(1))
        l = (img[:, :, 2]).view(img.size(0)*img.size(1))
        proc_img = hsl_to_rgb_torch(h, s, l)
        proc_img = proc_img.view(img.size(0), img.size(1), 3)
    else:
        print("Inserted colormode '{}' is not part ob supported ones: [rgb, bw, cmyk, hsv, hsl]".format(colormode))
        raise Exception("Non-supported colormode {}".format(colormode))

    res = torch.cat([proc_img, a], dim=-1)
    return res

def init_image(rows, cols):
    img = np.zeros(shape=(rows, cols, 4))
    return img

def prep_nnet_arch(n_depth, n_size, activation, colormode, alpha):
    layers = [5] #x, y, r, z1, z2
    for i in range(n_depth):
        layers.append(n_size)
    
    colormode = colormode.lower()
    
    ### Output layer. Append number of output neurons depending on which colormode is selected
    if colormode in ["rgb", "hsv", "hsl"] : #RGB
        if not alpha:
            layers.append(3)
        else:
            layers.append(4)
    elif colormode == "cmyk":
        if not alpha:
            layers.append(4)
        else:
            layers.append(5) 
    elif colormode == "bw":
        if not alpha: 
            layers.append(1)
        else:
            layers.append(2)
    else:
        print("Inserted colormode '{}' is not part ob supported ones: [rgb, bw, cmyk, hsv, hsl]".format(colormode))
        raise Exception("Non-supported colormode {}".format(colormode))
    
    possibles = ["sigmoid", "tanh", "relu", "identity", "softsign", "sin", "cos", "softmax"]
    if not activation.lower() in possibles:
        print('defined activation {} not supported in {}'.format(activation, str(possibles)))
        return None
        
    activations_fnc = [activation] * (len(layers)-2)
    activations_fnc.append("sigmoid")
    
    return layers, activations_fnc   


def process_xy_el(xy_el, symmetry, trig, z1, z2):
    if symmetry:
        xy_el = [a**2 for a in xy_el]
    xy_el.append(np.sqrt(xy_el[0]**2 + xy_el[1]**2))
    if trig:
        xy_el.append(np.cos(z1*xy_el[0]))
        xy_el.append(np.sin(z2*xy_el[1]))
    else:
        xy_el.append(z1)
        xy_el.append(z2)
    return xy_el


def init_data(img_height=500, img_width=700, symmetry=True, trig=True, z1=-0.618, z2=0.618):
    factor = min(img_height, img_width)
    # get input: x,y,r,z_1,z_2
    x = [(i/factor-0.5)*2 for i in range(img_height)]
    y = [(j/factor-0.5)*2 for j in range(img_width)]
    #xy = [[x0, y0] for x0 in x for y0 in y]
    in_data = [process_xy_el([x0, y0], symmetry, trig, z1, z2) for x0 in x for y0 in y]
    in_data = torch.Tensor(in_data)
    return in_data


def generate_image_torch(my_net, img_height=500, img_width=700, symmetry=True, trig=True, colormode="rgb", alpha=False,
                         z1=-0.618, z2=0.618, show=True, fname="netart.png", save=True):

    input_data = init_data(img_height, img_width, symmetry, trig, z1, z2)
    img = my_net(input_data)
    img = img.view(img_height, img_width, img.size(-1))
    img = transform_colors(img, colormode, alpha)
    #noise = nn.functional.sigmoid(torch.empty(size=img.size()).normal_(mean=0, std=0.1))
    #img = img*noise
    img = img.data.numpy() ## fix
    if not show:
        matplotlib.use("Agg")

    plt.figure()
    fig = plt.imshow(img, interpolation="bilinear", aspect="auto")
    plt.axis("off")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    if show:
        plt.show()

    if save:
        plt.imsave("{}".format(fname), img, format="png")

    return img


def generate_image(img_height, img_width, n_depth, n_size, activation, colormode, alpha, z1, z2,
                   fname="netart.png", nnet_dict=None, save=False, show=False, symmetry=False):
    factor = min(img_height, img_width)
    if nnet_dict is None:
        layers, activations_fnc = prep_nnet_arch(n_depth, n_size, activation, colormode, alpha)
    else:
        try:
            layers = nnet_dict["layers"]
            activations_fnc = nnet_dict["activations_fnc"]
            assert len(activations_fnc) == len(layers)-1
            assert layers[0] == 5
            assert activations_fnc[-1].lower() in ["sigmoid", "softmax"] 
        except Exception as e:
            print(e)
        
    nnet = NeuralNet(layers, activations_fnc) 
    img = init_image(img_height, img_width)
    for i in range(img_height):
        for j in range(img_width):
            x = (i/factor - 0.5)*2
            y = (j/factor - 0.5)*2
            if symmetry:
                x = x**2
                y = y**2
                #r_ = 0
                ## Uncomment z1 and/or z2 to add trigonometric functions
                #z1 = np.cos(z1*x)
                #z2 = np.sin(z2*y)
            r_ = np.sqrt(x**2 + y**2)
            #Get RGBA values
            r, g, b, a = get_color_at(nnet, x=x, y=y, r=r_,
                                   z1=z1, z2=z2, colormode=colormode, alpha=alpha)
            #Populate the image
            img[i, j, 0] = r
            img[i, j, 1] = g
            img[i, j, 2] = b
            img[i, j, 3] = a
    
    if not show:
        matplotlib.use("Agg")
    
    plt.figure()    
    fig = plt.imshow(img, interpolation="bilinear", aspect="auto")
    plt.axis("off")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    
    if show:
        plt.show()
    
    if save:
        plt.imsave("{}".format(fname), img, format="png")
        
    return img

def args_parser():
    parser = argparse.ArgumentParser(description="Generate random art with a deep neural network")
    
    parser.add_argument("-img_height", metavar="", type=int, default=512,
                        help="Image height of created random art. Default is 512")
    
    parser.add_argument("-img_width", metavar="", type=int, default=512,
                        help="Image width of created random art. Default is 512")
    
    parser.add_argument("-colormode", metavar="", type=str, default="RGB",
                        help="How image color should be generated. Options are ['BW', 'RGB', 'CMYK', 'HSV', 'HSL']. By default this value is 'RGB'")
        
    parser.add_argument("-alpha", metavar="", type=str, default="False",
                        help="Whether or not to add a alpha channel for the image. Default is False")
    
    parser.add_argument("-n_images", metavar="", type=int, default=1,
                        help="Number of images to generate. Default is 1")
        
    parser.add_argument("-n_depth", metavar="", type=int, default=10,
                        help="Number of layers for the neural network. Default is 10")
    
    parser.add_argument("-n_size", metavar="", type=int, default=15,
                        help="Number of neurons in each hidden layer. Default is 15")
    
    parser.add_argument("-activation", metavar="", type=str, default="tanh",
                        help="Activation function to apply on the hidden layers. Default is 'tanh'")
        
    parser.add_argument("-z1", metavar="", type=float, default=-0.618,
                        help="Input variable 1 to insert determinism into the random art. The value should be between -1 and 1. Default is -0.618")
        
    parser.add_argument("-z2", metavar="", type=float, default=+0.618,
                        help="Input variable 2 to insert determinism into the random art. The value should be between -1 and 1. Default is +0.618")
    parser.add_argument("-sym", metavar="", type=str, default="False",
                        help="Use symmetry network. Default is False")

    parser.add_argument("-vector", metavar="", type=str, default="True",
                        help="Use vectorized calculation. Defaults to True")


    args = parser.parse_args()
    
    return args

def info_print(args):
    """
    This function prints the input arguments from argparse when calling this script via python shell.
    Args:
        args [argparse.Namespace]: argument namespace from main.py
    Returns:
        None
    """
    print(37*"-")
    print("Random Art with Deep Neural Networks:")
    print(37*"-")
    print("Script Arguments:")
    print(17*"-")
    for arg in vars(args):
        print (arg, ":", getattr(args, arg))
    print(17*"-")
    return None

def str_to_bool(s):
    """
    This function converts a string into a boolean value
    Args:
        s [str]: string representing tre value
    Returns:
        b [bool]: boolean value of string representation
        
    """
    if s.lower() in ["true", "yes", "y", "t", "1"]:
        b = True
    elif s.lower() in ["false", "no", "f", "n", "0"]:
        b = False
    else:
        print("boolean string not correctly specified")
        sys.exit(1)
    return b


def main():
    ## retrieve arguments and print out in shell
    args = args_parser()
    ## print out information on shell
    info_print(args)
    
    ### Params ###
    img_height = args.img_height
    img_width = args.img_width
    colormode= args.colormode
    alpha = str_to_bool(args.alpha)
    n_images = args.n_images
    n_depth = args.n_depth
    n_size = args.n_size
    activation = args.activation
    z1 = args.z1
    z2 = args.z2
    symmetry = str_to_bool(args.sym)
    vectorize = str_to_bool(args.vector)
    
    if not os.path.exists("results"):
        print("Creating subdirectory 'results'.")
        os.makedirs("results")

    if vectorize:
        print("Using vectorized computation.")
    else:
        print("Using nested for loop.")

    for i in range(n_images):
        start_time = time.time()
        print("Generating image number {}...".format(i+1))
        save_path = "results/{}_generated{}.png".format(colormode, i+1)
        if vectorize:
            my_net = TorchNeuralNet(layers_dims=[n_size]*n_depth, activation_fnc=activation,
                                    colormode=colormode, alpha=alpha)
            my_net = my_net.apply(weight_init)
            g_img = generate_image_torch(my_net, img_height, img_width, symmetry=False, trig=False, colormode=colormode,
                                 alpha=alpha, z1=z1, z2=z2, show=False, fname=save_path, save=True)
        else:
            g_img = generate_image(img_height, img_width, n_depth, n_size, activation, colormode, alpha, z1, z2,
                           fname=save_path, save=True, show=False, symmetry=symmetry)
        delta = time.time()-start_time
        print("Generating image took {} seconds".format(delta))
        print("Image number {} saved at {}".format(i+1, save_path))
        
if __name__ == "__main__":
    main()