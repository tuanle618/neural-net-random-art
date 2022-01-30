# -*- coding: utf-8 -*-
"""
@title: random_art.py
@author: Tuan Le
@email: tuanle@hotmail.de

Main python programme to generate random images using feedforward nets
"""

import os
import sys
import time
import json
import argparse
import matplotlib
import matplotlib.pyplot as plt

import torch
from nnetart.network import FeedForwardNetwork
from nnetart.artgen import init_data, transform_colors


def generate_image_torch(my_net: FeedForwardNetwork = FeedForwardNetwork(),
                         img_height=512,
                         img_width=512,
                         symmetry=False,
                         trig=True,
                         colormode="rgb",
                         alpha=False,
                         z1=-0.618, z2=0.618,
                         show=True,
                         fname="netart",
                         format="png",
                         save=True,
                         gpu=False,
                         with_noise=False,
                         noise_std=0.01):
    input_data = init_data(img_height, img_width, symmetry, trig, z1, z2, noise=with_noise, noise_std=noise_std)
    if gpu:
        input_data = input_data.cuda()
        my_net = my_net.cuda()

    with torch.no_grad():
        img = my_net(input_data)

    if gpu:
        img = img.cpu()

    img = img.view(img_height, img_width, img.size(-1))
    img = transform_colors(img, colormode, alpha)
    img = img.numpy()
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
        plt.imsave(f"{fname}.{format}", img, format=format)
    return img


def args_parser():
    parser = argparse.ArgumentParser(description="Generate random art with a deep neural network")
    
    parser.add_argument("-img_height", metavar="", type=int, default=512,
                        help="Image height of created random art. Default is 512")
    
    parser.add_argument("-img_width", metavar="", type=int, default=512,
                        help="Image width of created random art. Default is 512")
    
    parser.add_argument("-colormode", metavar="", type=str, default="rgb",
                        help="How image color should be generated. Options are ['bw', 'rgb', 'cmyk', 'hsv', 'hsl']."
                             " By default this value is 'rgb'")
        
    parser.add_argument("-alpha", metavar="", type=str, default="False",
                        help="Whether or not to add a alpha channel for the image. Default is False")
    
    parser.add_argument("-n_images", metavar="", type=int, default=1,
                        help="Number of images to generate. Default is 1")
        
    parser.add_argument("-n_depth", metavar="", type=int, default=5,
                        help="Number of layers for the neural network. Default is 5")
    
    parser.add_argument("-n_size", metavar="", type=int, default=10,
                        help="Number of neurons in each hidden layer. Default is 10")
    
    parser.add_argument("-activation", metavar="", type=str, default="tanh",
                        help="Activation function to apply on the hidden layers. Default is 'tanh'")
        
    parser.add_argument("-z1", metavar="", type=float, default=-0.618,
                        help="Input variable 1 to insert determinism into the random art."
                             " The value should be between -1 and 1. Default is -0.618")
        
    parser.add_argument("-z2", metavar="", type=float, default=+0.618,
                        help="Input variable 2 to insert determinism into the random art."
                             " The value should be between -1 and 1. Default is +0.618")

    parser.add_argument("-trig", metavar="", type=str, default="True",
                        help="If the z1 and z2 values should be transformed with cosine and sine respectively. "
                             "Defaults to True.")

    parser.add_argument("-noise", metavar="", type=str, default="False",
                        help="If gaussian noise should be added for the generated image. Defaults to False")

    parser.add_argument("-noise_std", metavar="", type=float, default=0.01,
                        help="Gaussian noise standard deviation if it should be added to the generated image. "
                             " Defaults to 0.01.")

    parser.add_argument("-sym", metavar="", type=str, default="False",
                        help="Use symmetry network. Default is False")

    parser.add_argument("-gpu", metavar="", type=str, default="False",
                        help="Use GPU to generate (vectorized) image. Defaults to False")

    parser.add_argument("-format", metavar="", type=str, default="png",
                        help="File format to save the images. Defaults to 'png'."
                             "Choices are 'pnd', 'jpg', 'pdf' and 'svg'.",
                        choices=['png', 'jpg', 'svg', 'pdf']
                        )

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
    print(50*"-")
    print("Random Art with Deep Neural Networks:")
    print(50*"-")
    print("Script Arguments:")
    print(25*"-")
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print(25*"-")
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
    # retrieve arguments and print out in shell
    args = args_parser()
    # print out information on shell
    info_print(args)
    
    # Params #
    img_height = args.img_height
    img_width = args.img_width
    colormode = args.colormode.lower()
    alpha = str_to_bool(args.alpha)
    n_images = args.n_images
    n_depth = args.n_depth
    n_size = args.n_size
    activation = args.activation
    z1 = args.z1
    z2 = args.z2
    symmetry = str_to_bool(args.sym)
    trig = str_to_bool(args.trig)
    gpu = str_to_bool(args.gpu)
    with_noise = str_to_bool(args.noise)
    noise_std = args.noise_std
    
    if not os.path.exists("results"):
        print("Creating subdirectory 'results'.")
        os.makedirs("results")

    save_dir = "results/" + time.strftime('%Y%m%d%H%M%S')

    os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    for i in range(n_images):
        start_time = time.time()
        print("Generating image number {}...".format(i+1))
        save_path = "{}/{}_generated{}.{}".format(save_dir, colormode, i+1, args.format)
        my_net = FeedForwardNetwork(layers_dims=[n_size]*n_depth, activation_fnc=activation,
                                    colormode=colormode.lower(), alpha=alpha)

        _ = generate_image_torch(my_net, img_height, img_width, symmetry=symmetry,
                                 trig=trig, colormode=colormode,
                                 alpha=alpha, z1=z1, z2=z2, with_noise=with_noise, noise_std=noise_std,
                                 show=False, fname=save_path, save=True, gpu=gpu)

        delta = time.time()-start_time
        print("Generating image took {} seconds".format(delta))
        print("Image number {} saved at {}".format(i+1, save_path))


if __name__ == "__main__":
    main()