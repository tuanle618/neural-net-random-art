# neural-net-random-art
Create a grayscale or colour image with predefined size `img_height` and `img_width` using fully connected neural networks.  
The generation of images only requires python `numpy`, `torch` and `matplotlib`.  
Medium article can be found [here](https://medium.com/@tuanle618/generate-abstract-random-art-with-a-neural-network-ecef26f3dd5f).
# Usage
You can either have a look at the jupyter notebook [nb_random_art.ipynb](https://github.com/tuanle618/neural-net-random-art/blob/master/nb_random_art.ipynb) if you want to understand the algorithm and check out several settings of the method.  
For fast image generation is is recommended to use the python main programm file [random_art.py](https://github.com/tuanle618/neural-net-random-art/blob/master/random_art.py)

# Using Google Colab
I've created a minimal working example on Google Colab here: https://colab.research.google.com/drive/1TFmQQOUHOPjSrB0dVeoYiD7d7FPidilW?usp=sharing in case you want to experiment with this library a little bit, prior to installing it on your local machine.

# Dependencies: Python 3
```
numpy
matplotlib
seaborn
torch
```

# Installation
```
git clone https://github.com/tuanle618/neural-net-random-art.git
cd neural-net-random-art
pip install -e .
```

# Execution

For the `random_art.py` program `argparse` is used to define several input parameters:
```python 3
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
```

So in order to create 1 RGB image of size 400x500, no alpha channel, a dense net with 15 layers, each laying having 15 neurons, type in following command in the shell:  

`
python random_art.py -img_height 400 -img_width 500 -colormode rgb -alpha False -n_images 1 -n_depth 15 -n_size 15
`

# Examples
Following commands were used [default params were used] to get the images stored in the [result subdirectory](https://github.com/tuanle618/neural-net-random-art/tree/master/results):  
```
python random_art.py -img_height 512 -image_width 512 -colormode bw -alpha False -n_images 5
```
E.g, leading to following 2 random images (resized in order to have next to each other):  
<p float="left">
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/BW_generated3.png" width="420" />
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/BW_generated5.png" width="420" /> 
</p>  
  
```
python random_art.py -img_height 512 -img_width 512 -colormode rgb -alpha False -n_images 10
```

E.g, leading to following 2 random images (resized in order to have next to each other):  
<p float="left">
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/RGB_generated4.png" width="420" />
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/RGB_generated5.png" width="420" /> 
</p>
  
```
python random_art.py -img_height 512 -img_width 512 -colormode cmyk -alpha False -n_images 5
```

E.g, leading to following 2 random images (resized in order to have next to each other):  
<p float="left">
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/CMYK_generated2.png" width="420" />
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/CMYK_generated5.png" width="420" /> 
</p>

```
python random_art.py -img_height 512 -img_width 512 -colormode hsv -alpha False -n_images 5
```

E.g, leading to following 2 random images (resized in order to have next to each other):  
<p float="left">
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/HSV_generated4.png" width="420" />
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/HSV_generated5.png" width="420" /> 
</p>

```
python random_art.py -img_height 512 -img_width 512 -colormode hsl -alpha False -n_images 5
```

E.g, leading to following 2 random images (resized in order to have next to each other):  
<p float="left">
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/HSL_generated2.png" width="420" />
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/HSL_generated3.png" width="420" /> 
</p>
  
You can try out different input arguments (larger networks and neurons, different actiation functions etc..) as suggested in the Jupyter Notebook, to see what images will be created.  
For example the following images are created by deeper neural nets.  
Image 1: `n_depth=15` and `n_size=25`, Image 2: `n_depth=25` and `n_size=45`:
<p float="left">
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/deepRGB.png" width="420" />
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/verydeepRGB.png" width="420" />
</p>

------

### If you like this repo, feel free to ⭐ and share it!


# License
Code under MIT License.
