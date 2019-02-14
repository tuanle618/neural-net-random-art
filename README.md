# neural-net-random-art
Create a grayscale or colour image with predefined size `image_height` and `image_width` using fully connected neural networks.  
The generation of images only requires python `numpy` and `matplotlib`.  
Medium article can be found [here](https://medium.com/@tuanle618/generate-abstract-random-art-with-a-neural-network-ecef26f3dd5f).
# Usage
You can either have a look at the jupyter notebook [nb_random_art.ipynb](https://github.com/tuanle618/neural-net-random-art/blob/master/nb_random_art.ipynb) if you want to understand the algorithm and check out several settings of the method.  
For fast image generation is is recommended to use the python main programm file [random_art.py](https://github.com/tuanle618/neural-net-random-art/blob/master/random_art.py)

# Dependencies: Python 3
```
numpy==1.15.3
matplotlib==3.0.0
seaborn==0.9.0
```

# Execution

For the `random_art.py` programm `argparse` is used to define several input parameters:
```python 3
parser = argparse.ArgumentParser(description="Generate random art with a deep neural network")
parser.add_argument("-img_height", metavar="", type=int, default=512,
                   help="Image height of created random art. Default is 512") 
parser.add_argument("-img_width", metavar="", type=int, default=512,
                   help="Image width of created random art. Default is 512") 
parser.add_argument("-colormode", metavar="", type=str, default="RGB",
                   help="How image color should be generated. Options are ['BW', 'RGB', 'CMYK', 'HSV', 'HSL']. By default this
                    value is 'RGB'")    
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
                   help="Input variable 1 to insert determinism into the random art. The value should be between -1 and 1. Default 
                    is -0.618")    
parser.add_argument("-z2", metavar="", type=float, default=+0.618,
                   help="Input variable 2 to insert determinism into the random art. The value should be between -1 and 1. Default 
                   is +0.618")
args = parser.parse_args()
```

So in order to create 1 RGB image of size 400x500, no alpha channel, a dense net with 15 layers, each laying having 15 neurons, type in following command in the shell:  

`
python random_art.py -img_height 400 -image_width 500 -colormode RGB -alpha False -n_images 1 -n_depth 15 -n_size 15
`

# Examples
Following commands were used [default params were used] to get the images stored in the [result subdirectory](https://github.com/tuanle618/neural-net-random-art/tree/master/results):  
```
python random_art.py -img_height 512 -image_width 512 -colormode BW -alpha False -n_images 5
```
E.g, leading to following 2 random images (resized in order to have next to each other):  
<p float="left">
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/BW_generated3.png" width="420" />
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/BW_generated5.png" width="420" /> 
</p>  
  
```
python random_art.py -img_height 512 -image_width 512 -colormode RGB -alpha False -n_images 10
```

E.g, leading to following 2 random images (resized in order to have next to each other):  
<p float="left">
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/RGB_generated4.png" width="420" />
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/RGB_generated5.png" width="420" /> 
</p>
  
```
python random_art.py -img_height 512 -image_width 512 -colormode CYMYK -alpha False -n_images 5
```

E.g, leading to following 2 random images (resized in order to have next to each other):  
<p float="left">
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/CMYK_generated2.png" width="420" />
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/CMYK_generated5.png" width="420" /> 
</p>

```
python random_art.py -img_height 512 -image_width 512 -colormode HSV -alpha False -n_images 5
```

E.g, leading to following 2 random images (resized in order to have next to each other):  
<p float="left">
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/HSV_generated4.png" width="420" />
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/HSV_generated5.png" width="420" /> 
</p>

```
python random_art.py -img_height 512 -image_width 512 -colormode HSL -alpha False -n_images 5
```

E.g, leading to following 2 random images (resized in order to have next to each other):  
<p float="left">
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/HSL_generated2.png" width="420" />
  <img src="https://github.com/tuanle618/neural-net-random-art/blob/master/results/HSL_generated3.png" width="420" /> 
</p>
  
  
You can try out different input arguments (larger networks and neurons, different actiation functions etc..) as suggested in the Jupyter Notebook, to see what images will be created.

# License
Code under MIT License.
