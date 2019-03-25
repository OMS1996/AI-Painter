"""
Last edited on Tue Mar 19 20:32:04 2019
@author: Omar M.Hussein
"""
import os #The OS module in Python provides a way of using operating system dependent functionality. The functions that the OS module provides allows you to interface with the underlying operating system that Python
import sys # System-specific parameters and functions. This module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
import scipy.io #Input and output ( scipy.io ) SciPy has many modules, classes, and functions available to read data from and write data to a variety of file formats. See also. numpy-reference.routines.io (in Numpy)
import scipy.misc # to read images
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image # Image handling
from nst_utils import *
import numpy as np #numpy
import tensorflow as tf #To create tensors and better use deeplearning
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

def compute_content_cost(activation_Content, activation_Generated):
    """
    Function  Number : 1
    Purpose :
    Computes the content cost takes the hidden layer activation function
    of both content image as well as the generated image and measures the difference between them
    
    Arguments / Function Input
    n refers to number
    activation_Content : tensor of dimension (1, n_Height, n_Width, n_channels), hidden layer activations representing content of the content image 
    activation_Generated : tensor of dimension (1, n_Height, n_Width, n_channels), hidden layer activations representing content of the generated image
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    # Getting the dimensions of that specific layer which as the channels,the area = width and height
    m, n_Height, n_Width, n_channels = activation_Generated.get_shape().as_list()
    
    # Reshape activation_Content, activation_Generated activation layer of content and generated image
    # Unrolling convention for better calculations
    activation_Content_unrolled = tf.transpose(tf.reshape(activation_Content, [n_Height * n_Width, n_channels]))
    activation_Generated_unrolled = tf.transpose(tf.reshape(activation_Generated, [n_Height * n_Width, n_channels]))
    
    # Compute the content_cost with tensor-flow mathematical functions 
    J_content = J_content = tf.reduce_sum(tf.divide(tf.square(tf.subtract(activation_Generated_unrolled, activation_Content_unrolled)),(4 * n_Height * n_Width * n_channels)))
    # Calculating the Cost function for the each indivisual value then getting the sum gets the sum so we can have the cost represente in One number
    #Personal note : Omar run it in a session later dont forget that :D
    
    #return only the cost of the image content
    return J_content

def gram_matrix(unrolled_mat):
    """
    Function Number : 2
    
    Purpose : 
    In order to create a Style cost function you must first create a Style Matrix
    so what this does is it takes in the "unrolled" filter matrix and multiplies[Matrix Multiplication] them by their transpose resulting in the gram matrix
    and with that it measures how similair the activation filters are
    
    Argument:
    unrolled_mat : matrix of shape (n_channels, n_Height * n_Width)
    
    Returns:
    Grammed : Gram matrix of A, of shape (n_channels, n_channels)
    """
    Grammed = tf.matmul(unrolled_mat,tf.transpose(unrolled_mat))
    return Grammed

def compute_layer_style_cost(activation_Style, activation_Generated):
    """
    Function Number : 3
    
    Purpose :: obtain the activation matrices when for Style Image and the other the generated image
              then unroll them so they can be turned into gram matrix to obtain their styles the both of them 
              and then find then finding the cost function using the style cost function introduced in the papers
    
    
    
    Arguments:
    activation_Style :: tensor of dimension (1, n_Height, n_Width, n_channels), hidden layer activations representing style of the image S 
    activation_Generated :: tensor of dimension (1, n_Height, n_Width, n_channels), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer :: tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    
    # Retrieve dimensions from a_G (≈1 line)
    m, n_Height, n_Width, n_channels = activation_Generated.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_channels, n_Height * n_Width)) 
    # When reshaping you first choose which arguments to reshape then you reshape it however youlike
    # un rolling you first get the area then you, put the number of channels
    activation_Style = tf.transpose(tf.reshape(activation_Style,[n_Height*n_Width,n_channels]))
    activation_Generated = tf.transpose(tf.reshape(activation_Generated,[n_Height*n_Width,n_channels]))

    # Computing gram_matrices for both images
    GrammedStyle = gram_matrix(activation_Style)
    GrammedGenerated = gram_matrix(activation_Generated)

    # Computing the loss
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GrammedStyle,GrammedGenerated))) / (4 * n_channels **2 * (n_Height * n_Width) **2)

    
    return J_style_layer

"""
Initalization for the convulotional layers value
Purpose : After getting the style from layer the next logical step is to concatenate all the layers together to get more precise results
so they will be initiailized
i choose to initalize them all equally so 10/5 = 2
that mean each will be 0.2 they can however change depending on which features you want to test for lower or higher level features

The way it is calculated is by obtaining the sum of each layer*by weight in this case 0.2

"""
#Convolutional layers initialized
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    """
    Function number 4
    
    Purpose : is to obtain the overall style cost from several chosen layers [summation[lambda * stylecost]i....l]
    
    Arguments:
    model : VGG 19
    STYLE_LAYERS: A list containing:
    - the names of the layers we would like to use their style.
    - Value for each of them.
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation in Gatys et al paper
    """
    
    # Accumulation of style cost
    J_style = 0
    # Getting the name and the coeff
    # LOOPING THROUGH STYLE LAYERS EXTRACTING NAMES AND VALUES Coeff is lambda
    
    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set act_Style to be the hidden layer activation from the layer we have selected, by running the session on out
        act_Style = sess.run(out)

        # Set act_Gen to be the hidden layer activation from same layer. Here, act_Gen references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image Generated as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with Generated as input.
        act_Gen = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(act_Style, act_Gen)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Function Number 5
    
    Purpose is to Compute the total cost function of both styles Content and style
    
    Arguments:
    J_content : content cost.
    J_style : style cost.
    alpha : hyper-parameter weighting the contribution of the content cost
    beta : hyper-parameter weighting the contribution of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    
    return J

"""
last edited on Tue Mar 12 2019
@author: Omar M.Hussein
"""
# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

# Content image loaded and reshaped
content_image = scipy.misc.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)

# Content image reloaded and reshaped
style_image = scipy.misc.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)

# Initializing the generated image with Noisy Content Image
generated_image = generate_noise_image(content_image)

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

# Content Image Vs Generated Image Cost
# Feeding the content image into the VGG-19 model
sess.run(model['input'].assign(content_image))
# Choosing the layer conv4_2 (Tensor)
out = model['conv4_2']
# Set act_content
act_content = sess.run(out)
# This is kept unevaluated on purpose here rather only refences model conv4_2
act_gen = out
# Compute the content cost
J_content = compute_content_cost(act_content, act_gen)

# Style Image Vs Generated Image Cost
# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

#getting the total cost with a weighting betweem style and content described in alpha and beta
J = total_cost(J_content, J_style,alpha = 10,beta = 40)

#optimisation
op = tf.train.AdamOptimizer(1.9)
train_step = op.minimize(J)

def model_nn(sess, input_image, num_iterations = 200):
    
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model.
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        _ =sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])
        
        # Print every 50 iteration.
        if i%50 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Summary of all the Values and costs")
            print("Iteration(S) " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            #SAVE the generated image after every 50 iterations 
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image after the final iteration
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image

model_nn(sess, generated_image)