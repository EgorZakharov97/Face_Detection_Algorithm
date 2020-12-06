"""
SCI 2000 Fall 2020
Assignment 4 Solution
Jennifer Vaughan
"""

import imagefuncs_A4 as imf
import numpy as np
import sys

if len(sys.argv) > 1:

    filename = sys.argv[1]
   
    image = imf.read_image(filename)
    
    basename = filename[:filename.rfind('.')]

    # choice of Gaussian smoothing kernel:  standard deviation of 2,
    # and 4 nearest neighbors in each direction
    sigma = 2
    r = 4
    gauss2 = imf.gaussian1D(sigma, r)
    
    # Step 0:  smooth the image
    smooth_image = imf.convolve2D_separable(image, gauss2)
    smooth_name = basename + "-0-smooth.pgm"
    imf.write_image(smooth_name, smooth_image)
        
    # Step 1:  calculate lengths of gradient vectors
    gradientjk = imf.gradient_vectors(smooth_image.data)
    grad_lengths = imf.gradient_lengths(gradientjk)
    grad_max = np.amax(grad_lengths)
    grad_image = imf.PGMFile(grad_max, grad_lengths)
    grad_name = basename + "-1-grad.pgm"
    imf.write_image(grad_name, grad_image)
    
    # Step 2:  thin the edges
    thin_data = imf.thin_edges(gradientjk, grad_lengths)
    thin_image = imf.PGMFile(grad_max, thin_data)
    thin_name = basename + "-2-thin.pgm"
    imf.write_image(thin_name, thin_image)

    # choices for low and high thresholds for noise suppression:
    # the low threshold is 10% of the maximum pixel value, and 
    # the high threshold is 18%
    low_threshold = 0.1*thin_image.max_shade
    high_threshold = 0.18*thin_image.max_shade

    # Step 3:  suppress noise
    suppr_data = imf.suppress_noise(thin_image.data, low_threshold, high_threshold)
    suppr_image = imf.PGMFile(thin_image.max_shade, suppr_data)
    suppr_name = basename + "-3-suppr.pgm"
    imf.write_image(suppr_name, suppr_image)
   
else:
    print("Please enter a file name.")
