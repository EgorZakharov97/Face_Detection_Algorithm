"""
Library of image manipulation functions - Assignment 4
"""

import numpy as np
import math
from collections import namedtuple

# the only relevant file type is P2
FILETYPE = 'P2'

# structure for storing the data of a PGM file
PGMFile = namedtuple('PGMFile', 'max_shade, data')

# kernel K2 from class
K2 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

# kernels for use in calculating central differences
CDIFF = np.array([-0.5,0,0.5])
CDIFF_TRUNC = np.array([-1,1])



"""
The header of a PGM file has the form

P2
20 30
255

where P2 indicates a PGM file with ASCII encoding,
and the next three numbers are the number of columns in the image,
the number of rows in the image, and the maximum pixel value.

Comments are indicated by # and could occur anywhere in the file.
For simplicity, we assume that a # is preceded by whitespace.

The remaining entries are pixel values.

This function receives the name of a PGM file and returns the data in
the form of a PGMFile.
"""
def read_image(filename):
    
    with open(filename) as imagefile:
        # list to hold integer entries in the file
        entries = []
    
        firstword = True
        for line in imagefile:
            words = line.split()
            worditer = iter(words)
            comment = False
            endline = False
            while not comment and not endline:
                try:
                    word = next(worditer)
                    if not word.startswith('#'):
                        if not firstword:
                            # an entry that is not part of a comment and is not 
                            # the first word is an integer
                            entries.append(int(word))
                        else:
                            # the first word that is not a comment is the file type
                            assert word == FILETYPE, f"The only supported file type is P2."
                            firstword = False
                    else:
                        # this is a comment; drop the rest of the line
                        comment = True
                except StopIteration:
                    endline = True

    num_cols = entries[0] # number of columns in the image
    num_rows = entries[1] # number of rows in the image
    max_shade = entries[2] # maximum pixel value
        
    # all remaining integers are pixel values
    # arrange them in a numpy array using dimensions read from the header
    data = np.reshape(np.array(entries[3:]), (num_rows, num_cols))

    return PGMFile(max_shade, data)


"""
This function receives a file name and a PGMFile, and writes
a PGM file with the given data.

The pixel data must be in a NumPy array whose dimensions are the 
number of rows and number of columns in the image.  

Entries in the array will be cast to integers before being written
to the file, e.g. an entry of 9.9 will be written as 9.
"""
def write_image(filename, image):

    # read the dimensions of the image from the shape of the pixel data array
    num_rows, num_cols = image.data.shape

    # create the file header
    header = f"{FILETYPE}\n{num_cols} {num_rows}\n{image.max_shade}"

    # entries in the pixel data array are written to the file as integers
    np.savetxt(filename, image.data, fmt="%d", comments='', header=header)

    return


"""
Given a standard deviation sigma, the formulas for the 1D and 2D Gaussian
distributions are

P(x) = [1/sqrt(2 pi sigma^2)] * exp[-x^2/(2 sigma^2)]

P(x,y) = [1/(2 pi sigma^2)] * exp[-(x^2+y^2)/(2 sigma^2)]
"""

# return the value of a 1D Gaussian at a point
def gaussianPt1D(x, sigma=1):
    A = 1/math.sqrt(2*math.pi*sigma*sigma)
    return math.exp(-(x*x)/(2*sigma*sigma))*A

# return the value of a 2D Gaussian at a point
def gaussianPt2D(x, y, sigma=1):
    A = 1/(2*math.pi*sigma*sigma)
    return math.exp(-(x*x+y*y)/(2*sigma*sigma))*A

# given the stddev sigma and the number of neighbors r, return a 1D Gaussian
# kernel of length 2r + 1, renormalized so the entries sum to 1
def gaussian1D(sigma=1, r=2):
    gaussArray = np.zeros(2*r+1)
    for i in range(r+1):
        # use symmetry of the Gaussian distribution under x -> -x
        gaussArray[r+i] = gaussArray[r-i] = gaussianPt1D(i,sigma)
    # rescale so the entries sum to 1
    return gaussArray/np.sum(gaussArray)

# given the stddev sigma and the number of neighbors r, return a 2D Gaussian
# kernel of size (2r + 1) x (2r + 1), renormalized so the entries sum to 1
def gaussian2D(sigma=1, r=2):
    gaussArray = np.zeros((2*r+1,2*r+1))
    for i in range(r+1):
        for j in range(r+1):
            # use symmetry of the 2D Gaussian distribution 
            # under x -> -x and y -> -y
            gaussArray[r+i,r+j] = gaussArray[r+i,r-j] = gaussArray[r-i,r+j] = gaussArray[r-i,r-j] = gaussianPt2D(i,j,sigma)
    # rescale so the entries sum to 1
    return gaussArray/np.sum(gaussArray)


"""
In the convolution operation, the current pixel is aligned with the center of
the kernel, which is assumed to have length 2r+1.  If the kernel extends past 
the boundary of the image, the kernel is truncated.  

current position
      |
   [p p p p] p p p ...
 k [k k k k]

The following functions return the left and right indices for the slice of 
the kernel and the slice of the pixel array.

In both functions, index refers to the current position within the pixel
array, maximum_index is length of the array, and margin is the value r
if the length of the kernel is 2r+1.
"""
def kernel_boundaries(index, maximum_index, margin):
    kernel_size = 2*margin+1
    # Indices for the slice of the kernel
    kl = max(0,margin-index)
    kr = kernel_size - max(index+margin+1-maximum_index,0)

    return kl, kr


def array_boundaries(index, maximum_index, margin):
    # Indices for the slice of the array
    pl = max(index-margin,0)
    pr = min(index+margin+1,maximum_index)

    return pl, pr


"""
This function implements the operation of convolving an array of pixel values
with a 1D kernel of length 2r+1, applied in the horizontal direction.

Position (j,k) in the array is always matched with the center of the kernel,
position r

If k is at least a distance r from the boundaries of the array, then the 
entire kernel can be matched to a 1 x (2r+1) slice of the array centered on
position (j,k).  We take this slice, multiply it component-wise by the 
kernel, and sum the results.  This number becomes entry (j,k) in the 
convolved image.
    
If position (j,k) is such that a 1 x (2r+1) slice of the array centered on 
this position would exceed the boundaries of the array, then we adjust the 
slice to stop at the boundaries, and we truncate the kernel to match.

For example, if (j,k) = (0,0), then the slice of the array would be
array[0, 0:r+1], and the corresponding slice of the kernel would be
kernel[r:2r+1].

Multiple options, selected by keyword, are available at the boundaries
of the image.

The resulting array is rounded to the nearest integer before it is returned.
"""
def convolve1D_horizontal(data, kernel, boundary_option="renormalize"):
    # Assumes the length of the kernel has the form 2r+1
    kernel_size = kernel.shape[0]
    margin = int((kernel_size-1)/2)

    # Get the dimensions of the pixel array
    numrows, numcols = data.shape

    # The result has the same dimensions as the pixel array
    conv_array = np.zeros((numrows, numcols))

    for ak in range(numcols):
        # Indices for the slices of the pixel array and the kernel
        pl, pr = array_boundaries(ak, numcols, margin)

        if margin <= ak and ak < numcols-margin:
            # no adjustments to the kernel are needed
            for aj in range(numrows):
                # Take the dot product of the slice of the array with the kernel
                conv_array[aj,ak] = np.dot(data[aj,pl:pr],kernel)
                
        else:
            kl, kr = kernel_boundaries(ak, numcols, margin)
            # various boundary options
            if boundary_option == "renormalize":
                # for use if kernel entries are positive and sum to 1
                # the truncated kernel is renormalized to sum to 1
                kernel_sum = np.sum(kernel[kl:kr])
                for aj in range(numrows):
                    conv_array[aj,ak] = np.dot(data[aj,pl:pr],kernel[kl:kr]/kernel_sum)
            if boundary_option == "central-diff":
                # for use in calculating gradient vectors
                # the slice of the pixel array is dotted with the truncated
                # version of the central differences kernel
                for aj in range(numrows):
                    conv_array[aj,ak] = np.dot(data[aj,pl:pr],CDIFF_TRUNC)
                
    return np.rint(conv_array)


"""
This function implements the operation of convolving an array of pixel values
with a 2D kernel of dimensions (2r+1) x (2r+1).

Position (j,k) in the array is always matched with the center of the kernel,
position (r,r).

If j and k are at least a distance r from the boundaries of the array, then
the entire kernel can be matched to a (2r+1) x (2r+1) slice of the array
centered on position (j,k).  We take this slice, multiply it component-wise
by the kernel, and sum the results.  This number becomes entry (j,k) in
the convolved image.
    
If position (j,k) is such that a (2r+1) x (2r+1) slice of the array
centered on this position would exceed the boundaries of the array, then we
adjust the slice to stop at the boundaries, and we truncate the kernel to 
match.

For example, if (j,k) = (0,0), then the slice of the array would be
array[0:r+1, 0:r+1], and the corresponding slice of the kernel would be
kernel[r:2r+1, r:2r+1].

Multiple options, determined by keyword, are available for the boundaries
of the image.

The resulting array is rounded to the nearest integer before it is returned.
"""
def convolve2D_array(data, kernel, boundary_option="renormalize"):
    # Assumes that the dimensions of the kernel have the form (2r+1) x (2r+1)
    kernel_size = kernel.shape[0]
    margin = int((kernel_size-1)/2)
    
    # Get the dimensions of the pixel array
    numrows, numcols = data.shape

    # The result has the same dimensions as the pixel array
    conv_array = np.zeros((numrows, numcols))

    # Iterate over the pixel array to perform the convolution
    for aj in range(numrows):
        # First indices for the slice of the pixel array
        pu, pd = array_boundaries(aj, numrows, margin)

        for ak in range(numcols):    
            # Second indices for the slice of the pixel array
            pl, pr = array_boundaries(ak, numcols, margin)

            if margin <= aj and aj < numrows-margin and margin <= ak and ak < numcols-margin:
                # no adjustments to the kernel are needed
                conv_array[aj,ak] = np.sum(data[pu:pd,pl:pr]*kernel)
            else:
                # indices for the slice of the kernel
                ku, kd = kernel_boundaries(aj, numrows, margin)
                kl, kr = kernel_boundaries(ak, numcols, margin)

                # various boundary options
                if boundary_option == "renormalize":
                    # for use if kernel entries are positive and sum to 1
                    # the truncated kernel is renormalized to sum to 1
                    kernel_sum = np.sum(kernel[ku:kd,kl:kr])
                    conv_array[aj,ak] = np.sum(data[pu:pd,pl:pr]*kernel[ku:kd,kl:kr]/kernel_sum)

                elif boundary_option == "center-shift":
                    # for use with kernels K1 and K2 from class
                    # the center entry of the kernel is shifted
                    # to preserve the kernel sum
                    temp_kernel = kernel.copy()
                    kernel_sum = np.sum(temp_kernel)
                    new_sum = np.sum(temp_kernel[ku:kd,kl:kr])
                    temp_kernel[margin, margin] -= (new_sum - kernel_sum)
                    conv_array[aj,ak] = np.sum(data[pu:pd,pl:pr]*temp_kernel[ku:kd,kl:kr])
              
    return np.rint(conv_array)
   

"""
Receives an image as a PGMFile and a (2r+1)x(2r+1) kernel as a NumPy array.
Returns the PGMFile containing the convolved result.

This operation is suitable for performing 2D weighted averages.  The entries
in the kernel must be positive and sum to 1.  At the boundaries, the kernel is
truncated and the remaining entries are renormalized to sum to 1.
"""
def convolve2D(image, kernel):
    newdata = convolve2D_array(image.data, kernel, boundary_option="renormalize")
    return PGMFile(image.max_shade, newdata)


"""
Receives an image as a PGMFile and a kernel of length 2r+1 as a NumPy array.
Returns the PGMFile containing the result of convolving the given image with
the kernel twice:  once in the horizontal direction and once in the vertical
direction.

This operation is suitable for performing weighted averages with kernels that
are separable.  At the boundaries, the kernel is truncated and the remaining 
entries are renormalized to sum to 1.
"""
def convolve2D_separable(image, kernel):
    step1 = convolve1D_horizontal(image.data,kernel)
    step2 = (convolve1D_horizontal(step1.T,kernel)).T
    return PGMFile(image.max_shade, step2)


"""
Receives an image as a PGMFile and performs edge detection by convolving with 
the kernel K2 given in class.  See the function convolve2D_array() for the
appropriate boundary conditions.

After convolution, take the absolute value of the array, then find the 
maximum value in the result.

Different options are possible for the rest of the image processing.  I have
found by trial and error that if the maximum value in the convolved array is
larger than the original maximum pixel value, then a better image of the edges
results if I clip the array.  On the other hand, if the maximum value in the 
convolved array is smaller than the original maximum pixel value, then the
maximum shade in the resulting image should be set to that smaller value.

You should do *at least one* of clipping the pixel array and resetting the
maximum pixel value.
"""
def edge_detect_K2(image):
    conv2array = convolve2D_array(image.data, K2, boundary_option="center-shift")
    conv2array = np.fabs(conv2array)
    new_max = int(np.amax(conv2array))
    if new_max > image.max_shade:
        conv2array = np.clip(conv2array, 0, image.max_shade)
        new_max = image.max_shade
    return PGMFile(new_max, conv2array)


"""
Given a function of two variables, f(j,k), where j and k are integers, the
central difference approximation to the gradient of the function at the
position (j,k) is

grad f(j,k) = ( ( f(j+1,k)-f(j-1,k) )/2, ( f(j,k+1)-f(j,k-1) )/2 ).

If f is defined for j in [0,N], then the j-component of grad f(0,k) is
f(1,k)-f(0,k), and the j-component of grad f(N,k) is f(N,k)-f(N-1,k).

If f is defined for k in [0,M], then the k-component of grad f(j,0) is
f(j,1)-f(j,0), and the y-component of grad f(j,M) is f(j,M)-f(j,M-1).

Away from the boundaries, each component of the gradient vector can be
obtained by convolution with the kernel [-0.5, 0, 0.5].  The boundary cases
have been added to the 1D convolution function.
 
This function receives a NumPy array of function values whose shape is 
(numrows, numcols).  It returns a NumPy ndarray of shape (numrows, numcols, 2),
where the 2-vector at position (j,k) is the gradient vector grad f(j,k) as 
calculated above.
"""
def gradient_vectors(data):
    numrows, numcols = data.shape

    gradientjk = np.zeros((numrows, numcols, 2))
    
    # the horizontal direction corresponds to the index k
    gradientjk[:,:,1] = convolve1D_horizontal(data, CDIFF, boundary_option="central-diff")
    gradientjk[:,:,0] = (convolve1D_horizontal(data.T, CDIFF, boundary_option="central-diff")).T

    return gradientjk

    
"""
This function receives a NumPy array of gradient vectors of shape (M, N, 2),
and returns a NumPy array of shape (M, N), where each entry is the length of
the gradient 2-vector at that position.
"""
def gradient_lengths(gradientjk):
    return np.rint(np.sqrt(np.sum(gradientjk*gradientjk, axis=2)))    


"""
This function receives a NumPy array of gradient vectors, and a NumPy array of
the lengths of these gradient vectors.  This is redundant since the latter can
be computed from the former.  However, this function is used in Step 2 of the 
Canny Edge Detection algorithm, and the array of gradient vector lengths is
computed in Step 1.  Since it almost certainly exists already, we pass it as
an argument rather than recomputing it.

[JV note:  If you find this inelegant... I agree!  A lot of problems would be 
solved by writing a CannyEdge class.  We're not requiring classes in this 
course, so I'm doing this the hard way, but feel free to improve on my style 
in your solution.]

The values in the array of lengths form a preliminary image of the edges.  At 
each position (j,k), gradientjk has the corresponding gradient vector.  We use
the gradient vectors to thin the edges.

First, we calculate the angle of the gradient vector, where the 
+j-direction corresponds to an angle of 0.  Then we identify angles that
differ by pi, and round the angle to the closest of 0, pi/4, pi/2 and 
3*pi/4.  This rounded angle determines the two nearest neighbors to which
we compare the current pixel value, as follows.

0 -> compare neighbors at positions (j+1,k) and (j-1,k)

pi/4 -> compare neighbors at positions (j+1,k+1) and (j-1,k-1)

pi/2 -> compare neighbors at positions (j,k+1) and (j,k-1)

3*pi/4 -> compare neighbors at positions (j+1,k-1) and (j-1,k+1)

If the current value is smaller than one of these two neighbors, it is not the 
sharpest part of the edge, so we suppress it by setting it to 0.
"""
def thin_edges(gradientjk, length_array):
    numrows, numcols = length_array.shape
    
    # array to hold the results of the edge thinning
    thin_array = np.zeros((numrows, numcols))
    
    """
    At each position (j,k), calculate the angle that the gradient vector makes 
    with the positive j-axis.  Each angle theta is in (-pi, pi].
    """
    theta_array = np.arctan2(gradientjk[:,:,1],gradientjk[:,:,0])
    
    for tj in range(numrows):
        for tk in range(numcols):
            # current pixel value
            length = length_array[tj,tk]

            theta = theta_array[tj,tk]
            """
            Angles that differ by pi are identified.  If theta is negative, we
            replace it by theta + pi.
            """
            if theta < 0:
                theta = theta + math.pi
            
            """
            theta is now an angle in [0, pi].  We round it to one of four bins:
            0, pi/4, pi/2 and 3*pi/4.  Corresponding to this angle, we compare
            the current pixel to its nearest neighbors in the appropriate
            direction.
            
            Pixels on the edges of the image might not have a nearest neighbor
            in one of the required directions.  We use the helper functions for
            convolution to determine the indices of the nearest neighbors, if
            they exist.
            """
            
            # Attempt to take 1 step in the +/- j directions
            jdn, jup = array_boundaries(tj, numrows, 1)

            # Attempt to take 1 step in the +/- k directions
            kdn, kup = array_boundaries(tk, numcols, 1)

            # array_boundaries() returns the upper index for a slice
            # in this context, we want a position, so adjust down by 1
            jup = jup - 1
            kup = kup - 1
            
            up_pixel = dn_pixel = 0
            
            if theta < math.pi/8. or theta >= 7*math.pi/8:
                #theta rounds to 0 - check +/- j
                up_pixel = length_array[jup,tk]
                dn_pixel = length_array[jdn,tk]
            elif theta < 3*math.pi/8:
                #theta rounds to pi/4 - check (+1,+1) and (-1,-1)
                up_pixel = length_array[jup,kup]
                dn_pixel = length_array[jdn,kdn]
            elif theta < 5*math.pi/8:
                #theta rounds to pi/2 - check +/- k
                up_pixel = length_array[tj,kup]
                dn_pixel = length_array[tj,kdn]
            elif theta < 7*math.pi/8:
                #theta rounds to 3*pi/4 - check (+1,-1) and (-1,+1)
                up_pixel = length_array[jup,kdn]
                dn_pixel = length_array[jdn,kup]
    
            # Only keep the current value if it is a local max along the 
            # direction of the gradient
            if length >= up_pixel and length >= dn_pixel:
                # Local max, keep this value
                thin_array[tj,tk] = length
    
    return thin_array


"""
This function receives a NumPy array containing the pixel data for an image
with thinned edges, and it suppresses noise using low and high cutoffs.
 - If a pixel value is below the low cutoff, suppress it by setting it to 0.
 - If a pixel is above the low cutoff but below the high cutoff, suppress it
   unless it has a pixel above the high cutoff among its eight nearest
   neighbors.
The function returns a NumPy array with the pixel values after noise
suppression.
"""
def suppress_noise(thin_data, low_threshold, high_threshold):

    numrows, numcols = thin_data.shape
    
    suppr_data = np.zeros((numrows, numcols))
    
    for tj in range(numrows):
        for tk in range(numcols):
            pixel = thin_data[tj,tk]

            if pixel >= high_threshold:
                """
                This edge is strong:  keep it
                """
                suppr_data[tj,tk] = pixel
            elif pixel >= low_threshold:
                """
                This edge is too weak to stand alone, but it will be kept if
                it is adjacent to a strong edge.
                Consider its neighbors one step in every direction, if they are
                within the boundaries of the array.  If any one of these
                neighbors is greater than the high threshold, we keep the 
                current edge.
                """
                jdn, jup = array_boundaries(tj, numrows, 1)
                kdn, kup = array_boundaries(tk, numcols, 1)
                
                if np.amax(thin_data[jdn:jup,kdn:kup]) >= high_threshold:
                    suppr_data[tj,tk] = pixel
                
    return suppr_data




