"""
A collection of functions to create a randomly generated library of smiley faces
J Vaughan
"""


import math
import numpy as np
import imagefuncs_A3 as imf
from collections import namedtuple

BASENAME = "Faces/face_"

FILETYPE = 'P2'
SIZE = 200
MAX_SHADE = 255
HOWMANYFILES = 100
BORDER = 20

HALFSIZE = int(SIZE/2)
QUARTSIZE = int(SIZE/4)

"""
The data required to define a circle are:
radius
x-cooordinate of center
y-coordinate of center
"""    
circle_data = namedtuple("circle_data", ["radius", "xcoord", "ycoord"])


"""
The data required to define a crescent are:
radius of outer circle
radius of inner circle
x-coordinate of center
y-coordinate of center
fraction of crescent to be printed - see WriteACrescent
"""
crescent_data = namedtuple("crescent_data", ["outer_radius", "inner_radius", "xcoord", "ycoord", "fraction"])


"""
The data required to define a face are:
main circle
left eye
right eye
nose
mouth
"""
face_data = namedtuple("face_data", ["main_circle", "left_eye", "right_eye", "nose", "mouth"])


"""
The means and stddevs below are for use in the random generation of faces.
The values are the result of trial and error.
"""
MAIN_RADIUS_MEAN = int(HALFSIZE*0.95) # 95
EYE_RADIUS_MEAN = int(HALFSIZE*0.1) # 10
LEFTEYE_X_MEAN = int(QUARTSIZE*1.2) # 60
RIGHTEYE_X_MEAN = SIZE - LEFTEYE_X_MEAN # 140
LEFTEYE_Y_MEAN = RIGHTEYE_Y_MEAN = int(QUARTSIZE*1.2) # 60
NOSE_RADIUS_MEAN = int(EYE_RADIUS_MEAN*0.8) # 8
NOSE_X_MEAN = NOSE_Y_MEAN = HALFSIZE
MOUTH_OUTER_MEAN = QUARTSIZE # 50
MOUTH_INNER_MEAN = int(QUARTSIZE*0.92) # 46
MOUTH_X_MEAN = MOUTH_Y_MEAN = HALFSIZE # 100
MOUTH_FRAC_MEAN = -0.25

BIG_STDDEV = 3
SM_STDDEV = BIG_STDDEV/2
MED_STDDEV = 2
TINY_STDDEV = MED_STDDEV/2

MAIN_RADIUS_STDDEV = BIG_STDDEV
EYE_RADIUS_STDDEV = TINY_STDDEV
LEFTEYE_X_STDDEV = SM_STDDEV
LEFTEYE_Y_STDDEV = SM_STDDEV
RIGHTEYE_X_STDDEV = SM_STDDEV
RIGHTEYE_Y_STDDEV = SM_STDDEV
NOSE_RADIUS_STDDEV = TINY_STDDEV
NOSE_X_STDDEV = SM_STDDEV
NOSE_Y_STDDEV = SM_STDDEV
MOUTH_OUTER_STDDEV = MED_STDDEV
MOUTH_INNER_STDDEV = TINY_STDDEV
MOUTH_X_STDDEV = SM_STDDEV
MOUTH_Y_STDDEV = SM_STDDEV*0.85
MOUTH_FRAC_STDDEV = 0.075


# Gaussian kernel to smooth each image
GAUSSs12 = np.array([4, 22, 60, 85, 60, 22, 4])/257  #sigma = 1.2


"""
We use the following axes:

------> x
|
|
|
\/ y

Shifting in the positive y direction is equivalent to moving down the image
"""


"""
This function receives an array of pixel data and draws a circle on it.

circle_data contains the radius of the circle and the x- and y- coordinates 
of the center of the circle.  All of these values should be integers.

By default, the pixels making up the circle are set to 0.  If black is
set to False, the pixels making up the circle are set to MAX_SHADE.
"""
def WriteACircle(circle, array, black=True):
    
    for ay in range(-circle.radius,circle.radius):
        # ay is the y-coordinate, measured with respect to the center
        # of the circle.        
        # at height ay, the horizontal distance from the y-axis to the 
        # boundary of the circle is sqrt(r^2 - ay^2)
        boundx = math.sqrt(circle.radius*circle.radius - ay*ay)
        for ax in range(-circle.radius,circle.radius):
            if -boundx < ax and ax < boundx:
                if black:
                    array[circle.ycoord+ay, circle.xcoord+ax]=0
                else:
                    array[circle.ycoord+ay, circle.xcoord+ax]=MAX_SHADE


"""
This function receives an array of pixel data and draws a crescent on it.
A crescent is bounded by two concentric circles.

crescent_data contains the outer and inner radii of the crescent, the x- and 
y-coordinates of the center of the crescent, and the fraction of the crescent 
to be printed.  All of these values except the fraction should be integers.

The fraction of the crescent should be a value between -1 and 1.

If fraction > 0, we start printing from the TOP of the circle:  fraction = 1/2
gives the top half of the circle, fraction = 1 gives the entire circle.

If fraction < 0, we start printing from the BOTTOM of the circle:
fraction = -1/2 gives the lower half of the circle, fraction = -1 gives the 
entire circle

By default, the pixels making up the crescent are set to 0.  If black is set
to False, the pixels making up the crescent are set to MAX_SHADE.
"""
def WriteACrescent(crescent, array, black=True):
    outlowbdy = -crescent.outer_radius
    outupbdy = crescent.outer_radius
    if crescent.fraction < 0:
        outlowbdy = crescent.outer_radius + int(crescent.outer_radius*2*crescent.fraction)
    elif crescent.fraction > 0:
        outupbdy = crescent.outer_radius - int(crescent.outer_radius*2*crescent.fraction)
    else:
        outlowbdy = outupbdy = 0
        
    for ay in range(outlowbdy,outupbdy):
        # ay is the y-coordinate, measured with respect to the center of the
        # crescent
        # at height ay, the horizontal distance from the y-axis to the 
        # boundary of the outer circle is sqrt(outradius^2 - ay^2).
        outboundx = math.sqrt(crescent.outer_radius*crescent.outer_radius - ay*ay)
        inboundx = 0.0
        if math.fabs(ay) <= crescent.inner_radius:
            # ay is inside the inner circle
            # at height ay, the horizontal distance from the y-axis to the
            # boundary of the inner circle is sqrt(inradius^2 - ay^2).
            inboundx = math.sqrt(crescent.inner_radius*crescent.inner_radius - ay*ay)
        for ax in range(-crescent.outer_radius,crescent.outer_radius):
            if -outboundx <= ax and ax < outboundx:
                # inside the outer circle
                if ax <= -inboundx or inboundx < ax:
                    # outside the inner circle
                    if black:
                        array[crescent.ycoord+ay, crescent.xcoord+ax]=0
                    else:
                        array[crescent.ycoord+ay, crescent.xcoord+ax]=MAX_SHADE


"""
This function draws a face!
"""
def DrawFace(face):
    # starting point:  a pure black image
    array = np.zeros((SIZE, SIZE))

    # Print the large white circle
    WriteACircle(face.main_circle, array, False)
    
    # Print the left eye
    WriteACircle(face.left_eye, array)
    
    # Print the right eye
    WriteACircle(face.right_eye, array)
    
    # Print the nose
    WriteACircle(face.nose, array)
    
    # Print the mouth
    WriteACrescent(face.mouth, array)
    
    return array


"""
This function produces a randomly generated face, using the various means and
stddevs that are laboriously defined at the beginning of this file.
"""
def RandomFaceData():
    # main circle
    # the radius of the circle must be no more than half the size of the image
    main_radius = SIZE
    while main_radius > HALFSIZE:
        main_radius = np.random.normal(MAIN_RADIUS_MEAN, MAIN_RADIUS_STDDEV)
    main_circle = circle_data(radius=int(round(main_radius)), xcoord=HALFSIZE, ycoord=HALFSIZE)
    
    # left eye
    lefteye_radius = np.random.normal(EYE_RADIUS_MEAN, EYE_RADIUS_STDDEV)
    lefteye_x = np.random.normal(LEFTEYE_X_MEAN, LEFTEYE_X_STDDEV)
    lefteye_y = np.random.normal(LEFTEYE_Y_MEAN, LEFTEYE_Y_STDDEV)
    left_eye = circle_data(radius=int(round(lefteye_radius)), xcoord=int(round(lefteye_x)), ycoord=int(round(lefteye_y)))
    
    # right eye
    # righteye_radius = np.random.normal(EYE_RADIUS_MEAN, EYE_RADIUS_STDDEV)
    righteye_radius = lefteye_radius
    righteye_x = np.random.normal(RIGHTEYE_X_MEAN, RIGHTEYE_X_STDDEV)
    righteye_y = np.random.normal(RIGHTEYE_Y_MEAN, RIGHTEYE_Y_STDDEV)
    right_eye = circle_data(radius=int(round(righteye_radius)), xcoord=int(round(righteye_x)), ycoord=int(round(righteye_y)))
    
    # nose
    nose_radius = np.random.normal(NOSE_RADIUS_MEAN, NOSE_RADIUS_STDDEV)
    nose_x = np.random.normal(NOSE_X_MEAN, NOSE_X_STDDEV)
    nose_y = np.random.normal(NOSE_Y_MEAN, NOSE_Y_STDDEV)
    nose = circle_data(radius=int(round(nose_radius)), xcoord=int(round(nose_x)), ycoord=int(round(nose_y)))
    
    # mouth
    # If the outer radius of the mouth isn't at least one unit bigger
    # than the inner radius, then the face will have no mouth.
    mouth_outer = np.random.normal(MOUTH_OUTER_MEAN, MOUTH_OUTER_STDDEV)
    mouth_inner = mouth_outer
    while mouth_inner >= mouth_outer-1:
        mouth_inner = np.random.normal(MOUTH_INNER_MEAN, MOUTH_INNER_STDDEV)

    mouth_x = np.random.normal(MOUTH_X_MEAN, MOUTH_X_STDDEV)
    mouth_y = np.random.normal(MOUTH_Y_MEAN, MOUTH_Y_STDDEV)

    # If the mouth fraction is nonnegative, the image will be ridiculous
    fraction = 1
    while fraction >= 0:
        fraction = np.random.normal(MOUTH_FRAC_MEAN, MOUTH_FRAC_STDDEV)

    mouth = crescent_data(outer_radius=int(round(mouth_outer)), inner_radius=int(round(mouth_inner)), xcoord=int(round(mouth_x)), ycoord=int(round(mouth_y)), fraction=fraction)

    face = face_data(main_circle, left_eye, right_eye, nose, mouth)
    
    return face


"""
Draw one new random face
"""
def DrawRandomFace():
    face = RandomFaceData()
    sharp_data = DrawFace(face)
    sharp_image = imf.PGMFile(data=sharp_data, max_shade=MAX_SHADE)
    smooth_face = imf.convolve2D_separable(sharp_image, GAUSSs12)
    return smooth_face


"""
This function handles the random generation of a library of faces,
as well as the printing of the files.
"""
def CreateFaceLibrary():
    
    xi = 1
    while xi <= HOWMANYFILES:
        current_name = BASENAME + str(xi) + '.pgm'
        smooth_face = DrawRandomFace()
        imf.write_image(current_name, smooth_face)
        xi += 1

    return        

CreateFaceLibrary()