###
### File created by Joschua Simon-Liedtke on 22nd of January 2014
###

from PIL import Image, ImageDraw, ImageEnhance
import numpy
from scipy.misc import imresize
import colour
from pylab import *
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata
import os
import settings
import threading
import socket
import time
from colordeficiency_tools import getVariantFromImage, getStatsFromFilename, setStatsToFilename
from colordeficiency_tools import dxp1, dxm1, dyp1, dym1, computeDaltonizationUnitVectors, computeChiAndLambda, anisotropicG, multiscaling, GRMSE, makeGaussianChromaMap, makeGaussianSkinMap


try:
    from pymatbridge import Matlab
except ImportError:
    print 'Daltonization method ??? cannot be used.'
    pass
import sys

path_matlab = os.path.join(settings.module_path,'code','Matlab','implementation')
path_matlab_alsam = os.path.join(settings.module_path,'code','Matlab','alsam')
module_path = settings.module_path

def coldefType2Int(coldef_type):
    
    if coldef_type == "p": return 1
    elif coldef_type == "d": return 2
    elif coldef_type == "t": return 3
    else: return 0

def convertToLuminanceImage(img_in, options):
    
    img_in = img_in.convert('RGB')
    img_in_array = numpy.asarray(img_in, dtype=float)
    
    
    img_out_array = numpy.dot(img_in_array[...,:3], [0.299, 0.587, 0.144])
    img_out_array[img_out_array<0.0] = 0.0
    img_out_array[img_out_array>255.0] = 255.0
    
    img_out = Image.fromarray(numpy.uint8(img_out_array))
    
    return img_out

def makeLMSDeficientMatrix(rgb2lms, coldef_type):
    """
    Make color deficiency reduction matrices based on the on the algorithms proposed by Brettel and Vienot
    """
    
    l = numpy.array([1,0,0])
    m = numpy.array([0,1,0])
    s = numpy.array([0,0,1])
    
    # alle rgb[0][...] - L, alle rgb[1][...] - M, alle rgb[2][...] - S
    # alle rgb[...][0] - R, alle rgb[...][1] - G, alle rgb[...][2] - B 
    
    if coldef_type == "p":
        #Matrix for protanopes
        alpha   = sum(rgb2lms[1][0:3]) * rgb2lms[2][2]- rgb2lms[1][2] * sum(rgb2lms[2][0:3])
        beta    = sum(rgb2lms[2][0:3]) * rgb2lms[0][2]- rgb2lms[2][2] * sum(rgb2lms[0][0:3])
        gamma   = sum(rgb2lms[0][0:3]) * rgb2lms[1][2]- rgb2lms[0][2] * sum(rgb2lms[1][0:3])
        
        l_p = (-1.)/( alpha ) * numpy.array([0,beta,gamma])
        l = l_p 
        
        m = numpy.array([0,1,0])
        s = numpy.array([0,0,1])
        
    elif coldef_type == "d":
        #Matrix for deuteranopes
        alpha   = sum(rgb2lms[1][0:3]) * rgb2lms[2][2]- rgb2lms[1][2] * sum(rgb2lms[2][0:3])
        beta    = sum(rgb2lms[2][0:3]) * rgb2lms[0][2]- rgb2lms[2][2] * sum(rgb2lms[0][0:3])
        gamma   = sum(rgb2lms[0][0:3]) * rgb2lms[1][2]- rgb2lms[0][2] * sum(rgb2lms[1][0:3])
        
        l = numpy.array([1,0,0])
        
        m_d = (-1.)/( beta ) * numpy.array([alpha,0,gamma])
        m = m_d
        
        s = numpy.array([0,0,1])
        
    elif coldef_type == "t":
        #Matrix for tritanopes
        alpha   = sum(rgb2lms[1][0:3]) * rgb2lms[2][0]- rgb2lms[1][0] * sum(rgb2lms[2][0:3])
        beta    = sum(rgb2lms[2][0:3]) * rgb2lms[0][0]- rgb2lms[2][0] * sum(rgb2lms[0][0:3])
        gamma   = sum(rgb2lms[0][0:3]) * rgb2lms[1][0]- rgb2lms[0][0] * sum(rgb2lms[1][0:3])
        
        l = numpy.array([1,0,0])
        m = numpy.array([0,1,0])
        
        s_t = (-1.)/( gamma ) * numpy.array([alpha,beta,0])
        s = s_t
    else:
        print "Error: unknown color deficiency chosen. Chose either p for protanopes, d for deuteranopes, or t for tritanopes."
   
    matrix = numpy.array([l,m,s])
    #print matrix
    
    return matrix
    
#print makeLMSDeficientMatrix(rgb2lms_vienot,"d")   

def makeLMSDeficientMatrix_brettel(constantHues_LMS, whitepoint_LMS, coldef_type):
    """
    Make color deficiency reduction matrices based on the on the algorithms proposed by Brettel and Vienot
    """
    
    l = numpy.array([1,0,0])
    m = numpy.array([0,1,0])
    s = numpy.array([0,0,1])
    
    # alle lms[0] - L, lms[1] - M, lms[2] - S
    
    if coldef_type == "p":
        #Matrix for protanopes
        
        #Compute matrix for b475  
        b475_LMS = constantHues_LMS['b475']
        
        alpha   = whitepoint_LMS[1] * b475_LMS[2] - whitepoint_LMS[2] * b475_LMS[1]
        beta    = whitepoint_LMS[2] * b475_LMS[0] - whitepoint_LMS[0] * b475_LMS[2]
        gamma   = whitepoint_LMS[0] * b475_LMS[1] - whitepoint_LMS[1] * b475_LMS[0]
        
        l_p = (-1.)/( alpha ) * numpy.array([0,beta,gamma])
        l = l_p 
        
        m = numpy.array([0,1,0])
        s = numpy.array([0,0,1])
        
        matrix_b475 = numpy.array([l,m,s])
        
        #Compute matrix for y575
        
        y575_LMS = constantHues_LMS['y575']
        
        alpha   = whitepoint_LMS[1] * y575_LMS[2] - whitepoint_LMS[2] * y575_LMS[1]
        beta    = whitepoint_LMS[2] * y575_LMS[0] - whitepoint_LMS[0] * y575_LMS[2]
        gamma   = whitepoint_LMS[0] * y575_LMS[1] - whitepoint_LMS[1] * y575_LMS[0]
        
        l_p = (-1.)/( alpha ) * numpy.array([0,beta,gamma])
        l = l_p 
        
        m = numpy.array([0,1,0])
        s = numpy.array([0,0,1])
        
        matrix_y575 = numpy.array([l,m,s])
        
        # Combine matrices
        matrix = {}
        matrix['b475'] = matrix_b475
        matrix['y575'] = matrix_y575
        
        
    elif coldef_type == "d":
        #Matrix for deuteranopes
        
        #Compute matrix for b475  
        b475_LMS = constantHues_LMS['b475']
        
        alpha   = whitepoint_LMS[1] * b475_LMS[2] - whitepoint_LMS[2] * b475_LMS[1]
        beta    = whitepoint_LMS[2] * b475_LMS[0] - whitepoint_LMS[0] * b475_LMS[2]
        gamma   = whitepoint_LMS[0] * b475_LMS[1] - whitepoint_LMS[1] * b475_LMS[0]
        
        l = numpy.array([1,0,0])
        
        m_d = (-1.)/( beta ) * numpy.array([alpha,0,gamma])
        m = m_d
        
        s = numpy.array([0,0,1])
        
        matrix_b475 = numpy.array([l,m,s])
        
        #Compute matrix for y575
        
        y575_LMS = constantHues_LMS['y575']
        
        alpha   = whitepoint_LMS[1] * y575_LMS[2] - whitepoint_LMS[2] * y575_LMS[1]
        beta    = whitepoint_LMS[2] * y575_LMS[0] - whitepoint_LMS[0] * y575_LMS[2]
        gamma   = whitepoint_LMS[0] * y575_LMS[1] - whitepoint_LMS[1] * y575_LMS[0]
        
        l = numpy.array([1,0,0])
        
        m_d = (-1.)/( beta ) * numpy.array([alpha,0,gamma])
        m = m_d
        
        s = numpy.array([0,0,1])
        
        matrix_y575 = numpy.array([l,m,s])
        
        # Combine matrices
        matrix = {}
        matrix['b475'] = matrix_b475
        matrix['y575'] = matrix_y575
        
    elif coldef_type == "t":
        #Matrix for tritanopes
        
        #Compute matrix for r660  
        r660_LMS = constantHues_LMS['r660']
        
        alpha   = whitepoint_LMS[1] * r660_LMS[2] - whitepoint_LMS[2] * r660_LMS[1]
        beta    = whitepoint_LMS[2] * r660_LMS[0] - whitepoint_LMS[0] * r660_LMS[2]
        gamma   = whitepoint_LMS[0] * r660_LMS[1] - whitepoint_LMS[1] * r660_LMS[0]
        
        l = numpy.array([1,0,0])
        m = numpy.array([0,1,0])
        
        s_t = (-1.)/( gamma ) * numpy.array([alpha,beta,0])
        s = s_t
        
        matrix_r660 = numpy.array([l,m,s])
        
        #Compute matrix for c485
        
        c485_LMS = constantHues_LMS['c485']
        
        alpha   = whitepoint_LMS[1] * c485_LMS[2] - whitepoint_LMS[2] * c485_LMS[1]
        beta    = whitepoint_LMS[2] * c485_LMS[0] - whitepoint_LMS[0] * c485_LMS[2]
        gamma   = whitepoint_LMS[0] * c485_LMS[1] - whitepoint_LMS[1] * c485_LMS[0]
        
        l = numpy.array([1,0,0])
        m = numpy.array([0,1,0])
        
        s_t = (-1.)/( gamma ) * numpy.array([alpha,beta,0])
        s = s_t
        
        matrix_c485 = numpy.array([l,m,s])
        
        # Combine matrices
        matrix = {}
        matrix['r660'] = matrix_r660
        matrix['c485'] = matrix_c485
        
    else:
        print "Error: unknown color deficiency chosen. Chose either p for protanopes, d for deuteranopes, or t for tritanopes."

    return matrix

def crossOut(img_in):
    
    is_numpy_array = type(img_in)==numpy.ndarray
    
    if is_numpy_array:
        img_in = numpy.uint8(img_in*256)
        img_in = Image.fromarray(img_in)
    
    im = img_in.copy()
    converter = ImageEnhance.Color(im)
    im = converter.enhance(0.33)
    del converter
    
    
    draw = ImageDraw.Draw(im)
    draw.line((0, 0) + im.size, fill=128, width=10)
    draw.line((0, im.size[1], im.size[0], 0), fill=128, width=10)
    del draw
    
    if is_numpy_array:
        im = im.convert('RGB')
        im = numpy.asarray(im, dtype=float)/255.
    
    return im

def simulation_vienot(img_in, coldef_type,coldef_strength=1.0):
    """
    This is a colour deficiency simulation for deuteranopes and protanopes based on 'Computerized ...' by Francoise Vienot et al.
    Input:  img_in -          Original PIL image
            coldef_type -     Type of color deficiency. Either p for protanopes, d for deuteranopes, or t for tritanopes.
            coldef_strength - Severity of color deficiency where 1.0 are completely and 0.0 is nothing. OBS: This algorithm only allows full severity!
    Output: img_out -         Simulated PIL image
    """
    
    if not (coldef_type == "p" or coldef_type == "d"):
        print "Error: unknown color deficiency chosen. Chose either p for protanopes, or d for deuteranopes."
        
        img_out = crossOut(img_in)
        return img_out
    
    img_in = img_in.convert('RGB')
    img_array = (numpy.asarray(img_in, dtype=float)/255.)**2.2
    m,n,dim = numpy.shape(img_array)
    
    # Modified RGB space based on ITU-R BT.709 primaries - same as sRGB - and Judd-Vos colorimetric modification
    rgb2xyz = numpy.array([[40.9568, 35.5041,17.9167],
                           [21.3389, 70.6743, 7.9868],
                           [ 1.86297,11.462, 91.2367]])
    vienotRGBSpaceLinear = colour.space.TransformLinear(colour.space.xyz,numpy.linalg.inv(rgb2xyz))
    vienotRGBOriginal_arr = colour.data.Data(vienotRGBSpaceLinear, img_array)
    XYZOriginal_arr = colour.data.Data(colour.space.xyz,vienotRGBOriginal_arr.get(colour.space.xyz))
        
    # LMS space based on Smith and Pokorny
    xyz2lms = numpy.array([[.15514, .54312, -.03286],
                           [-.15514, .45684, .03286],
                           [0, 0, .01608]])
    lmsSpace = colour.space.TransformLinear(colour.space.xyz,xyz2lms) #.01608 .00801
    lmsOriginal_arr = XYZOriginal_arr.get(lmsSpace)
    
    rgb2lms = numpy.dot(xyz2lms,rgb2xyz)
    #print rgb2lms
    lms2lms_deficient = makeLMSDeficientMatrix(rgb2lms, coldef_type)
    
    # This is the actual simulation
    lmsOriginal_vector = numpy.reshape(lmsOriginal_arr,(m*n,3))
    lmsSimulated_vector = numpy.dot(lmsOriginal_vector, lms2lms_deficient.transpose())
    lmsSimulated_arr = colour.data.Data(lmsSpace, numpy.reshape(lmsSimulated_vector, (m,n,3)))
    
    # We propose this gamut clipping instead for hte one proposed by vienot
    rgbVienot_arr = lmsSimulated_arr.get(vienotRGBSpaceLinear)
    rgbVienot_arr[rgbVienot_arr<0] = 0
    rgbVienot_arr[rgbVienot_arr>1] = 1
    
    vienotRGBSimulated_arr = (rgbVienot_arr**(1/2.2))*255.
    img_array = numpy.uint8(vienotRGBSimulated_arr)
    
    img_out = Image.fromarray(img_array)
    
    return img_out

def simulation_vienot_adjusted(img_in, coldef_type,coldef_strength=1.0):
    """
    This is a colour deficiency simulation for deuteranopes and protanopes based on 'Computerized ...' by Francoise Vienot et al.
    Some variations have been made: instead of using the gamma correction proposed in the paper, we use the standard sRGB conversion from sRGB to XYZ and we used an adjusted XYZ to LMS matrix.
    Input:  img_in -          Original PIL image
            coldef_type -     Type of color deficiency. Either p for protanopes, d for deuteranopes, or t for tritanopes.
            coldef_strength - Severity of color deficiency where 1.0 are completely and 0.0 is nothing. OBS: This algorithm only allows full severity!
    Output: img_out -         Simulated PIL image
    """
    
    if not (coldef_type == "p" or coldef_type == "d" or coldef_type == "t"):
        print "Error: Unknown color deficiency chosen. Chose either p for protanopes, d for deuteranopes, or t for tritanopes."
        return img_in
    
    img_in = img_in.convert('RGB')
    img_array = numpy.asarray(img_in, dtype=float)/255.
    m,n,dim = numpy.shape(img_array)
    
    # Modified RGB space based on ITU-R BT.709 primaries - same as sRGB - and Judd-Vos colorimetric modification
    xyz2rgb = numpy.array([[ 3.2404542, -1.5371385, -0.4985314],
                           [-0.9692660,  1.8760108,  0.0415560],
                           [ 0.0556434, -0.2040259,  1.0572252]])
    rgb2xyz = numpy.linalg.inv(xyz2rgb)
    sRGBOriginal_arr = colour.data.Data(colour.space.srgb, img_array)
       
    # LMS space based on Smith and Pokorny
    xyz2lms = numpy.array([[.15514, .54312, -.03286],
                           [-.15514, .45684, .03286],
                           [0., 0., .00801]])
    lmsSpace = colour.space.TransformLinear(colour.space.xyz,xyz2lms) #.01608 .00801
    lmsOriginal_arr = sRGBOriginal_arr.get(lmsSpace)
    
    rgb2lms = numpy.dot(xyz2lms,rgb2xyz)*100.
    lms2lms_deficient = makeLMSDeficientMatrix(rgb2lms, coldef_type)
    
    # This is the actual simulation
    lmsOriginal_vector = numpy.reshape(lmsOriginal_arr,(m*n,3))
    lmsSimulated_vector = numpy.dot(lmsOriginal_vector, lms2lms_deficient.transpose())
    lmsSimulated_arr = colour.data.Data(lmsSpace, numpy.reshape(lmsSimulated_vector, (m,n,3)))
    
    # We propose this gamut clipping instead for hte one proposed by vienot
    sRGBSimulated_arr = lmsSimulated_arr.get(colour.space.srgb)*255.
    img_array = numpy.uint8(sRGBSimulated_arr)
    img_out = Image.fromarray(img_array)
    
    return img_out

#simulate(img_in,"d","videnot").show()

def simulation_kotera(img_in, coldef_type, coldef_strength=1.):
    """
    Function to simulate color deficiency for vienot, vienot adjusted.
    Input:  simulation_type - Type of simulation as defined in simulation_types
            img_in -          Original PIL image
            coldef_type -     Type of color deficiency. Either p for protanopes, d for deuteranopes, or t for tritanopes.
            coldef_strength - Severity of color deficiency where 1.0 are completely and 0.0 is nothing. 
    Output: img_out -         Simulated PIL image
    """
    
    if not (coldef_type == "p" or coldef_type == "d" or coldef_type == "t"):
        print "Error: Unknown color deficiency chosen. Chose either p for protanopes, d for deuteranopes, or t for tritanopes."
        return img_in
    
    img_in = img_in.convert('RGB')
    img_array = numpy.asarray(img_in, dtype=float)/255.
    
    sRGBOriginal_arr = colour.data.Data(colour.space.srgb,img_array)
    iptOriginal_arr = sRGBOriginal_arr.get(colour.space.ipt)
    if (coldef_type == "p" or coldef_type == "d"):
        iptSimulated_arr = iptOriginal_arr
        iptSimulated_arr[:,:,1] = iptOriginal_arr[:,:,1]*(1.0-coldef_strength)
    else:
        iptSimulated_arr = iptOriginal_arr
        iptSimulated_arr[:,:,2] = iptOriginal_arr[:,:,2]*(1.0-coldef_strength)
    iptSimulated_arr = colour.data.Data(colour.space.ipt,iptSimulated_arr)
    sRGBSimulated_arr = iptSimulated_arr.get(colour.space.srgb)*255.
    
    img_array = numpy.uint8(sRGBSimulated_arr)
    img_out = Image.fromarray(img_array)
    
    return img_out
    

def simulation_brettel(img_in, coldef_type, coldef_strength=1.0):
    
    
    # Check if correct color deficiency has been chosen
    if not (coldef_type == "p" or coldef_type == "d" or coldef_type == "t"):
        print "Error: Unknown color deficiency type chosen. Chose either p for protanopes, d for deuteranopes, or t for tritanopes."
        return img_in
    
    is_numpy_array = type(img_in)==numpy.ndarray
    #if is_numpy_array:
    #    print "Note: This is a numpy array."
    #else:
    #    print "Note: This is a PIL Image."
        
    
    #print settings.data_path
    data = numpy.genfromtxt(os.path.join(settings.data_path,'ciexyz31.csv'), delimiter=',')
    # LMS space based on Smith and Pokorny
    xyz2lms = numpy.array([[.15514, .54312, -.03286],
                           [-.15514, .45684, .03286],
                           [0., 0., .00801]])
    
    # Non-confusion equilluminant stimuli for protanopes, detueranopes and tritanopes 
    equiwhite_XYZ = [sum(data[:,1]),sum(data[:,2]),sum(data[:,3])]
    k_norm = equiwhite_XYZ[1]
    equiwhite_XYZ = equiwhite_XYZ/k_norm
    equiwhite_LMS = numpy.dot(xyz2lms,equiwhite_XYZ)
    #print equiwhite_XYZ 
    
    # Non-confusion colors for protanopes and deuteranopes
    y575_XYZ =  data[data[:,0]==575.][0,1:4]
    y575_LMS = numpy.dot(xyz2lms,y575_XYZ)
    
    b475_XYZ =  data[data[:,0]==475.][0,1:4]
    b475_LMS = numpy.dot(xyz2lms,b475_XYZ)
    
    # Non-confusion colors for tritanopes
    r660_XYZ =  data[data[:,0]==660.][0,1:4]
    r660_LMS = numpy.dot(xyz2lms,r660_XYZ)
    
    c485_XYZ =  data[data[:,0]==485.][0,1:4]
    c485_LMS = numpy.dot(xyz2lms,c485_XYZ)
    
    if is_numpy_array: img_array = img_in
    else: img_in = img_in.convert('RGB'); img_array = numpy.asarray(img_in, dtype=float)/255.
    m,n,dim = numpy.shape(img_array)
    
    # Modified RGB space based on ITU-R BT.709 primaries - same as sRGB - and Judd-Vos colorimetric modification
    xyz2rgb = numpy.array([[ 3.2404542, -1.5371385, -0.4985314],
                           [-0.9692660,  1.8760108,  0.0415560],
                           [ 0.0556434, -0.2040259,  1.0572252]])
    rgb2xyz = numpy.linalg.inv(xyz2rgb)
    sRGBOriginal_arr = colour.data.Data(colour.space.srgb, img_array)
       
    # LMS space based on Smith and Pokorny
    lmsSpace = colour.space.TransformLinear(colour.space.xyz,xyz2lms) #.01608 .00801
    lmsOriginal_arr = sRGBOriginal_arr.get(lmsSpace)
    
    #rgb2lms = numpy.dot(xyz2lms,rgb2xyz)*100.
    #lms2lms_deficient = makeLMSDeficientMatrix(rgb2lms, coldef_type)
    
    # This is the actual simulation
    lmsOriginal_vector = numpy.reshape(lmsOriginal_arr,(m*n,3))

    #Avoid division by 0
    eps = 0.0000000000001
    lmsOriginal_vector[lmsOriginal_vector == 0.0] = eps
    
    if coldef_type == "p":
        constantHues_LMS = {}
        constantHues_LMS['y575'] = y575_LMS
        constantHues_LMS['b475'] = b475_LMS
        
        lms2lms_deficient = makeLMSDeficientMatrix_brettel(constantHues_LMS, equiwhite_LMS, coldef_type)
        lms2lms_y575 = lms2lms_deficient['y575']
        lms2lms_b475 = lms2lms_deficient['b475']
        
        neutral_ratio = equiwhite_LMS[2]/equiwhite_LMS[1]
        indices = lmsOriginal_vector[:,2]/lmsOriginal_vector[:,1] < neutral_ratio
        not_indices = indices == False
        
        lmsOriginalY575_vector = lmsOriginal_vector[indices,0:4]
        lmsSimulatedY575_vector = numpy.dot(lmsOriginalY575_vector, lms2lms_y575.transpose()) 
        
        lmsOriginalB475_vector = lmsOriginal_vector[not_indices,0:4]
        lmsSimulatedB475_vector = numpy.dot(lmsOriginalB475_vector, lms2lms_b475.transpose()) 
        
        lmsSimulated_vector = lmsOriginal_vector.copy()
        lmsSimulated_vector[indices] = lmsSimulatedY575_vector
        lmsSimulated_vector[not_indices] = lmsSimulatedB475_vector
        
    elif coldef_type == "d":
        constantHues_LMS = {}
        constantHues_LMS['y575'] = y575_LMS
        constantHues_LMS['b475'] = b475_LMS
        
        lms2lms_deficient = makeLMSDeficientMatrix_brettel(constantHues_LMS, equiwhite_LMS, coldef_type)
        lms2lms_y575 = lms2lms_deficient['y575']
        lms2lms_b475 = lms2lms_deficient['b475']
        
        neutral_ratio = equiwhite_LMS[2]/equiwhite_LMS[0]
        indices = lmsOriginal_vector[:,2]/lmsOriginal_vector[:,0] < neutral_ratio
        not_indices = indices == False
        
        lmsOriginalY575_vector = lmsOriginal_vector[indices,0:4]
        lmsSimulatedY575_vector = numpy.dot(lmsOriginalY575_vector, lms2lms_y575.transpose()) 
        
        lmsOriginalB475_vector = lmsOriginal_vector[not_indices,0:4]
        lmsSimulatedB475_vector = numpy.dot(lmsOriginalB475_vector, lms2lms_b475.transpose()) 
        
        lmsSimulated_vector = lmsOriginal_vector.copy()
        lmsSimulated_vector[indices] = lmsSimulatedY575_vector
        lmsSimulated_vector[not_indices] = lmsSimulatedB475_vector
        
    elif coldef_type == "t":
        constantHues_LMS = {}
        constantHues_LMS['r660'] = r660_LMS
        constantHues_LMS['c485'] = c485_LMS
        
        lms2lms_deficient = makeLMSDeficientMatrix_brettel(constantHues_LMS, equiwhite_LMS, coldef_type)
        lms2lms_r660 = lms2lms_deficient['r660']
        lms2lms_c485 = lms2lms_deficient['c485']
        
        neutral_ratio = equiwhite_LMS[1]/equiwhite_LMS[2]
        indices = lmsOriginal_vector[:,1]/lmsOriginal_vector[:,2] < neutral_ratio
        not_indices = indices == False
        
        lmsOriginalr660_vector = lmsOriginal_vector[indices,0:4]
        lmsSimulatedr660_vector = numpy.dot(lmsOriginalr660_vector, lms2lms_r660.transpose()) 
        
        lmsOriginalc485_vector = lmsOriginal_vector[not_indices,0:4]
        lmsSimulatedc485_vector = numpy.dot(lmsOriginalc485_vector, lms2lms_c485.transpose()) 
        
        lmsSimulated_vector = lmsOriginal_vector.copy()
        lmsSimulated_vector[indices] = lmsSimulatedr660_vector
        lmsSimulated_vector[not_indices] = lmsSimulatedc485_vector
        
        if False:
            #testing some crzay shit
            lmsr660 = numpy.dot(lmsOriginal_vector,lms2lms_r660.transpose())
            lmsr660_arr = colour.data.Data(lmsSpace, numpy.reshape(lmsr660, (m,n,3)))
            srgbr660_arr = lmsr660_arr.get(colour.space.srgb)*255.
            img_r660 = Image.fromarray(numpy.uint8(srgbr660_arr))
            pylab.subplot(223)
            pylab.title("R660 projection")
            pylab.axis("off")
            pylab.imshow(img_r660) 
             
            #testing some crzay shit
            lmsc485 = numpy.dot(lmsOriginal_vector,lms2lms_c485.transpose())
            lmsc485_arr = colour.data.Data(lmsSpace, numpy.reshape(lmsc485, (m,n,3)))
            srgbc485_arr = lmsc485_arr.get(colour.space.srgb)*255.
            img_c485 = Image.fromarray(numpy.uint8(srgbc485_arr))
            pylab.subplot(224)
            pylab.title("C485 projection")   
            pylab.axis("off")
            pylab.imshow(img_c485) 
    else:
        lmsSimulated_vector = lmsOriginal_vector
    lmsSimulated_arr = colour.data.Data(lmsSpace, numpy.reshape(lmsSimulated_vector, (m,n,3)))
    
    # We propose this gamut clipping instead for the one proposed by vienot
    sRGBSimulated_arr = lmsSimulated_arr.get(colour.space.srgb)
    
    if is_numpy_array: img_out = sRGBSimulated_arr
    else: img_array = numpy.uint8(sRGBSimulated_arr*255.); img_out = Image.fromarray(img_array)
    
    #showLMSSpace()
    
    #img_out = crossOut(img_in)
    
    return img_out

#wech
# class StoppableThread(threading.Thread):
#     """Thread class with a stop() method. The thread itself has to performance_test
#     regularly for the stopped() condition."""
# 
#     def __init__(self):
#         super(StoppableThread, self).__init__()
#         self._stop = threading.Event()
# 
#     def stop(self):
#         self._stop.set()
# 
#     def stopped(self):
#         return self._stop.isSet()

#wech
# def counting():
#     i = 0
#     while True:
#         time.sleep(.1)
#         i += 1
#         print i


#print module_path"""

def daltonization_huan(img_in,options):
    
    print "Caution: Please do not use this function for serious bussiness.\nIt is horrible implementation, mixing up Matlab snippets in the Python code."
    
    """
    if not dict.has_key('img_in'):
        print "Error: No input file chosen."
        return
    img_in = dict['img_in']
    """
    
    if not options.has_key('coldef_type'):
        print 'Caution: No color deficiency type chosen. Choosing default value: '+settings.default['coldef_type']
        dict['coldef_type'] = settings.default['coldef_type']
        
    coldef_type = options['coldef_type']
    mlab = Matlab(matlab='/Applications/MATLAB_R2015a.app/bin/matlab') # OBS: Make this more accessible
    mlab.start()
    #socket.setdefaulttimeout(50)
    #mlab.run
    try:
        par_dir = os.path.abspath(os.path.join(settings.module_path, os.pardir))
        file_matlab = os.path.join(par_dir,'code','Matlab','implementation','callImgRecolorFromPython.m')
        path_tmp = os.path.join(par_dir,'colordeficiency-images','tmp','matlab_tmp.png')
        img_in.save(path_tmp,'png')
        print 2
        dict_matlab = {'path_tmp': path_tmp, 'coldef_type': coldef_type, 'from_python': 1}
        res = mlab.run_func(file_matlab, dict_matlab,60)#(file,60)
        print 3
        #img_out = Image.open(path_tmp)
        #os.remove(path_tmp)
        #img_out.show()
        #os.remove(path_tmp)
    except socket.timeout:
        print "Error: There was a timeout. Waiting."
        
        time.sleep(30)
        print "\t ... and moving on"
    
    try:
        img_out = Image.open(path_tmp)
        os.remove(path_tmp)
    except Exception,e:
        img_out = crossOut(img_in)
    mlab.stop()
    return img_out 
        
def runcvdKuhn2008ExeThread(dict):
    
    par_dir = os.path.abspath(os.path.join(settings.module_path, os.pardir))
    commando = "wine "+os.path.join(par_dir,'c++','cvdKuhn','cvdKuhn2008.exe')+" "+dict['filepath_orig_tmp']+" "+str(coldefType2Int(dict['coldef_type'])-1)+ " "+str(dict['exagerated'])+" "+str(dict['max_num_quant'])
    # -1 because Kuhn starts to count color deficiency types at 0.
    os.popen(commando)

def c2g_alsam(img_in):
    """
    Converts RGB image to gray image using the fastColour2Grey method by Ali Alsam and Drew ???
    """
    
    mlab = Matlab(matlab='/Applications/MATLAB_R2013b.app/bin/matlab') # OBS: Make this more accessible
    mlab.start()
    try:
        file_matlab = os.path.join(path_matlab_alsam,"c2g_alsam.m")
        path_in_tmp = os.path.join(settings.module_path,'colordeficiency-images','tmp','matlab_tmp_in.png')
        path_out_tmp = os.path.join(settings.module_path,'colordeficiency-images','tmp','matlab_tmp_out.png')
        img_in.save(path_in_tmp,'png')

        dict_matlab = {'path_in': path_in_tmp, 'path_out': path_out_tmp}# 'img_out_path': path_out_tmp}
        #dict_matlab = {'path':path_in_tmp}
        res = mlab.run_func(file_matlab, func_args = dict_matlab)#(file,60)

        #img_gray = Image.open(path_out_tmp)
        #img_gray_array = numpy.asarray(img_gray,dtype=float)/255.
        #img_out.show()
        #os.remove(path_tmp)
        img_out = Image.open(path_out_tmp)
    except Exception,e:
        print "Error: Something went wrong. " + str(e)
        img_out = crossOut(img_in)
    mlab.stop()
    
    
    return img_out

def daltonization_yoshi_alsam(img_in,options):
    
    if not options.has_key('coldef_type'):
        print 'Caution: No color deficiency type chosen. Choosing default value: '+settings.default['coldef_type']
        options['coldef_type'] = settings.default['coldef_type']
    
    # 1.step: convert original image to IPT
    img_in = img_in.convert('RGB')
    img_array = numpy.asarray(img_in, dtype=float)/255.
    
    sRGBOriginal_arr = colour.data.Data(colour.space.srgb,img_array)
    iptOriginal_arr = sRGBOriginal_arr.get(colour.space.ipt)
    
    # 2.step: convert original image into gray array using Alsam's method
    img_gray = c2g_alsam(img_in)
    #img_gray.show()
    img_gray = img_gray.convert('RGB')
    imgGray_arr = numpy.asarray(img_gray)/255.
    imgLuminance_arr = imgGray_arr[:,:,0]
    
    # 3.step: Replace I channel with gray image and convert back to RGB    
    iptDaltonized_arr = iptOriginal_arr.copy()
    iptDaltonized_arr[:,:,0] = imgLuminance_arr
    
    # 4.step: Convert back to sRGB image
    iptDaltonized_arr = colour.data.Data(colour.space.ipt,iptDaltonized_arr)
    sRGBDaltonized_arr = iptDaltonized_arr.get(colour.space.srgb)*255.
    
    img_array = numpy.uint8(sRGBDaltonized_arr)
    img_out = Image.fromarray(img_array)
    
    return img_out

def daltonization_yoshi_alsam_extra(img_in,options):
    
    if not options.has_key('coldef_type'):
        print 'Caution: No color deficiency type chosen. Choosing default value: '+settings.default['coldef_type']
        options['coldef_type'] = settings.default['coldef_type']
    coldef_type = options['coldef_type']    
    
    if options.has_key('alpha'):
        alpha = float(options['alpha'])
    else:
        sys.stdout.write(' Caution: No alpha has been chosen. Using default value of 1.0.')
        alpha = 1.0
        
    if options.has_key('beta'):
        beta = float(options['beta'])
    else:
        sys.stdout.write(' Caution: No beta has been chosen. Using default value of 0.0.')
        beta = 0.0
    
    # 1.step: convert original image to IPT
    img_in = img_in.convert('RGB')
    img_array = numpy.asarray(img_in, dtype=float)/255.
    
    sRGBOriginal_arr = colour.data.Data(colour.space.srgb,img_array)
    iptOriginal_arr = sRGBOriginal_arr.get(colour.space.ipt)
    
    # 2.step: convert original image into gray array using Alsam's method
    img_gray = c2g_alsam(img_in)
    #img_gray.show()
    img_gray = img_gray.convert('RGB')
    imgGray_arr = numpy.asarray(img_gray)/255.
    imgLuminance_arr = imgGray_arr[:,:,0]
    
    # 3.step: Replace I channel with gray image and convert back to RGB    
    iptDaltonized_arr = iptOriginal_arr.copy()
    #iptDaltonized_arr[:,:,0] = imgGray_arr+iptP_arr
    if (coldef_type == 'p') or (coldef_type == 'd'):
        iptP_arr = iptOriginal_arr[:,:,1]
        #Image.fromarray(iptP_arr*255.).show()
        iptDaltonized_arr[:,:,0] = alpha*imgLuminance_arr+beta*(iptP_arr+1)/2
    elif coldef_type == 't':
        iptT_arr = iptOriginal_arr[:,:,2]
        #Image.fromarray(iptT_arr*255.).show()
        iptDaltonized_arr[:,:,0] = alpha*imgLuminance_arr+beta*(iptT_arr+1)/2
    
    # 4.step: Convert back to sRGB image
    iptDaltonized_arr = colour.data.Data(colour.space.ipt,iptDaltonized_arr)
    sRGBDaltonized_arr = iptDaltonized_arr.get(colour.space.srgb)*255.
    
    img_array = numpy.uint8(sRGBDaltonized_arr)
    img_out = Image.fromarray(img_array)
    
    return img_out   
    
def daltonization_kuhn(img_in,options):
    
    print "Caution: Please do not use this function for serious bussiness. It is horrible implementation!!"
    
    if not options.has_key('coldef_type'):
        print 'Caution: No color deficiency type chosen. Choosing default value: '+settings.default['coldef_type']
        options['coldef_type'] = settings.default['coldef_type']
        
    if not options.has_key('max_num_quant'):
        print 'Caution: No max number for quantization chosen. Choosing default value: 128.'
        options['max_num_quant'] = 128  
        
    if not options.has_key('exagerated'):
        print 'Caution: No attribute chosen for whether the contrast should be exagerated or not. Choosing default value: 0.'
        options['exagerated'] = 0
    
    par_dir = os.path.abspath(os.path.join(settings.module_path, os.pardir))
    path_tmp = os.path.join(par_dir,'colordeficiency-images','tmp') # we have to think from the c++ folder
    filepath_orig_tmp = os.path.join(path_tmp,"kuhnorig_tmp.bmp")
    options['filepath_orig_tmp'] = filepath_orig_tmp
    img_in.save(filepath_orig_tmp)
    
    filepath_recolor_tmp = os.path.join(path_tmp,"kuhnorig_tmp_recolor_kuhn.bmp")
    filepath_sim_tmp = os.path.join(path_tmp,"kuhnorig_tmp_simulate_brettel.bmp")
    
    wine_thread = threading.Thread(target=runcvdKuhn2008ExeThread,args=(options,))
    wine_thread.start()
    
    recolor_img_does_not_exist = True
    while recolor_img_does_not_exist:
        e = ""
        #print "Nope."
        time.sleep(.2)
        if os.path.isfile(filepath_recolor_tmp):
            try:
                img_out = Image.open(filepath_recolor_tmp)
                #img_sim = Image.open(filepath_sim_tmp)
                try: 
                    img_out.show()
                except Exception,e:
                    print "EH eh EH eh!"
                    print "Error: Saving not done yet."
            except Exception,e:
                print "Error: Computation not done yet."
            
            if not e:    
                print "Success: Computation done. Juchee!!"
                recolor_img_does_not_exist = False
    #print "Resting litt"
    time.sleep(1)        
        
    #print "Kill kill kill! Mothafucka!!!!"
    os.popen("killall wine")
    os.popen("killall Preview")
    #print "Deleting tmp files!"
    #img_sim.show()
    os.remove(filepath_orig_tmp)
    os.remove(filepath_recolor_tmp)
    os.remove(filepath_sim_tmp)
    
    print 'and moving on ...'
    return img_out

def showLMSSpace():
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import product, combinations
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    
    #draw cube
    r = [-1, 1]
    ax.plot3D((1,-1), color="b")
    """
    for s, e in combinations(np.array(list(product(r,r,r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s,e), color="b")
    """     
            
    """
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")
    
    #draw a point
    ax.scatter([0],[0],[0],color="g",s=100)
    
    #draw a vector
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs
    
        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)
    
    a = Arrow3D([0,1],[0,1],[0,1], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(a)
    
    """
    plt.show()   

def simulate( simulation_type, img_in, coldef_type, coldef_strength=1.0):
    """
    Function to simulate color deficiency.
    Input:  simulation_type - Type of simulation as defined in simulation_types
            img_in -          Original PIL image
            coldef_type -     Type of color deficiency. Either p for protanopes, d for deuteranopes, or t for tritanopes.
            coldef_strength - Severity of color deficiency where 1.0 are completely and 0.0 is nothing. OBS: Some algorithms only allow full severity!
    Output: img_out -         Simulated PIL image
    """
    #img_out = crossOut(img_in)
    
    if simulation_type == "vienot": img_out = simulation_vienot(img_in, coldef_type,coldef_strength)
    elif simulation_type == "vienot-adjusted": img_out = simulation_vienot_adjusted(img_in, coldef_type,coldef_strength)
    elif simulation_type == "kotera": img_out = simulation_kotera(img_in, coldef_type,coldef_strength)
    elif simulation_type == "brettel": img_out = simulation_brettel(img_in, coldef_type, coldef_strength)
    elif simulation_type == "dummy": img_out = convertToLuminanceImage(img_in,{})
    else:
        print 'Error: Simulation type does not exist. Choose either one of the following - "'+'" , "'.join(settings.simulation_types)+'".'
        img_out = crossOut(img_in)
    return img_out

def daltonization_anagnostopoulos(img_in,options):
    """
    """
    
    coldef_type = options['coldef_type']
    
    if not (coldef_type == "p" or coldef_type == "d"):
        print "Error: Unknown color deficiency chosen. Chose either p for protanopes, or d for deuteranopes."
        
        img_out = crossOut(img_in)
        return img_out
    
    sRGBOriginal_arr = numpy.asarray(img_in.convert('RGB'), dtype=float)
    sRGBSimulated_arr = numpy.asarray(simulation_vienot_adjusted(img_in, coldef_type))
    m,n,dim = numpy.shape(sRGBOriginal_arr)
    
    # This is the actual simulation
    #Computing error image
    err2mod = numpy.array([[0,0,0],
                           [0.7,1,0],
                           [0.7,0,1]])
    sRGBOriginal_vector = numpy.reshape(sRGBOriginal_arr,(m*n,3))
    sRGBSimulated_vector = numpy.reshape(sRGBSimulated_arr, (m*n,3))
    sRGBError_vector = sRGBOriginal_vector - sRGBSimulated_vector
    
    #Distributing error
    sRGBErrorAdjusted_vector = numpy.dot(sRGBError_vector,err2mod.transpose())
    #print numpy.shape(sRGBError_vector)
    #sRGBErrorAdjusted_vector = numpy.dot(err2mod,sRGBError_vector.transpose())
    sRGBDaltonized_vector = sRGBOriginal_vector + sRGBErrorAdjusted_vector
    
    sRGBDaltonized_array = numpy.reshape(sRGBDaltonized_vector, (m,n,3))
    sRGBDaltonized_array[sRGBDaltonized_array<0.] = 0.
    sRGBDaltonized_array[sRGBDaltonized_array>255.] = 255.
        
    img_array = numpy.uint8(sRGBDaltonized_array)
    img_out = Image.fromarray(img_array)
    
    return img_out

def lambdaShiftKotera(fund_img, lamda):
    """
    Shifts the fundamental image by a certain wavelength
    """
    
    k,mn = numpy.shape(fund_img)
    lamda = lamda % k
    
    a_first = fund_img[0:lamda+1,:]
    a_last = fund_img[lamda+1:k,:]
    
    fundShift_img = numpy.vstack([a_last, a_first])
    
    return fundShift_img

def visabilityCostKotera(shiftImage_vector, rdic):
    
    cost = 0
    
    #cost = numpy.linalg.norm(numpy.dot(rdic,shiftImage_vector))
    a = numpy.dot(rdic,shiftImage_vector)
    a = a.transpose()
    
    #print numpy.shape(a)
    #s = numpy.sum(numpy.dot(a,a.transpose()))
    #print numpy.shape(s)
    #cost = math.sqrt(s)
    #print cost
    
    #print numpy.linalg.norm(a), numpy.linalg.norm(a.transpose())
    
    cost = numpy.linalg.norm(a)
    
    return cost**2

def visualGapCostKotera(shiftImage_vector,deltaCDic_vector,rlms,rdic):
    
    cost = 0.
    
    #cost = numpy.linalg.norm(deltaCDic_vector+numpy.dot((rlms-rdic),shiftImage_vector))
    a = deltaCDic_vector+numpy.dot((rlms-rdic),shiftImage_vector)
    a = a.transpose()
    
    #print numpy.shape(a)
    #b= numpy.dot(a,a.transpose())
    #print numpy.shape(b)
    #cost = math.sqrt(numpy.sum(b))
    #print cost
    
    cost = numpy.linalg.norm(a)
    
    return cost**2
    
def costKotera(shiftImage_vector,deltaCDic_vector,rlms,rdic):
    
    cost = 0.
    
    #cost = visualGapCostKotera(shiftImage_vector,deltaCDic_vector,rlms,rdic)
    cost = 0.5*(visabilityCostKotera(shiftImage_vector,rdic)+1.0-visualGapCostKotera(shiftImage_vector,deltaCDic_vector,rlms,rdic))
   
    return cost
    
def daltonization_kotera(img_in, options):
    """
    """
    if not options.has_key('coldef_type'):
        print 'Caution: No color deficiency type chosen. Choosing default value: '+settings.default['coldef_type']
        dict['coldef_type'] = settings.default['coldef_type']
    coldef_type = options['coldef_type']
    if not (coldef_type == "p" or coldef_type == "d" or coldef_type == "t"):
        print "Error: Unknown color deficiency chosen. Chose either p for protanopes, d for deuteranopes, or t for tritanopes."
        return img_in
    
    img_in = img_in.convert('RGB')
    img_array = numpy.asarray(img_in, dtype=float)/255.
    m,n,d = numpy.shape(img_array)
    
    sRGB_arr = colour.data.Data(colour.space.srgb, img_array)
    xyz_arr = sRGB_arr.get(colour.space.xyz)
    xyz_vector = numpy.reshape(xyz_arr,(m*n,3))
    
    #Read xyz color matching functions
    data = numpy.genfromtxt(os.path.join(os.path.abspath(os.path.join(settings.module_path, os.pardir)),'colordeficiency-data','ciexyz31.csv'), delimiter=',')
    xyzMatchFuncs = data[:,1:4]
    xyz2lms = numpy.array([[.15514, .54312, -.03286],
                           [-.15514, .45684, .03286],
                           [0., 0., .00801]])
    lms2xyz = numpy.linalg.inv(xyz2lms)
    lmsMatchFuncs = numpy.dot(xyzMatchFuncs,xyz2lms.transpose()) # obtain the lms matching functions from the xyz matching functions
    
    lms_vector = numpy.dot(xyz_vector,xyz2lms.transpose())
    k,dd =  numpy.shape(lmsMatchFuncs)
    
    a = lmsMatchFuncs    
    pinv = numpy.dot(a,numpy.linalg.inv(numpy.dot(a.transpose(),a))) # pseudo inverse to translate from lms color space into pseudospectral color space
    cStarLMS_vector = numpy.dot(pinv,lms_vector.transpose())
    
    rlms = numpy.dot(a,numpy.dot(numpy.linalg.inv(numpy.dot(a.transpose(),a)),a.transpose()))
    #print numpy.shape(rlms) 
    
    if coldef_type == "p":
        #Matrix for protanopes
        rdic = numpy.array([lmsMatchFuncs[:,1],lmsMatchFuncs[:,2]])
    elif coldef_type == "d":
        #Matrix for deuteranopes
        rdic = numpy.array([lmsMatchFuncs[:,0],lmsMatchFuncs[:,2]])
    elif coldef_type == "t":
        #Matrix for tritanopes
        rdic = numpy.array([lmsMatchFuncs[:,0],lmsMatchFuncs[:,1]])
    adic = rdic.transpose()
    rdic = numpy.dot(adic,numpy.dot(numpy.linalg.inv(numpy.dot(adic.transpose(),adic)),adic.transpose()))
    
    deltaCDic_vector = numpy.dot((rlms-rdic),cStarLMS_vector)
    #print numpy.shape(deltaCDic_vector)
    #optimization
    
    lambda_opt = 0
    cost_opt = costKotera(lambdaShiftKotera(deltaCDic_vector,lambda_opt),deltaCDic_vector,rlms,rdic)
    #print "hiersimmer",
    int = numpy.linspace(0,k,95)
    costs = []
    cost_visibility = []
    for i in int:
        shiftImage_vector_tmp = lambdaShiftKotera(deltaCDic_vector,i)
        cost_visibility_tmp = visabilityCostKotera(shiftImage_vector_tmp,rdic)
        cost_visualGap_tmp = visualGapCostKotera(shiftImage_vector_tmp,deltaCDic_vector,rlms,rdic)
        #cost = costKotera(shifImage_vector_tmp,deltaCDic_vector,rlms,rdic)
        costs.append((i,cost_visibility_tmp,cost_visualGap_tmp))
        #         if cost >= cost_opt:
        #             cost_opt = cost
        #             lambda_opt = i
    
    costs =  numpy.array(costs)
    #print numpy.max(costs[:,1])
    costs[:,1] = costs[:,1]/numpy.max(costs[:,1])
    costs[:,2] = costs[:,2]/numpy.max(costs[:,2])
    kotera_costs = numpy.array(0.5*costs[:,2]+1-0.5*costs[:,1])
    costs = numpy.array([costs[:,0],costs[:,1],costs[:,2],kotera_costs]).transpose()
    #print costs
    #print numpy.shape(costs)
    #print numpy.shape(kotera_costs)
    #costs[:,:+1]=kotera_costs
    #print kotera_costs.transpose()
    #print numpy.shape(kotera_costs)
    #costs = numpy.append(costs,kotera_costs.transpose(),1)
    #print numpy.shape(costs)
    
    
    #pylab.figure()
    #pylab.plot(costs[:,0]*5,costs[:,1])#/numpy.max(costs[:,1])),
    #pylab.plot(costs[:,0]*5,costs[:,2])#/numpy.max(costs[:,2]))
    #pylab.plot(costs[:,0]*5,costs[:,3])#/numpy.max(costs[:,2]))
    #pylab.show()
    
    lambda_opt = numpy.argmin(costs[:,3])
    #cost_opt = costs[lambda_opt,3]
    
    deltaCStarSht_vector = lambdaShiftKotera(deltaCDic_vector,lambda_opt)
    #print lambda_opt*5, cost_opt, costs[lambda_opt+3,3]
    
    
    cDaltLMS_vector = cStarLMS_vector + deltaCStarSht_vector
    
    lmsOut_vector = numpy.dot(lmsMatchFuncs.transpose(),cDaltLMS_vector)
    xyzOut_vector = numpy.dot(lms2xyz,lmsOut_vector)
    
    xyzOut_arr = colour.data.Data(colour.space.xyz, numpy.reshape(xyzOut_vector.transpose(), (m,n,3)))
    sRGBOut_arr = xyzOut_arr.get(colour.space.srgb)*255.
    img_array = numpy.uint8(sRGBOut_arr)
    
    img_out = Image.fromarray(img_array)
    
    return img_out

def daltonization_yoshi_c2g_only(img_in, options):
    
    # 1.step: convert original image to IPT
    img_in = img_in.convert('RGB')
    img_array = numpy.asarray(img_in, dtype=float)/255.
    
    sRGBOriginal_arr = colour.data.Data(colour.space.srgb,img_array)
    iptOriginal_arr = sRGBOriginal_arr.get(colour.space.ipt)
    
    # 2.step: Enhance luminance channel
    # Make gray image of original RGB image
    if options.has_key('its'):
        its = options['its']
    else:
        its = settings.its_default # number of iterations (ni) should be >100
    
    if options.has_key('pts'):
        pts = options['pts']
    else:
        pts =  settings.pts_default # number of sample points (ns) should be between 2 and 10, where 2 means local adjustments and 10 means mainly global adjustments
        # The noise is anti-proportional to the product ni*ns, which should be > 1000.
        # The complexityis proportional to the product ni*ns. 
        
    name_in_tmp = os.path.join(settings.module_path,'colordeficiency-images','tmp','img_in_tmp.png')
    name_out_tmp = os.path.join(settings.module_path,'colordeficiency-images','tmp','img_out_tmp.png')
    img_in.save(name_in_tmp)
        
    if not ":/usr/local/bin" in os.environ['PATH']:
        os.environ['PATH'] = os.environ['PATH'] + ":/usr/local/bin"
        print os.environ['PATH']
    os.system(os.path.join(os.path.abspath(os.path.join(settings.module_path, os.pardir)),'colordeficiency-data','code','simone','stress')+" -i "+name_in_tmp+" -o "+name_out_tmp+" -g -ns "+str(pts)+" -ni "+str(its)) # Run the c2g C++ script in a shell
        
    img_gray = Image.open(name_out_tmp)
    os.remove(name_in_tmp)
    os.remove(name_out_tmp)
    
    img_gray = img_gray.convert('RGB')
    imgGray_arr = numpy.asarray(img_gray)/255.
    imgLuminance_arr = imgGray_arr[:,:,0]
    
    # 3.step: Replace I channel with gray image and convert back to RGB    
    iptDaltonized_arr = iptOriginal_arr.copy()
    iptDaltonized_arr[:,:,0] = imgLuminance_arr
    
    # 4.step: Convert back to sRGB image
    iptDaltonized_arr = colour.data.Data(colour.space.ipt,iptDaltonized_arr)
    sRGBDaltonized_arr = iptDaltonized_arr.get(colour.space.srgb)*255.
    
    img_array = numpy.uint8(sRGBDaltonized_arr)
    img_out = Image.fromarray(img_array)
 
    return img_out

def daltonization_yoshi_c2g(img_in,options):
        
    if options.has_key('coldef_type'): coldef_type = options['coldef_type']
    else:
        sys.stdout.write(' Error: No coldef_type. You have to choose a valid color deficiency type.')
        return img_in
    
    if options.has_key('alpha'): alpha = options['alpha']
    else:
        sys.stdout.write('Caution: No alpha has been chosen. Using default value of 1.0.')
        alpha = 1.0
        
    if options.has_key('beta'): beta = options['beta']
    else:
        sys.stdout.write(' Caution: No beta has been chosen. Using default value of 0.0.')
        beta = 0.0
       
    # 1.step: convert original image to IPT
    img_in = img_in.convert('RGB')
    img_array = numpy.asarray(img_in, dtype=float)/255.
    
    sRGBOriginal_arr = colour.data.Data(colour.space.srgb,img_array)
    iptOriginal_arr = sRGBOriginal_arr.get(colour.space.ipt)
    
    # 2.step: Enhance luminance channel if wanted.
    
    # Check if user wants to use enhanced gray channel. False by default
    if options.has_key('enhance'): enhance = options['enhance']
    else:
        sys.stdout.write(" Caution: No value for enhancement chosen. Chose default: True")
        enhance = True
        
    if enhance:
        # Make gray image of original RGB image
        if options.has_key('its'): its = options['its']
        else: its = settings.its_default # number of iterations (ni) should be >100
        
        if options.has_key('pts'): pts = options['pts']
        else:
            pts =  settings.pts_default # number of sample points (ns) should be between 2 and 10, where 2 means local adjustments and 10 means mainly global adjustments
        # The noise is anti-proportional to the product ni*ns, which should be > 1000.
        # The complexityis proportional to the product ni*ns. 
        
        
        name_in_tmp = os.path.join(settings.module_path,'colordeficiency-images','tmp','img_in_tmp.png')
        name_out_tmp = os.path.join(settings.module_path,'colordeficiency-images','tmp','img_out_tmp.png')
        
        img_in.save(name_in_tmp)
        
        if not ":/usr/local/bin" in os.environ['PATH']:
            os.environ['PATH'] = os.environ['PATH'] + ":/usr/local/bin"
            print os.environ['PATH']
        os.system(os.path.join(os.path.abspath(os.path.join(settings.module_path, os.pardir)),'colordeficiency-data','code','simone','stress')+" -i "+name_in_tmp+" -o "+name_out_tmp+" -g -ns "+str(pts)+" -ni "+str(its)) # Run the c2g C++ script in a shell
        
        img_gray = Image.open(name_out_tmp)
        os.remove(name_in_tmp)
        os.remove(name_out_tmp)
        
        img_gray = img_gray.convert('RGB')
        imgGray_arr = numpy.asarray(img_gray)/255.
        imgLuminance_arr = imgGray_arr[:,:,0]
    else:
        imgLuminance_arr = iptOriginal_arr[:,:,0]
    
    # 3.step: Replace I channel with gray image and convert back to RGB
    
    iptDaltonized_arr = iptOriginal_arr.copy()
    #iptDaltonized_arr[:,:,0] = imgGray_arr+iptP_arr
    if (coldef_type == 'p') or (coldef_type == 'd'):
        iptP_arr = iptOriginal_arr[:,:,1]
        #Image.fromarray(iptP_arr*255.).show()
        iptDaltonized_arr[:,:,0] = alpha*imgLuminance_arr+beta*(iptP_arr+1)/2
    elif coldef_type == 't':
        iptT_arr = iptOriginal_arr[:,:,2]
        #Image.fromarray(iptT_arr*255.).show()
        iptDaltonized_arr[:,:,0] = alpha*imgLuminance_arr+beta*(iptT_arr+1)/2
        
    # 4.step: Convert back to sRGB image
    iptDaltonized_arr = colour.data.Data(colour.space.ipt,iptDaltonized_arr)
    sRGBDaltonized_arr = iptDaltonized_arr.get(colour.space.srgb)*255.
    
    img_array = numpy.uint8(sRGBDaltonized_arr)
    img_out = Image.fromarray(img_array)
 
    return img_out   

def daltonize(img_in,options):
    """
    Function to daltonize image for color deficient people.
    Input:  daltonization_type -  Type of daltonization as defined in daltonization_types
            img_in -              Original PIL image
            coldef_type -         Type of color deficiency. Either p for protanopes, d for deuteranopes, or t for tritanopes.
    Output: img_out -             Simulated PIL image
    """
    
    # Check this!!!
    #img_in = img_in.convert('RGB')
    
    if not options.has_key('daltonization_type'):
        print "Caution: No daltonization_type chosen. Default value will be chosen: " + settings.default['daltonization_type']
        options['daltonization_type'] = settings.default['daltonization_type']
    daltonization_type = options['daltonization_type']
    
    if not options.has_key('coldef_type'):
        print 'Caution: No coldef_type chosen. Default value will be chosen: '+settings.default['coldef_type']
        options['coldef_type'] = settings.default['coldef_type']    
    #dict = {'img_in': img_in, 'coldef_type': coldef_type}
    
    if daltonization_type == "anagnostopoulos": img_out = daltonization_anagnostopoulos(img_in,options)
    elif daltonization_type == "fidaner": img_out = daltonization_anagnostopoulos(img_in,options)
    elif daltonization_type == "kotera": img_out = daltonization_kotera(img_in,options)
    elif daltonization_type == "kuhn": img_out = daltonization_kuhn(img_in,options)
    elif daltonization_type == "huan": img_out = daltonization_huan(img_in,options)
    elif daltonization_type == "yoshi-simone-only": img_out = daltonization_yoshi_c2g_only(img_in,options)
    elif daltonization_type == "yoshi-alsam-only": img_out = daltonization_yoshi_alsam(img_in,options)
    elif daltonization_type == "yoshi-simone": img_out = daltonization_yoshi_c2g(img_in,options)
    elif daltonization_type == "yoshi-alsam": img_out = daltonization_yoshi_alsam_extra(img_in,options)
    elif daltonization_type == "yoshi-ii": img_out = daltonization_yoshi_042016(img_in, options)
    elif daltonization_type == "dummy": img_out = convertToLuminanceImage(img_in,options)
    else:
        print 'Error: Daltonization type does not exist. Choose either one of the following - "'+'" , "'.join(settings.daltonization_types)+'".'
        return img_in
    return img_out


def simulate_pics(images,options):
    """
    Same as in tools_out.py, is not actually called up.
    """
    if options.has_key('path_out'):
        path_out = options['path_out']
    else:
        path_out = ""
     
    if not options.has_key('simulation_types'):
        print "Caution: No daltonization_type chosen. Default value will be chosen: " + ", ".join(settings.simulation_types)
        options['simulation_types'] = settings.simulation_types
     
    if not options.has_key('coldef_types'):
        print 'Caution: No coldef_type chosen. Default value will be chosen: '+", ".join(settings.coldef_types)
        options['coldef_types'] = settings.coldef_types
     
    for img in images:
        path_in_tmp, basename_tmp = os.path.split(img)
        file_name_in_tmp,file_ext = os.path.splitext(basename_tmp)
        dict_in = getStatsFromFilename(file_name_in_tmp)
         
        if not path_out:
            path_out = path_in_tmp
        sys.stdout.write("Computing "+str(file_name_in_tmp))
         
        for simulation_type in options['simulation_types']:
            sys.stdout.write( "..." + simulation_type)
            for coldef_type in options['coldef_types']:
                sys.stdout.write("."+coldef_type)
                 
                if (bool(dict_in['sim'])):
                    sys.stdout.write('.Cannot simulate already simulated images')
                 
                elif (bool(dict_in['dalt']) and not (int(dict_in['coldef_type'])) == settings.colDef2ID[coldef_type]):
                    sys.stdout.write('.Color deficiency type of daltonization and simulation has to be the same')
                else:
                     
                    img_in = Image.open(img)
                    img_sim =simulate(simulation_type, img_in, coldef_type)
                  
                    file_name_out_tmp = setStatsToFilename(
                                                       dict_in['img_id'],
                                                       dict_in['dalt'],
                                                       dict_in['dalt_id'],
                                                       True,
                                                       settings.sim2ID[simulation_type],
                                                       settings.colDef2ID[coldef_type]
                                                       )
                    file_path_out_tmp = os.path.join(path_out,file_name_out_tmp)
                    img_sim.save(file_path_out_tmp)
 
        sys.stdout.write("\n")

def daltonize_pics(images,options):
    """
    Braumer des werklech?
    """
     
    path_out = options['path_out'] if options.has_key('path_out') else ""
    if not options.has_key('daltonization_types'):
        print "Caution: No daltonization_type chosen. Default value will be chosen: " + ", ".join(settings.daltonization_types)
        options['daltonization_types'] = settings.daltonization_types
     
    if not options.has_key('coldef_types'):
        print 'Caution: No coldef_type chosen. Default value will be chosen: '+", ".join(settings.coldef_types)
        options['coldef_types'] = settings.coldef_types
        
    round = options['vs_round']
     
    for img in images:
        path_in_tmp, basename_tmp = os.path.split(img)
        file_name_in_tmp,file_ext = os.path.splitext(basename_tmp)
        dict_in = getStatsFromFilename(file_name_in_tmp)
        img_in = Image.open(img)
        
        img_id = dict_in['img_id']
        variant_id = getVariantFromImage(img_id,round)
        print variant_id
        if not variant_id == 0:
             
            if not path_out: path_out = path_in_tmp
            sys.stdout.write("Computing "+str(file_name_in_tmp))
            
            file_path_out_tmp = os.path.join(path_out,file_name_in_tmp+file_ext)
            img_in.save(file_path_out_tmp)
            
            for daltonization_type in options['daltonization_types']:
                sys.stdout.write( "..." + daltonization_type)
                for coldef_type in options['coldef_types']:
                    sys.stdout.write("."+coldef_type)
                     
                    if (bool(dict_in['sim'])): sys.stdout.write('.Cannot daltonize already simulated images')
                    elif bool(dict_in['dalt']): sys.stdout.write('.Cannot daltonize already daltonized images')
                    else:
                        options_tmp = options.copy()
                        options_tmp['coldef_type'] = coldef_type
                        options_tmp['daltonization_type'] = daltonization_type
                      
                        file_name_out_tmp = setStatsToFilename( dict_in['img_id'],
                                                                True,
                                                                settings.dalt2ID[daltonization_type],
                                                                dict_in['sim'],
                                                                dict_in['sim_id'],
                                                                settings.colDef2ID[coldef_type])
                        file_path_out_tmp = os.path.join(path_out,file_name_out_tmp)
                        if not os.path.exists(file_path_out_tmp):
                            img_dalt =daltonize(img_in, options_tmp)
                            img_dalt.save(file_path_out_tmp)
                        else:
                            sys.stdout.write(": file already exists ")
            sys.stdout.write("\n")
        else:
            if not path_out: path_out = path_in_tmp
            sys.stdout.write("Computing "+str(file_name_in_tmp))
            
            file_path_out_tmp = os.path.join(path_out,file_name_in_tmp+file_ext)
            img_in.save(file_path_out_tmp)
            sys.stdout.write("\n")

def makeSimulationLookupTable(simulation_type, coldef_type,accuracy=5,colorspace="sRGB"):
    
    itv = numpy.linspace(0,255,accuracy)
    
    input_tab = []
    for r in itv:
        for g in itv:
            for b in itv:
                input_tab.append([r,g,b])
    input_tab = numpy.uint8(numpy.asarray(input_tab))
    mn,d = numpy.shape(input_tab)
    
    input_arr = numpy.uint8(numpy.reshape(input_tab,(mn,1,3)))
    input_img = Image.fromarray(input_arr)
    output_img = simulate(simulation_type, input_img, coldef_type)
    
    output_array = numpy.asarray(output_img)
    output_tab = numpy.reshape(output_array,(mn,3))    
    
    if colorspace == "IPT":
        input_tab_srgb = colour.data.Data(colour.space.srgb,input_tab)
        input_tab = input_tab_srgb.get(colour.space.ipt)
        output_tab_srgb = colour.data.Data(colour.space.srgb,output_tab)
        output_tab = output_tab_srgb.get(colour.space.ipt)
                
    return input_tab, output_tab

def lookup(img_in, input_tab, output_tab):
    
    input_arr = numpy.asarray(img_in)
    res = numpy.shape(img_in)
    
    m = res[0]
    n = res[1]
    if numpy.shape(res)[0]<3:
        d = 1
        input_vec = numpy.reshape(input_arr,m*m)
        
    else:
        d = res[2]
        input_vec = numpy.reshape(input_arr,(m*n,d))
    
    output_vec = griddata(input_tab, output_tab, input_vec, 'linear')
    if numpy.shape(res)[0]<3: output_arr = output_vec.reshape(m,n)
    else: output_arr = output_vec.reshape(m,n,d)

    return output_arr

###
# Total variation Daltonization starts here
###

def s(im):
    """
    Compression function (could be lut)
    """
    retim = im.copy()
    retim[..., 0] = .5 * (im[..., 0] + im[..., 1])
    retim[..., 1] = .5 * (im[..., 0] + im[..., 1])
    return retim
  
def daltonization_yoshi_042016(im,dict):

    modus = dict['modus'] if dict.has_key('modus') else 0
    dict.update({'ms_first': 1,
                 'im0': im.copy()})
    
    is_numpy_array = type(im)==numpy.ndarray
    if is_numpy_array: im_array = im[...,0:3]
    else: im = im.convert('RGB'); im_array = numpy.asarray(im, dtype=float)/255.
    
    if modus==0:
        sys.stdout.write("Simple. ")
        im_dalt = daltonization_yoshi_gradient(im_array,dict)
    elif modus==1:
        sys.stdout.write("Multi scaling. ")
        im_dalt = multiscaling(im_array,daltonization_yoshi_gradient,dict)
    elif modus==2:
        sys.stdout.write("Edge. ")
        im_dalt = daltonization_yoshi_gradient(im_array,dict)
    #elif modus==2:
    #    sys.stdout.write("Smoothing. ")
    #    no_max_sigma = False
    #    if dict.has_key('max_sigma'): max_sigma = dict['max_sigma']
    #    else: max_sigma = 1; dict['max_sigma'] = max_sigma; no_max_sigma = True
    #    im_dalt = smoothing(im_array,max_sigma,daltonization_yoshi_gradient,dict)
    #    if no_max_sigma: del dict['max_sigma']
    
    if is_numpy_array: im_out = im_dalt
    else: im_array = numpy.uint8(im_dalt*255.); im_out = Image.fromarray(im_array)
        
    return im_out
   

from matplotlib.mlab import PCA

def daltonization_yoshi_iii(im,dict):
    
    name = "daltonization_yoshi-iii"
    # Required variables
    if not dict.has_key('yg_simulation_type'):
        print "Error: The simulation type has to be defined in the dictionary. ["+name+"]"
        return crossOut(im)
    else: simulation_type = dict['yg_simulation_type']
        
    if not dict.has_key('coldef_type'):
        print "Error: The color deficiency type has to be defined in the dictionary. ["+name+"]"
        return crossOut(im)
    else: coldef_type = dict['coldef_type']; #sys.stdout.write(coldef_type)
    
    # Optional variables for simulate
    coldef_strength = dict['coldef_strength'] if dict.has_key('coldef_strength') else 1.0
    sigma = dict['sigma'] if dict.has_key('sigma') else 3
    
    m,n,d = numpy.shape(im)
    im_arr = im.reshape((m*n,3))
    
    im_sim = simulate(simulation_type,im,coldef_type,coldef_strength); im_sim_arr = im_sim.reshape((m*n,3))
    d0 = im - im_sim; d0_arr = im.reshape((m*n,3));
    
    constant_lightness = 0; ms_first = 0; global_unit_vectors = 1
    ### Direction of lightness
    if constant_lightness:
        if ms_first: sys.stdout.write("El: Constant lightness. ")
        el = numpy.array([0.2126,0.7152,0.0722]) 
    else: 
        if ms_first: sys.stdout.write("El: Neutral gray lightness. ")
        el = numpy.array([1.,1.,1.])
    el = el / numpy.sqrt(numpy.sum(el**2))
    if not global_unit_vectors: el = numpy.array([el,]*(m*n))
    
    ### Direction of confusion colors
    if global_unit_vectors:  
        # Use difference in image domaine 
        if ms_first: sys.stdout.write("Global ed: Image PCA, ")
        ed_PCA = PCA(d0_arr, standardize=False)
        ed = ed_PCA.Wt[0]; ed = ed / numpy.sqrt(numpy.sum(ed**2))
        if ms_first: sys.stdout.write("not orthogonalized. ")
        
    else:
        if ms_first: sys.stdout.write("Spatial ed, ")
        d0_norm_1D = numpy.sqrt(numpy.sum(d0_arr**2, axis=1))
        d0_norm_3D = numpy.array([d0_norm_1D,]*d).transpose()
        ed = d0_arr / d0_norm_3D
        if ms_first: sys.stdout.write("not orthogonalized. ")
    
    ### Direction of optimal daltonization
    if global_unit_vectors: ec = numpy.cross(ed,el); ec = ec / numpy.sqrt(numpy.sum(ec**2))
    else: 
        ec = numpy.cross(ed,el)
        ec_norm_1D = numpy.sqrt(numpy.sum(ec**2, axis=1))
        ec_norm_3D = numpy.array([ec_norm_1D,]*d).transpose()
        ec = ec/ec_norm_3D
    if ms_first: sys.stdout.write('\n')
    
    if global_unit_vectors: k = numpy.dot(im_arr,ed)# k_y = numpy.dot(grady0_arr,ed)
    else: k_x = numpy.sum(gradx0_arr*ed, axis=1); k_y = numpy.sum(grady0_arr*ed, axis=1) # feil
    imdot_arr = numpy.array([k,]*d).transpose()#; grady0dot_arr = numpy.array([k_y,]*d).transpose()   
    
    chi_arr = 1.
    boost_ec = dict['boost_ec'] if dict.has_key('boost_ec') else 1.
    combination = dict['combination'] if dict.has_key('combination') else 1
    if combination==1:
        if ms_first: sys.stdout.write("Chroma only.\n")
        print numpy.shape(imdot_arr), numpy.shape(im_arr)
        im_rot_arr = imdot_arr*boost_ec*chi_arr*el; im_rot = im_rot_arr.reshape((m,n,3)) 
        im_dalt_arr = im_arr-imdot_arr+im_rot_arr; im_dalt = im_dalt_arr.reshape((m,n,3)) 
        #gradydalt_arr = grady0_arr+(grady0dot_arr*boost_ec*chi_arr*ec); gradydalt = gradydalt_arr.reshape((m,n,3))  
    #im_dalt = im.copy()+im_edg
    # PIL edges
    
    #im_ = 
    
    im_dalt[im_dalt>1.0] = .5
    im_dalt[im_dalt<0.0] = 0.5
    
    #img_out = im_err.copy()
    img_out = im_dalt
    print numpy.min(im_rot), numpy.max(im_rot), numpy.mean(im_rot)
    
    return img_out
    

def daltonization_yoshi_gradient(im,dict):
    
    name = "daltonization_yoshi_gradient"
    # Required variables
    if not dict.has_key('im0'):
        print "Error: The original image has to be defined in the dictionary. ["+name+"]"
        return
    else: im0 = dict['im0']
    
    if not dict.has_key('yg_simulation_type'):
        print "Error: The simulation type has to be defined in the dictionary. ["+name+"]"
        return crossOut(im)
    else: simulation_type = dict['yg_simulation_type']
        
    if not dict.has_key('coldef_type'):
        print "Error: The color deficiency type has to be defined in the dictionary. ["+name+"]"
        return crossOut(im)
    else: coldef_type = dict['coldef_type']; sys.stdout.write(coldef_type)

    # Optional variables for simulate
    coldef_strength = dict['coldef_strength'] if dict.has_key('coldef_strength') else 1.0
    # Optional variables for imresize
    interp = dict['interp'] if dict.has_key('interp') else 'bilinear'
    mode = dict['mode'] if dict.has_key('mode') else 'RGB'
    ms_first=dict['ms_first'] if dict.has_key('ms_first') else 0
    modus = dict['modus'] if dict.has_key('modus') else 0
    global_unit_vectors = dict['global_unit_vectors'] if dict.has_key('global_unit_vectors') else 1
    m,n,d = numpy.shape(im)
        
    #######
    ## Design the improved gradient
    #######
    
    if modus==1: im0_updated = imresize(im0,(m,n),interp,mode=mode)/255.; #im0_small_arr = im0_small.reshape(m*n,3)
    #elif modus==2: sigma = dict['max_sigma']; im0_updated = gaussian_filter(im0,(sigma,sigma,0))
    else: im0_updated=im0.copy()
    im0_updated_sim = simulate(simulation_type,im0_updated,coldef_type,coldef_strength); #im0_small_sim_arr = im0_small_sim.reshape(m*n,3)
    
    if modus==2:
        d0 = im0-im0_updated_sim;
        sigma = 1
        d0_blurred = gaussian_filter(d0,(sigma,sigma,0)) 
        #from scipy.ndimage.filters import gaussian_filter
        gradx0d = dxp1(d0_blurred,d0,dict); grady0d = dyp1(d0_blurred,d0,dict)
        mask = numpy.sqrt(gradx0d[:,:,0]**2+gradx0d[:,:,1]**2+gradx0d[:,:,2]**2 +\
                          grady0d[:,:,0]**2+grady0d[:,:,1]**2+grady0d[:,:,2]**2)
        mask = mask/numpy.max(mask)
        edge_threshold = dict['edge_threshold']
        mask[mask<edge_threshold]=.0; mask[mask>=edge_threshold]=1.0
        from scipy.ndimage.morphology import binary_dilation
        mask = binary_dilation(mask)
        mask_3D = numpy.array([mask.transpose(),]*d).transpose()
        
        dict['mask'] = mask_3D
        
        
    # Gradients of original image and simulated image 
    gradx0 = dxp1(im0_updated,im0_updated,dict); gradx0_arr = gradx0.reshape((m*n,3))
    grady0 = dyp1(im0_updated,im0_updated,dict); grady0_arr = grady0.reshape((m*n,3))
    gradx0s = dxp1(im0_updated_sim,im0_updated_sim,dict); gradx0s_arr = gradx0s.reshape((m*n,3))
    grady0s = dyp1(im0_updated_sim,im0_updated_sim,dict); grady0s_arr = grady0s.reshape((m*n,3))
   
    # Compute vectors in principal projection directions
    ed,el,ec = computeDaltonizationUnitVectors(im0_updated,im0_updated_sim,dict)
    
    if global_unit_vectors: k_x = numpy.dot(gradx0_arr,ed); k_y = numpy.dot(grady0_arr,ed)
    else: k_x = numpy.sum(gradx0_arr*ed, axis=1); k_y = numpy.sum(grady0_arr*ed, axis=1) # feil
    gradx0dot_arr = numpy.array([k_x,]*d).transpose(); grady0dot_arr = numpy.array([k_y,]*d).transpose()    
    
    no_chi_sign = False
    if not dict.has_key('chi_sign'): no_chi_sign = True; dict['chi_sign']=0
    chi_sign = dict['chi_sign'] if dict.has_key('chi_sign') else 1.0
    #if dict['chi_sign']==0: d0 = im0_updated - im0_updated_sim; #no_chi_sign = True
     
    chi_pos, chi_neg = computeChiAndLambda(gradx0_arr,grady0_arr,gradx0s_arr,grady0s_arr,ed,el,ec,dict)
    
    # Combination for the gradient 
    boost_ec = dict['boost_ec'] if dict.has_key('boost_ec') else 1.
    boost_el = dict['boost_el'] if dict.has_key('boost_el') else 1.
    combination = dict['combination'] if dict.has_key('combination') else 1
    if combination==1:
        if ms_first: sys.stdout.write("Chroma only.\n")
        if chi_sign == -1:
            if ms_first: sys.stdout.write("Chi-sign: green. ") 
            chi = chi_neg
        if chi_sign == 0:
            if ms_first: sys.stdout.write("Computing Chi-sign automatically: ")        
            if numpy.sum(numpy.abs(chi_neg)) > numpy.sum(numpy.abs(chi_pos)): 
                chi = chi_pos; 
                sys.stdout.write("red") if ms_first else sys.stdout.write("r")
            else: 
                chi = chi_neg; 
                sys.stdout.write("green") if ms_first else sys.stdout.write("g")
            if ms_first: sys.stdout.write(". ")
        elif chi_sign == 1:
            if ms_first: sys.stdout.write("Chi-sign: red. ") 
            chi = chi_pos
        elif chi_sign == 2:
            if ms_first: sys.stdout.write("Computing Chi-sign individually. ")
            # All of them are either bigger or smaller
            p=chi_pos.copy();p[chi_pos>=chi_neg]=0
            n=chi_neg.copy();n[chi_neg>chi_pos]=0
            chi=p+n
            
        chi[numpy.isnan(chi)] = 0.
        chi_arr = numpy.array([chi,]*d).transpose() 
        
        gradxdalt_arr = gradx0_arr+(gradx0dot_arr*boost_ec*chi_arr*ec); gradxdalt = gradxdalt_arr.reshape((m,n,3)) 
        gradydalt_arr = grady0_arr+(grady0dot_arr*boost_ec*chi_arr*ec); gradydalt = gradydalt_arr.reshape((m,n,3))
    elif combination==0:    
        if ms_first: sys.stdout.write("Lightness only.\n")
        gradxdalt_arr = gradx0_arr+(gradx0dot_arr*boost_el*chi_arr*el); gradxdalt = gradxdalt_arr.reshape((m,n,3)) 
        gradydalt_arr = grady0_arr+(grady0dot_arr*boost_el*chi_arr*el); gradydalt = gradydalt_arr.reshape((m,n,3))
    if no_chi_sign and modus!=1: del dict['chi_sign'] # keep the chi_sign for all steps in multiscaling
    
    #######
    ## Compute the daltonized image through gradient descent
    #######  
    
    if modus==3:
        
        chi_pos[numpy.isnan(chi_pos)] = 0.
        chi_pos = numpy.array([chi_pos,]*d).transpose() 
        gradxdalt_r_arr = gradx0_arr+(gradx0dot_arr*boost_ec*chi_pos*ec); gradxdalt_r = gradxdalt_r_arr.reshape((m,n,3)) 
        gradydalt_r_arr = grady0_arr+(grady0dot_arr*boost_ec*chi_pos*ec); gradydalt_r = gradydalt_r_arr.reshape((m,n,3))
        im_r_dalt = optimization(im,im0_updated,gradxdalt_r,gradydalt_r,dict)
        
        chi_neg[numpy.isnan(chi_neg)] = 0.
        chi_neg = numpy.array([chi_neg,]*d).transpose() 
        gradxdalt_g_arr = gradx0_arr+(gradx0dot_arr*boost_ec*chi_neg*ec); gradxdalt_g = gradxdalt_g_arr.reshape((m,n,3)) 
        gradydalt_g_arr = grady0_arr+(grady0dot_arr*boost_ec*chi_neg*ec); gradydalt_g = gradydalt_g_arr.reshape((m,n,3))
        im_g_dalt = optimization(im,im0_updated,gradxdalt_g,gradydalt_g,dict)
        
        
        mask = mask_3D[:,:,0]
        indices = mask_3D.astype(bool) 
        indices_2 = (im_r_dalt-im) > (im_g_dalt-im)
        
        im_dalt = im.copy()
        im_dalt[indices_2] = im_r_dalt[indices_2]
        im_dalt[~indices_2] = im_g_dalt[~indices_2]
        
    else:
        im_dalt = optimization(im,im0_updated,gradxdalt,gradydalt,dict)
    if ms_first: dict['ms_first'] = 0 # Only for first iteration
    if numpy.shape(im) == numpy.shape(im0): # Only for last iteration
        too_many = dict['too_many'] if dict.has_key('too_many') else 0
        if too_many: sys.stdout.write("\nX: Too many iterations. Breaking off.")
        sys.stdout.write('\n')
    
    return im_dalt

from scipy import ndimage

def optimization(im,im0,gradxdalt,gradydalt,dict):

    # Find optimization type    
    optimization = dict['optimization'] if dict.has_key('optimization') else 1
    ms_first=dict['ms_first'] if dict.has_key('ms_first') else 0
    if optimization==1: # Use Poisson optimization
        name = "Poisson optimization"
        if ms_first: sys.stdout.write("Poisson optimization, ")
    elif optimization==2: # Use total variation optimization
        name = "total variation optimization"
        if ms_first: sys.stdout.write("Total variation optimization, ")
    elif optimization==3: # Use anisotropic optimization
        name = "anisotropic optimization"
        if ms_first: sys.stdout.write("Anisotropic optimization, ")
    
    m,n,d = numpy.shape(im0)
    
    # Optional parameters
    cutoff = dict['cutoff'] if dict.has_key('cutoff') else .1
    max_its = dict['max_its'] if dict.has_key('max_its') else 1000
    data = dict['data'] if dict.has_key('data') else None
    is_simulated = dict['is_simulated'] if dict.has_key('is_simulated') else False
    dt = dict['dt']if dict.has_key('dt') else .25
    boundary = int(dict['boundary']) if dict.has_key('boundary') else 0
    modus = dict['modus'] if dict.has_key('modus') else 0
    
    gradx0 = dxp1(im0,im0,dict); grady0 = dyp1(im0,im0,dict)
    chroma_data_attachment = dict['chroma_data_attachment'] if dict.has_key('chroma_data_attachment') else 0
    skin_data_attachment = dict['skin_data_attachment'] if dict.has_key('skin_data_attachment') else 0
    
    l = 1.
    gauss_chroma = numpy.zeros((m,n,d))
    gauss_skin = numpy.zeros((m,n,d))
    if chroma_data_attachment:
        sigma_chroma = dict['sigma_chroma'] if dict.has_key('sigma_chroma') else 25./100
        if ms_first: sys.stdout.write('with chroma data attachment,  '+str(sigma_chroma)+': ')
        # Compute weighting function for data attachment based on chroma
        gauss_chroma = makeGaussianChromaMap(im0,sigma_chroma)     
    if skin_data_attachment:
        sigma_skin = dict['sigma_skin'] if dict.has_key('sigma_skin') else 1.
        if ms_first: sys.stdout.write('with skin data attachment '+str(sigma_skin)+': ')
        gauss_skin = makeGaussianSkinMap(im0,sigma_skin)
    
    if not (chroma_data_attachment or skin_data_attachment): 
        if ms_first: sys.stdout.write('without data attachment ') 
    gauss = gauss_chroma+gauss_skin
    
    if optimization==3: g = anisotropicG(gradx0,grady0,dict); dict['g']=g
    
    if modus==2:
        mask_3D = dict['mask']; #del dict['mask']
        mask = mask_3D[:,:,0]
        indices = mask_3D.astype(bool)
        edge_option = dict['edge']
        if edge_option==0: gradxdalt = mask_3D*gradxdalt; gradydalt = mask_3D*gradydalt

    if optimization==1: gradgradxdalt = dxm1(gradxdalt,dict); gradgradydalt = dym1(gradydalt,dict)
    im_new = im.copy(); cted = True; first_RMSE = True; its = 0; gradx = numpy.array([]); grady = numpy.array([])
    sys.stdout.write('|')
    #cted = False # wech
    
    print numpy.shape(im[indices]), numpy.shape(im[~indices])
    while cted:
        its += 1
        if (its // 100. == its / 100.): sys.stdout.write('.')
        if its >= max_its: cted = False; dict['too_many']=1; sys.stdout.write('X')
        
        gradx = dxp1(im_new,im0,dict); grady = dyp1(im_new,im0,dict)
        if modus==2: 
            if edge_option==0: gradx = mask_3D*gradx; grady = mask_3D*grady
        if optimization==1: opt = optimization_poisson(gradx,grady,gradgradxdalt,gradgradydalt,dict) # mask_3d wech
        elif optimization==2: opt = optimization_total_variation(gradx,grady,gradxdalt,gradydalt,dict) 
        elif optimization==3: opt = optimization_anisotropic(gradx,grady,gradxdalt,gradydalt,dict)
        
        if boundary==0: 
            a =  dt * opt[1:-1, 1:-1] - dt*(l*gauss)*(im_new-im0)[1:-1, 1:-1]# Keep boundary values constant
            im_new[1:-1, 1:-1] = im_new[1:-1, 1:-1] + a
        else: 
            a = dt*opt - dt*(l*gauss)*(im_new-im0)
            im_new = im_new + a
        
        im_new[im_new < 0.] = 0.; im_new[im_new > 1.] = 1. # Gamut clipping
        
        if modus==2:
            if first_RMSE: 
                first_RMSE = False; test = numpy.inf
                d_new = GRMSE({'gradx': gradx[indices], 'grady': grady[indices]}, \
                              {'gradx': gradxdalt[indices], 'grady': gradydalt[indices]})
            else: 
                d_old = d_new
                d_new = GRMSE({'gradx': gradx[indices], 'grady': grady[indices]}, \
                              {'gradx': gradxdalt[indices], 'grady': gradydalt[indices]})
                test = numpy.abs((d_old-d_new)/d_old) 
        else:
            if first_RMSE: 
                first_RMSE = False; test = numpy.inf
                d_new = GRMSE({'gradx': gradx, 'grady': grady}, \
                              {'gradx': gradxdalt, 'grady': gradydalt})
            else: 
                d_old = d_new
                d_new = GRMSE({'gradx': gradx, 'grady': grady}, \
                              {'gradx': gradxdalt, 'grady': gradydalt})
                test = numpy.abs((d_old-d_new)/d_old) 
        if (test < cutoff): cted = False 
        if data:
            if is_simulated:
                simulation_type = dict['yg_simulation_type']; coldef_type = dict['coldef_type']
                coldef_strength = dict['coldef_strength'] if dict.has_key('coldef_strength') else 1.0
                im_new_sim = simulate(simulation_type,im_new,coldef_type,coldef_strength)
                data.set_array(im_new_sim)
            else: data.set_array(im_new)
            plt.draw(); plt.show()
    if modus==2:
        #mask = mask_3D[:,:,0]
        #indices = mask.astype(bool)
        b = im.copy()
        b[indices] = im_new[indices]
        im_new = b
                
    if optimization==3: del dict['g']
    
    #im_new = mask_3D
    #im_new[gradxdalt==0]=.5
    
    return im_new 

def optimization_poisson(gradx,grady,gradgradxdalt,gradgradydalt,dict):
    """
    Using Poisson equation to solve optimization formula in order to obtain gradxdalt and gradydalt.
    """

    gradgradx = dxm1(gradx,dict); gradgrady = dym1(grady,dict)
    
    pois = gradgradx+gradgrady-(gradgradxdalt+gradgradydalt)
    
    return pois

def optimization_total_variation(gradx,grady,gradxdalt,gradydalt,dict):
    """
    Using total variation equation to solve optimization formula in order to obtain gradxdalt and gradydalt.
    """
    
    eps = 0.99 # Has influence on dt: Watch out for computational instabilities. Should be much less than 1. 
    fx = gradx-gradxdalt; fy = grady-gradydalt
    norm = numpy.sqrt(fx**2+fy**2)+eps   
    gradgradx = dxm1(fx/norm, dict); gradgrady = dym1(fy*norm, dict)
    tv = gradgradx + gradgrady
    
    return tv

def optimization_anisotropic(gradx,grady,gradxdalt,gradydalt,dict):
    """
    Using anisotropic equation to solve optimization formula in order to obtain gradxdalt and gradydalt.
    """
    
    g = dict['g']
    fx = gradx-gradxdalt; fy=grady-gradydalt
    gradgradx = dxm1(g*fx, dict); gradgrady = dym1(g*fy, dict)
    anis = (gradgradx + gradgrady)
    
    return anis
