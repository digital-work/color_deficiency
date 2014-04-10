###
### File created by Joschua Simon-Liedtke on 22nd of January 2014
###

from PIL import Image, ImageDraw, ImageEnhance
from scipy.interpolate import griddata
#from tools import showLMSSpace
import numpy
import scipy.io
import colour
import time
import math
import pylab
import os
from settings import *
#import cv2

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
    
    im = img_in.copy()
    converter = ImageEnhance.Color(im)
    im = converter.enhance(0.33)
    del converter
    
    
    draw = ImageDraw.Draw(im)
    draw.line((0, 0) + im.size, fill=128, width=10)
    draw.line((0, im.size[1], im.size[0], 0), fill=128, width=10)
    del draw
    
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

def simulation_kotera(img_in, coldef_type, coldef_strength=1.0):
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
        print "Error: Unknown color deficiency chosen. Chose either p for protanopes, d for deuteranopes, or t for tritanopes."
        return img_in
    
    data = numpy.genfromtxt('data/ciexyz31.csv', delimiter=',')
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
    sRGBSimulated_arr = lmsSimulated_arr.get(colour.space.srgb)*255.
    img_array = numpy.uint8(sRGBSimulated_arr)
    img_out = Image.fromarray(img_array)
    
    #showLMSSpace()
    
    #img_out = img_in
    
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
    img_out = img_in
    
    if simulation_type == "vienot":
        img_out = simulation_vienot(img_in, coldef_type,coldef_strength)
    elif simulation_type == "vienot-adjusted":
        img_out = simulation_vienot_adjusted(img_in, coldef_type,coldef_strength)
    elif simulation_type == "kotera":
        img_out = simulation_kotera(img_in, coldef_type,coldef_strength)
    elif simulation_type == "brettel":
        img_out = simulation_brettel(img_in, coldef_type, coldef_strength)
    else:
        print 'Error: Simulation type does not exist. Choose either one of the following - "'+'" , "'.join(simulation_types)+'".'
        img_out = img_in
    return img_out

def daltonization_anagnostopoulos(img_in, coldef_type):
    """
    """
    if not (coldef_type == "p" or coldef_type == "d"):
        print "Error: Unknown color deficiency chosen. Chose either p for protanopes, or d for deuteranopes."
        return img_in
    
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
    #print k,mn
    
    a_first = fund_img[0:lamda+1,:]
    a_last = fund_img[lamda+1:k,:]
    
    fundShift_img = numpy.vstack([a_last, a_first])
    #print numpy.shape(fundShift_img)
    
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
    
def daltonization_kotera(img_in, coldef_type):
    """
    """
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
    data = numpy.genfromtxt('data/ciexyz31.csv', delimiter=',')
    xyzMatchFuncs = data[:,1:4]
    xyz2lms = numpy.array([[.15514, .54312, -.03286],
                           [-.15514, .45684, .03286],
                           [0., 0., .00801]])
    lms2xyz = numpy.linalg.inv(xyz2lms)
    lmsMatchFuncs = numpy.dot(xyzMatchFuncs,xyz2lms.transpose())
    
    lms_vector = numpy.dot(xyz_vector,xyz2lms.transpose())
    k,dd =  numpy.shape(lmsMatchFuncs)
    
    a = lmsMatchFuncs    
    pinv = numpy.dot(a,numpy.linalg.inv(numpy.dot(a.transpose(),a)))
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
    print "hiersimmer",
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
    cost_opt = costs[lambda_opt,3]
    
    deltaCStarSht_vector = lambdaShiftKotera(deltaCDic_vector,lambda_opt)
    print lambda_opt*5, cost_opt, costs[lambda_opt+3,3]
    
    
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
        its = its_default # number of iterations (ni) should be >100
    
    if options.has_key('pts'):
        pts = options['pts']
    else:
        pts =  pts_default # number of sample points (ns) should be between 2 and 10, where 2 means local adjustments and 10 means mainly global adjustments
        # The noise is anti-proportional to the product ni*ns, which should be > 1000.
        # The complexityis proportional to the product ni*ns. 
        
    name_in_tmp = "./images/tmp/img_in.tmp.png"
    name_out_tmp = "./images/tmp/img_out.tmp.png"
    img_in.save(name_in_tmp)
        
    if not ":/usr/local/bin" in os.environ['PATH']:
        os.environ['PATH'] = os.environ['PATH'] + ":/usr/local/bin"
        print os.environ['PATH']
    os.system("./stress -i "+name_in_tmp+" -o "+name_out_tmp+" -g -ns "+str(pts)+" -ni "+str(its)) # Run the c2g C++ script in a shell
        
    img_gray = Image.open(name_out_tmp)
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
        
    if options.has_key('coldef_type'):
        coldef_type = options['coldef_type']
    else:
        print 'Error: No coldef_type. You have to choose a valid color deficiency type'
        return img_in
       
    # 1.step: convert original image to IPT
    img_in = img_in.convert('RGB')
    img_array = numpy.asarray(img_in, dtype=float)/255.
    
    sRGBOriginal_arr = colour.data.Data(colour.space.srgb,img_array)
    iptOriginal_arr = sRGBOriginal_arr.get(colour.space.ipt)
    
    # 2.step: Enhance luminance channel if wanted.
    
    # Check if user wants to use enhanced gray channel. False by default
    if options.has_key('enhance'): 
        enhance = options['enhance']
    else:
        enhance = False
        
    if enhance:
        # Make gray image of original RGB image
        if options.has_key('its'):
            its = options['its']
        else:
            its = its_default # number of iterations (ni) should be >100
        
        if options.has_key('pts'):
            pts = options['pts']
        else:
            pts =  pts_default # number of sample points (ns) should be between 2 and 10, where 2 means local adjustments and 10 means mainly global adjustments
        # The noise is anti-proportional to the product ni*ns, which should be > 1000.
        # The complexityis proportional to the product ni*ns. 
        
        name_in_tmp = "./images/tmp/img_in.tmp.png"
        name_out_tmp = "./images/tmp/img_out.tmp.png"
        img_in.save(name_in_tmp)
        
        if not ":/usr/local/bin" in os.environ['PATH']:
            os.environ['PATH'] = os.environ['PATH'] + ":/usr/local/bin"
            print os.environ['PATH']
        os.system("./stress -i "+name_in_tmp+" -o "+name_out_tmp+" -g -ns "+str(pts)+" -ni "+str(its)) # Run the c2g C++ script in a shell
        
        img_gray = Image.open(name_out_tmp)
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
        iptDaltonized_arr[:,:,0] = imgLuminance_arr+iptP_arr
    elif coldef_type == 't':
        iptT_arr = iptOriginal_arr[:,:,2]
        #Image.fromarray(iptT_arr*255.).show()
        iptDaltonized_arr[:,:,0] = imgLuminance_arr+iptT_arr
        
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
    
    if options.has_key('daltonization_type'):
        daltonization_type = options['daltonization_type']
    else:
        print "Error: No daltonization_type. You have to choose a valid daltonization_type."
        return img_in
    
    if options.has_key('coldef_type'):
        coldef_type = options['coldef_type']
    else:
        print 'Error: No coldef_type. You have to choose a valid color deficiency type'
        return img_in
    
    if daltonization_type == "anagnostopoulos":
        img_out = daltonization_anagnostopoulos(img_in,coldef_type)
    elif daltonization_type == "kotera":
        img_out = daltonization_kotera(img_in,coldef_type)
    elif daltonization_type == "yoshi_c2g":
        img_out = daltonization_yoshi_c2g(img_in,options)
    elif daltonization_type == "yoshi_c2g_only":
        img_out = daltonization_yoshi_c2g_only(img_in,options)
    else:
        print 'Error: Daltonization type does not exist. Choose either one of the following - "'+'" , "'.join(daltonization_types)+'".'
        return img_in
    return img_out