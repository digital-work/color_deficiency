###
### File created by Joschua Simon-Liedtke on 22nd of January 2014
###

from PIL import Image
import numpy
import scipy.io
import colour

simulation_types = ["vienot", "vienot-adjusted", "IPT"]
daltonization_types = ["anagnostopoulos", "kotera"]
coldef_types = ["d","p","t"]
img_in = Image.open("images/test1.jpg")

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
        #Matrix for deuteranopes
        alpha   = sum(rgb2lms[1][0:3]) * rgb2lms[2][2]- rgb2lms[1][2] * sum(rgb2lms[2][0:3])
        beta    = sum(rgb2lms[2][0:3]) * rgb2lms[0][2]- rgb2lms[2][2] * sum(rgb2lms[0][0:3])
        gamma   = sum(rgb2lms[0][0:3]) * rgb2lms[1][2]- rgb2lms[0][2] * sum(rgb2lms[1][0:3])
        
        l_p = (-1.)/( alpha ) * numpy.array([0,beta,gamma])
        l = l_p 
        
        m = numpy.array([0,1,0])
        s = numpy.array([0,0,1])
        
        
    elif coldef_type == "d":
        #Matrix for protanopes
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
        return img_in
    
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

def simulation_IPT(img_in, coldef_type, coldef_strength=1.0):
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
    elif simulation_type == "IPT":
        img_out = simulation_IPT(img_in, coldef_type,coldef_strength)
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
    
    cost = numpy.linalg.norm(numpy.dot(rdic,shiftImage_vector))
    
    return cost

def visualGapCostKotera(shiftImage_vector,deltaCDic_vector,rlms,rdic):
    
    cost = 0.
    
    cost = numpy.linalg.norm(deltaCDic_vector+numpy.dot((rlms-rdic),shiftImage_vector))
    
    return cost
    
def costKotera(shiftImage_vector,deltaCDic_vector,rlms,rdic):
    
    cost = 0.
    
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
    
    itv = numpy.linspace(0,k,20)
    costs = [0,0]
    for i in itv:
        cost = costKotera(lambdaShiftKotera(deltaCDic_vector,i),deltaCDic_vector,rlms,rdic)
        if cost >= cost_opt:
            cost_opt = cost
            lambda_opt = i
    
    deltaCStarSht_vector = lambdaShiftKotera(deltaCDic_vector,lambda_opt)
    print lambda_opt, cost_opt
    
    
    cDaltLMS_vector = cStarLMS_vector + deltaCStarSht_vector
    
    lmsOut_vector = numpy.dot(lmsMatchFuncs.transpose(),cDaltLMS_vector)
    xyzOut_vector = numpy.dot(lms2xyz,lmsOut_vector)
    
    xyzOut_arr = colour.data.Data(colour.space.xyz, numpy.reshape(xyzOut_vector.transpose(), (m,n,3)))
    sRGBOut_arr = xyzOut_arr.get(colour.space.srgb)*255.
    img_array = numpy.uint8(sRGBOut_arr)
    
    img_out = Image.fromarray(img_array)
    
    return img_out
   

def daltonize( daltonization_type, img_in, coldef_type ):
    """
    Function to daltonize image for color deficient people.
    Input:  daltonization_type -  Type of daltonization as defined in daltonization_types
            img_in -              Original PIL image
            coldef_type -         Type of color deficiency. Either p for protanopes, d for deuteranopes, or t for tritanopes.
    Output: img_out -             Simulated PIL image
    """
    
    if daltonization_type == "anagnastopoulos":
        img_out = daltonization_anagnostopoulos(img_in, coldef_type)
    elif daltonization_type == "kotera":
        img_out = daltonization_kotera(img_in, coldef_type)
    else:
        print 'Error: Daltonization type does not exist. Choose either one of the following - "'+'" , "'.join(daltonization_types)+'".'
        return img_in
    return img_out

def test2():
    
    name = "test10"
    coldef_type = "d"
    simulation_type = "IPT"
    daltonization_type = "kotera"
    img_in = Image.open("images/"+name+".jpg")
    img_in.show()
    img_in_sim = simulate(simulation_type, img_in, coldef_type)
    img_in_sim.show()
    
    img_out = daltonize(daltonization_type, img_in, coldef_type)
    img_out.show()
    img_out_sim = simulate(simulation_type, img_out, coldef_type)
    img_out_sim.show()
    

def test1():
    name = "test7"
    
    im = Image.open("images/"+name+".jpg")
    #im.show()
    simulation_type = "vienot"
    coldef_strength = 1.0
    
    for coldef_type in coldef_types:
        im_sim = simulate(simulation_type,im,coldef_type,coldef_strength)
        #im_sim.show()
        im_sim.save(name+"-"+simulation_type+"-"+coldef_type+".jpg")
        print coldef_type + " simulation done"    

test2()