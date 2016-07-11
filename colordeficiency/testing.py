from pylab import *
import os

from colordeficiency import *

def griddata_boundaries(points, values, xi, method='linear', fill_value=nan, rescale=False):
    
    xi_copy = xi.copy()
    
    index_min = numpy.array(xi<=points[0])
    if len(index_min):
        xi_copy[index_min] = points[0]
    
    index_max = xi>=points[-1]
    if len(index_max):
        xi_copy[index_max] = points[-1]
    #print xi_copy
    ip = griddata(points, values, xi_copy, method, fill_value, rescale)
    
    return ip
    
    
def total_variation_dalt_1D():
    
    ion()

    x = linspace(0, 1)
    x_2 = linspace(0,1,100)
    y0 = zeros(shape(x))
    y0[25:] = 1
    y0 = y0 + .3 * randn(shape(x)[0])
    
    y0 -= numpy.min(y0)
    y0 = y0/numpy.max(y0)
    y_2 = griddata(x,y0,x_2,'linear')
    x = x_2.copy()
    y = y_2.copy()
    #y = y0.copy()
    
    line, = plot(x,y)
    
    lut_1D_in, lut_1D_out = numpy.array([0,25,50,75,100,125,150,175,200,225,255]), numpy.array([0,75,80,85,115,120,125,165,170,175,255])
    
    lut_1D_in, lut_1D_out = numpy.array([0,25,50,75,100,125,150,175,200,225,250]), numpy.array([25,30,35,40,55,50,55,60,65,70,75])
    #lut_1D_in, lut_1D_out = numpy.array([0,25,50,75,100,125,150,175,200,225,255]), numpy.array([0,25,50,75,100,125,150,175,200,225,255])
    #lut_1D_in, lut_1D_out = numpy.array([0,255]), numpy.array([0,255])
    
    lut_1D_in = lut_1D_in/255. * 1.0
    #print numpy.max(y0)
    lut_1D_out = lut_1D_out/255. * 1.0
    #print lut_1D_in, lut_1D_out
    
    #y_sim = griddata(lut_1D_in, lut_1D_out, y_2, 'linear')
    
    dt = .01
    lambd = 1
    eps = .00001
    
    switch = 0
    
    # Compute 1st and 2nd derivatives of LUT
    lut_in_d = lut_1D_in[1:] - lut_1D_in[:-1]
    
    lut_1D_d = lut_1D_out[1:] - lut_1D_out[:-1]
    lut_1deriv = lut_1D_out.copy()
    lut_1deriv[1:] = lut_1D_d / (lut_in_d+eps)
    lut_1deriv[0] = lut_1deriv[1]
    
    #print lut_in_d, lut_1D_d, lut_1deriv 
    
    lut_2deriv_d = lut_1deriv[1:] - lut_1deriv[:-1]
    lut_2deriv = lut_1D_out.copy()
    lut_2deriv[1:] = lut_2deriv_d / (lut_in_d+eps)
    lut_2deriv[0] = lut_2deriv[1]
    
    def suf(x): 
        return 0.5*x**2
    def dsduf(x):
        return x
    
    #lut_1D_out = suf(lut_1D_in)
    #lut_1deriv = dsduf(lut_1D_in)
    
    # Compute y0'
    y0d = y0[1:]-y0[:-1]
    y0d = y_2[1:] - y_2[:-1]
    #print numpy.shape(y0d)
    ylim(-1,4) 
    
    while False:
        if switch:
            setp(line, 'color', 'r')
            line.set_ydata(y)
            switch = 0
        else:
            setp(line, 'color', 'b')
            line.set_ydata(y)
            switch = 1
        draw()
    
    while True:
        dsdu = griddata_boundaries(lut_1D_in, lut_1deriv, y, 'linear')
        yd = y[1:]-y[:-1]
        
        v = numpy.sign(yd*dsdu[1:]-y0d)
        vd = v[:-1] - v[1:]
        
        y[1:-1] = y[1:-1]+dt*(-dsdu[1:-1]*vd)
        
        too_big = y >= 1.0
        #y[too_big] = 1.0
        
        too_small = y <= 0.0
        #y[too_small] = 0.0
        
        if switch:
            setp(line, 'color', 'r')
            line.set_ydata(y_2)
            switch = 0
        else:
            setp(line, 'color', 'b')
            line.set_ydata(y)
            switch = 1
        draw()

def gradientd(im,dir,type="forward"):
    
    dir = dir
    if not dir in['x','y']:
        print "You either have to choose the x or y axis."
        return
    
    type = type
    #print type
    if not type in ['forward','backward']:
        print "You either have to choose forward or backward."
        return
    
    m,n = numpy.shape(im)
    im_zero = numpy.zeros((m,n))
    
    imd = im_zero.copy()
    if dir == "x":
        imd1 = im[:,1:]
        imd2 = im[:,:-1]  
        if type == "forward":
            imd[:,1:] = imd1-imd2
        else:
            imd[:,:-1] = imd2-imd1
    else:
        imd1 = im[1:,:]
        imd2 = im[:-1,:]  
        if type == "forward":
            imd[1:,:] = imd1-imd2
        else:
            imd[:-1,:] = imd2-imd1
    
    return imd

def total_variation_dalt_2D():
    
    ion()

    im = imread(os.path.join(settings.image_path, '0010000000.png'))
    im = sum(im, 2) / 3.
    im0 = im.copy()
    
    lut_1D_in, lut_1D_out = numpy.array([0,25,50,75,100,125,150,175,200,225,255]), numpy.array([0,75,80,85,115,120,125,165,170,175,255])
    lut_1D_in, lut_1D_out = numpy.array([0,25,50,75,100,125,150,175,200,225,250]), numpy.array([25,30,35,40,55,50,55,60,65,70,75])
    #lut_1D_in, lut_1D_out = numpy.array([0,25,50,75,100,125,150,175,200,225,255]), numpy.array([0,25,50,75,100,125,150,175,200,225,255])
    #lut_1D_in, lut_1D_out = numpy.array([0,255]), numpy.array([0,255])
    lut_1D_in = lut_1D_in/255. * 1.0
    lut_1D_out = lut_1D_out/255. * 1.0
    
    data = imshow(im, cm.gray)
    
    dt = .01
    eps = .01
    
    m,n =numpy.shape(im)
    im_zero = numpy.zeros((m,n))
    
    def s(im):
        return griddata_boundaries(lut_1D_in, lut_1D_out, im, 'linear')
        #return im/3. + .6
        
    while True:
        start = s(im)-im0
        gradx = im_zero.copy(); gradx[:,1:] = start[:,1:] - start[:,:-1]
        grady = im_zero.copy(); grady[1:,:] = start[1:,:] - start[:-1,:]
        
        length = numpy.sqrt(gradx**2+grady**2) + eps
        vx = gradx/length
        vy = grady/length
        vdx = im_zero.copy(); vdx[:,:-1] = vx[:,1:]-vx[:,:-1]
        vdy = im_zero.copy(); vdy[:-1,:] = vy[1:,:]-vy[:-1,:]
        tv = vdx+vdy
        
        im[1:-1,1:-1] = im[1:-1,1:-1] + dt * tv[1:-1,1:-1]
        
        too_big = im >= 1.0
        im[too_big] = 1.0
        
        too_small = im <= 0.0
        im[too_small] = 0.0

        data.set_array(im)        
        draw()        

def total_variation_dalt_3D():
    
    ion()

    im = imread(os.path.join(settings.image_path, '0340000000-small.png'))
    #im = sum(im, 2) / 3.
    im0 = im.copy()
    
    lut_1D_in, lut_1D_out = numpy.array([0,25,50,75,100,125,150,175,200,225,255]), numpy.array([0,75,80,85,115,120,125,165,170,175,255])
    lut_1D_in, lut_1D_out = numpy.array([0,25,50,75,100,125,150,175,200,225,250]), numpy.array([25,30,35,40,55,50,55,60,65,70,75])
    #lut_1D_in, lut_1D_out = numpy.array([0,25,50,75,100,125,150,175,200,225,255]), numpy.array([0,25,50,75,100,125,150,175,200,225,255])
    #lut_1D_in, lut_1D_out = numpy.array([0,255]), numpy.array([0,255])
    lut_1D_in, lut_1D_out = makeSimulationLookupTable('brettel', 'd',5)
    lut_1D_in = lut_1D_in/255. * 1.0
    lut_1D_out = lut_1D_out/255. * 1.0
    
    #im0_srgb = colour.data.Data(colour.space.srgb,im0)
    #im0 = im0_srgb.get(colour.space.ipt)
    
    #print numpy.shape(lut_1D_in)    
    data = imshow(im)
    
    dt = .01
    eps = .0001
    
    m,n,d =numpy.shape(im)
    im_zero = numpy.zeros((m,n,d))
    print numpy.shape(im)
    def s(im):
        #return griddata_boundaries(lut_1D_in, lut_1D_out, im, 'linear')
        #return im/3. + .6
        return lookup(im, lut_1D_in, lut_1D_out)
        #retim = im.copy()
        #retim[..., 0] = .5 * (retim[..., 0] + retim[..., 1])
        #retim[..., 1] = .5 * (retim[..., 0] + retim[..., 1])
        #return retim
        
    while True:
        #print 'poo'
        #im_srgb = colour.data.Data(colour.space.srgb,im)
        #im_ipt = im_srgb.get(colour.space.ipt)
        
        start =  s(im)-im0
        
        #a = im_zero[1:,:-1]
        #print numpy.shape(a)
        gradx = im_zero.copy(); gradx[:,1:] = start[:,1:] - start[:,:-1]
        grady = im_zero.copy(); grady[1:,:] = start[1:,:] - start[:-1,:]
        
        length = numpy.sqrt(gradx**2+grady**2) + eps
        vx = gradx/length
        vy = grady/length
        vdx = im_zero.copy(); vdx[:,:-1] = vx[:,1:]-vx[:,:-1]
        vdy = im_zero.copy(); vdy[:-1,:] = vy[1:,:]-vy[:-1,:]
        tv = vdx+vdy
        
        #total = numpy.sum(numpy.sqrt(tv[:,:,0]**2+tv[:,:,1]**2+tv[:,:,2]**2))#+eps
        #print total
        #print numpy.sum(numpy.sqrt(im[:,:,0]**2)) / total+numpy.sum(numpy.sqrt(im[:,:,1]**2)) / total+numpy.sum(numpy.sqrt(im[:,:,2]**2)) / total
        #tv[:,:,0] = tv[:,:,0] * numpy.sum(numpy.sqrt(tv[:,:,0]**2)) / total
        #tv[:,:,1] = tv[:,:,1] * numpy.sum(numpy.sqrt(tv[:,:,1]**2)) / total
        #tv[:,:,2] = tv[:,:,2] * numpy.sum(numpy.sqrt(tv[:,:,2]**2)) / total
        
        im[1:-1,1:-1] = im[1:-1,1:-1] + dt *  tv[1:-1,1:-1]
        
        #im_ipt = colour.data.Data(colour.space.ipt,im_ipt)
        #im = im_ipt.get(colour.space.srgb)
        
        too_big = im >= 1.0
        im[too_big] = 1.0
        
        too_small = im <= 0.0
        im[too_small] = 0.0

        data.set_array((im))        
        draw()  
    
#total_variation_dalt_3D()



#a = numpy.array([[2,3,4],[2,3.4]])
#b = numpy.array([[1,3,4],[1,3,4]])

#print dir(a,b)

"""    

lut_3D_in, lut_3D_out = makeSimulationLookupTable('brettel','d')
im = Image.open(os.path.join(settings.image_path, '0010000000.png'))
#im.show()
im_sim = lookup(im, lut_3D_in, lut_3D_out)
#im_sim.show()
from pylab import *
ion()
lut_2D_in, lut_2D_out = numpy.array([0,25,50,75,100,125,150,175,200,225,255]), numpy.array([25,75,80,85,115,120,125,165,170,175,240])
#plot(lut_2D_in, lut_2D_out)
im_gray = im.convert('L')
#im_gray.show()
#im_sim_gray = lookup(im_gray,lut_2D_in,lut_2D_out)

data = imshow(im_gray, cm.gray)
draw()

im_gray_vec = numpy.reshape(numpy.asarray(im_gray),1000*1000)
from scipy.interpolate import griddata
im_gray_sim_vec = griddata(lut_2D_in, lut_2D_out, im_gray_vec, 'linear')
im_gray_sim = im_gray_sim_vec.reshape(1000,1000)
c = Image.fromarray(im_gray_sim.astype('uint8'))
c.show()

b = lookup(im_gray, lut_2D_in, lut_2D_out)
data.set_array(im_gray_sim)
draw()

while True:
    pass

#im_sim = simulate('brettel', im, 'd')
#im_sim.show()
"""


def tvdalt_colortv():
    # Same as pupsi2
    
    im = imread('../colordeficiency-images/0810000000.png')
    im0 = im.copy()
    figure(0)
    #imshow(im0, vmin=0, vmax=1)
    title('Original')
    
    #figure(1)
    #imshow(s(im0), vmin=0, vmax=1)
    title('Simulated')
    #show()
    
    #figure(2)
    ion()
    data = imshow(s(im), vmin=0, vmax=1)
    #title('Daltonised, simulated')
    show()
    draw()
    dict = {}
    sim = s(im)
    dt = .1
    eps = .01
    while True:
        # Color TV
        
        #sim = s(sim)
        gradx = dxp1(sim - im0,dict)
        grady = dyp1(sim - im0,dict)
        normgrad = sqrt(gradx**2 + grady**2) + eps
        tv_layer1 = numpy.sum(numpy.sqrt(gradx[:,:,0]**2)+numpy.sqrt(grady[:,:,0]**2))
        tv_layer2 = numpy.sum(numpy.sqrt(gradx[:,:,1]**2)+numpy.sqrt(grady[:,:,1]**2))
        tv_layer3 = numpy.sum(numpy.sqrt(gradx[:,:,2]**2)+numpy.sqrt(grady[:,:,2]**2))
        tv_all = numpy.sqrt(tv_layer1**2+tv_layer2**2+tv_layer3**2)
        #print layer1
        tv = dxm1(gradx / normgrad,dict) + dym1(grady / normgrad,dict)
        tv[:,:,0] = tv_layer1/tv_all*tv[:,:,0]
        tv[:,:,1] = tv_layer2/tv_all*tv[:,:,1]
        tv[:,:,2] = tv_layer3/tv_all*tv[:,:,2]
        # tv = dxm1(dxp1(s(im) - im0)) + dym1(dyp1(s(im) - im0))
        sim[1:-1, 1:-1] = sim[1:-1, 1:-1] + dt * tv[1:-1, 1:-1]
        sim[sim < 0] = 0
        sim[sim > 1] = 1
        
        #sim = s(sim)
        data.set_array(sim)
        draw()

def tvdalt_colortv_anisotropic():
    # Same as pupsi2
    
    im = imread('../colordeficiency-images/0010000000.png')
    im0 = im.copy()
    figure(0)
    
    gradx0 = dxp1(im0)
    grady0 = dyp1(im0)
    
    g_xx = gradx0*gradx0
    g_xy = gradx0*grady0
    g_yx = grady0*gradx0
    g_yy = grady0*grady0
        
    eig_pos = (g_xy+g_yy+numpy.sqrt((g_xx-g_yy)**2+4*(g_xy)**2))/2.
    eig_neg = (g_xy+g_yy-numpy.sqrt((g_xx-g_yy)**2+4*(g_xy)**2))/2.
        
    print numpy.min((g_xx-g_yy)**2+4*g_xy)
    #f = numpy.sqrt(eig_pos-eig_neg)
    f = numpy.sqrt(eig_pos)
    #f = eig_neg/eig_pos
    
    #g = 1/(1+f)
    g = numpy.exp(-f)
    #imshow(im0, vmin=0, vmax=1)
    title('Original')
    
    #figure(1)
    #imshow(s(im0), vmin=0, vmax=1)
    title('Simulated')
    #show()
    
    #figure(2)
    ion()
    data = imshow(im, vmin=0, vmax=1)
    #title('Daltonised, simulated')
    show()
    draw()
    dict = {}
    sim = s(im)
    dt = .25
    eps = .99
    while True:
        # Color TV
        
        #sim = s(sim)
        #gradx = dxp1(sim - im0)
        #grady = dyp1(sim - im0)
        gradx = dxp1(im)
        grady = dyp1(im)
        
        
        norm_grad = sqrt(gradx**2 + grady**2) + eps
        """
        tv_layer1 = numpy.sum(numpy.sqrt(gradx[:,:,0]**2)+numpy.sqrt(grady[:,:,0]**2))
        tv_layer2 = numpy.sum(numpy.sqrt(gradx[:,:,1]**2)+numpy.sqrt(grady[:,:,1]**2))
        tv_layer3 = numpy.sum(numpy.sqrt(gradx[:,:,2]**2)+numpy.sqrt(grady[:,:,2]**2))
        tv_all = numpy.sqrt(tv_layer1**2+tv_layer2**2+tv_layer3**2)
        #print layer1
        tv = dxm1(gradx / normgrad) + dym1(grady / normgrad)
        tv[:,:,0] = tv_layer1/tv_all*tv[:,:,0]
        tv[:,:,1] = tv_layer2/tv_all*tv[:,:,1]
        tv[:,:,2] = tv_layer3/tv_all*tv[:,:,2]
        """
        """
        g_xx = gradx*gradx
        g_xy = gradx*grady
        g_yx = grady*gradx
        g_yy = grady*grady
        
        eig_pos = (g_xy+g_yy+numpy.sqrt((g_xx-g_yy)**2+4*(g_xy)**2))/2.
        eig_neg = (g_xy+g_yy-numpy.sqrt((g_xx-g_yy)**2+4*(g_xy)**2))/2.
        
        print numpy.min((g_xx-g_yy)**2+4*g_xy)
        f = numpy.sqrt(eig_pos-eig_neg)
        #f = eig_neg/eig_pos
        """
        tv = dxm1(gradx/norm_grad) + dym1(grady/norm_grad)
        #print numpy.mean(g_xy-g_yx)
        # tv = dxm1(dxp1(s(im) - im0)) + dym1(dyp1(s(im) - im0))
        im[1:-1, 1:-1] = im[1:-1, 1:-1] + dt * tv[1:-1, 1:-1]
        im[im < 0] = 0
        im[im > 1] = 1
        
        #sim = s(sim)
        data.set_array(im)
        draw()
     
def tvdalt_channelbychannel():
    # Same as pupsi2b
    
    im = imread('../colordeficiency-images/berries2.png')
    im0 = im.copy()
    figure(0)
    #imshow(im0, vmin=0, vmax=1)
    title('Original')
    
    #figure(1)
    #imshow(s(im0), vmin=0, vmax=1)
    title('Simulated')
    #show()
    
    #figure(2)
    ion()
    data = imshow(s(im), vmin=0, vmax=1)
    #title('Daltonised, simulated')
    show()
    draw()
    
    sim = s(im0)
    dt = .01
    eps = .0001
    while True:
        #sim = s(sim)
        
        #sim = s(sim)
        # Channel-by-channel
        gradx = dxp1(sim - im0,dict)
        grady = dyp1(sim - im0,dict)
        
        normgrad = sqrt(gradx**2 + grady**2) + eps
        tv = dxm1(gradx / normgrad,dict) + dym1(grady / normgrad,dict)
        # tv = dxm1(dxp1(s(im) - im0)) + dym1(dyp1(s(im) - im0))
        sim[1:-1, 1:-1] = sim[1:-1, 1:-1] + dt * tv[1:-1, 1:-1]
        sim[sim < 0] = 0
        sim[sim > 1] = 1
        data.set_array(sim)
        draw()
        
        
def tvdalt_completenormbtwgradient():
    
    
    tvdalt_completenormbtwgradient_onoriginal
    # Same as pupsi3

    im = imread('../colordeficiency-images/berries2.png')
    im0 = im.copy()
    figure(0)
    #imshow(im0, vmin=0, vmax=1)
    title('Original')
    
    #figure(1)
    #imshow(s(im0), vmin=0, vmax=1)
    title('Simulated')
    #show()
    
    #figure(2)
    ion()
    data = imshow(s(im), vmin=0, vmax=1)
    #title('Daltonised, simulated')
    show()
    draw()
    
    sim = s(im)
    dt = .1
    eps = .01
    while True:
        gradx = dxp1(sim - im0)
        grady = dyp1(sim - im0)
        normgrad_onelayer = sqrt(gradx[:,:,0]**2 + grady[:,:,0]**2+ \
                                 gradx[:,:,1]**2 + grady[:,:,1]**2+ \
                                 gradx[:,:,2]**2 + grady[:,:,2]**2) + eps
        m,n = numpy.shape(normgrad_onelayer)
        normgrad = numpy.zeros((m,n,3))
        normgrad[:,:,0] = normgrad_onelayer
        normgrad[:,:,1] = normgrad_onelayer
        normgrad[:,:,2] = normgrad_onelayer
        tv = dxm1(gradx / normgrad) + dym1(grady / normgrad)
        # tv = dxm1(dxp1(s(im) - im0)) + dym1(dyp1(s(im) - im0))
        sim[1:-1, 1:-1] = sim[1:-1, 1:-1] + dt * tv[1:-1, 1:-1]
        sim[sim < 0] = 0
        sim[sim > 1] = 1
        #sim = s(sim)
        data.set_array(sim)
        draw()
        
def tvdalt_completenormbtwabsoutegradient():
    # Same as pupsi4
    
    im = im = imread('../colordeficiency-images/berries2.png')
    im0 = im.copy()
    figure(0)
    #imshow(im0, vmin=0, vmax=1)
    title('Original')
    
    #figure(1)
    #imshow(s(im0), vmin=0, vmax=1)
    title('Simulated')
    #show()
    
    #figure(2)
    ion()
    data = imshow(s(im), vmin=0, vmax=1)
    #title('Daltonised, simulated')
    show()
    draw()
    
    gradx0 = dxp1(im0)
    grady0 = dyp1(im0)
    norm_im0 = sqrt(gradx0[:,:,0]**2+grady0[:,:,0]**2+gradx0[:,:,1]**2+grady0[:,:,1]**2+gradx0[:,:,2]**2+grady0[:,:,2]**2)
    
    #err = im0 - s(im0)
    dt = .1
    eps = .01
    #lambd = 0.05
    m,n,d = numpy.shape(im)
    
    sim = s(im)
    while True:
        
        gradx = dxp1(sim)
        grady = dyp1(sim)
        norm_sim = numpy.sqrt(gradx[:,:,0]**2 + grady[:,:,0]**2+gradx[:,:,1]**2 + grady[:,:,1]**2+gradx[:,:,2]**2 + grady[:,:,2]**2) + eps
        
        func = norm_sim - norm_im0
        sign = numpy.sign(func)
        sign_3 = numpy.zeros((m,n,3))
        norm_sim_3 = numpy.zeros((m,n,3))
        for i in range(0,d):
            sign_3[:,:,i] = sign
            norm_sim_3[:,:,i] = norm_sim
        
        tv = dxm1(sign_3 * gradx / norm_sim_3) + dym1(sign_3 * grady / norm_sim_3)
        sim[1:-1, 1:-1] = sim[1:-1, 1:-1] + dt*tv[1:-1, 1:-1]
        
        sim[sim < 0] = 0
        sim[sim > 1] = 1
        data.set_array(sim)
        draw()

def computeGradientperChannel(im,s_func,delta_u):
    
    sim = s_func(im)
    m,n,d = numpy.shape(im)
    
    im_gradient = []
    
    for i in range(0,d):
        im_i_temp = im.copy()
        im_i_temp[:,:,i] = im[:,:,i]+delta_u
        im_i_temp[im_i_temp > 1] = 1
        
        sim_istar = s_func(im_i_temp) 
        im_gradient_ui = (sim_istar - sim) / delta_u
        im_gradient.append(im_gradient_ui)
    im_gradient = numpy.array(im_gradient)
    
    return im_gradient

def tvdalt_completenormbtwgradient_onoriginal():
    
    # See page 5 of total variation formulas

    im = imread('../colordeficiency-images/berries2a.png')
    im0 = im.copy()
    figure(0)
    #imshow(im0, vmin=0, vmax=1)
    title('Original')
    
    #figure(1)
    #imshow(s(im0), vmin=0, vmax=1)
    title('Simulated')
    #show()
    
    #figure(2)
    ion()
    data = imshow(im, vmin=0, vmax=1)
    #title('Daltonised, simulated')
    show()
    draw()
    
    #sim = s(im)
    dt = .1
    eps = .01
    m,n,d = numpy.shape(im)
    delta_u = 0.1
    while True:
        
        print "Next round"
        
        gradx = dxp1(s(im) - im0)
        grady = dyp1(s(im) - im0)
        normgrad_onelayer = sqrt(gradx[:,:,0]**2 + grady[:,:,0]**2+gradx[:,:,1]**2 + grady[:,:,1]**2+gradx[:,:,2]**2 + grady[:,:,2]**2) + eps
        #m,n = numpy.shape(normgrad_onelayer)
        normgrad = numpy.zeros((m,n,3))
        for i in range(0,d):
            normgrad[:,:,i] = normgrad_onelayer
        
        ###
        gradgradx = dxm1(gradx / normgrad)
        gradgradx_arr = gradgradx.reshape((m*n,d))
        gradgrady = dym1(grady / normgrad)
        gradgrady_arr = gradgrady.reshape((m*n,d))
        im_gradient = computeGradientperChannel(im,s,delta_u)
        
        tv = numpy.zeros((m,n,d))
        for i in range(0,d):
            gradient_i = im_gradient[i]
            gradient_i_arr = gradient_i.reshape((m*n,d))
            
            x_component_i_arr = numpy.sum(gradient_i_arr * gradgradx_arr,axis=1)
            x_component_i = x_component_i_arr.reshape((m,n))
            y_component_i_arr = numpy.sum(gradient_i_arr * gradgrady_arr,axis=1)
            y_component_i = y_component_i_arr.reshape((m,n))
            tv[:,:,i] = x_component_i + y_component_i
        ###

        im[1:-1, 1:-1] = im[1:-1, 1:-1] + dt * tv[1:-1, 1:-1]
        im[im < 0] = 0
        im[im > 1] = 1
        #sim = s(sim)
        data.set_array(im)
        draw()

def tvdalt_completenormbtwabsoutegradient_onoriginal():
    # Same as pupsi4
    
    im = im = imread('../colordeficiency-images/berries2.png')
    im0 = im.copy()
    figure(0)
    #imshow(im0, vmin=0, vmax=1)
    title('Original')
    
    #figure(1)
    #imshow(s(im0), vmin=0, vmax=1)
    title('Simulated')
    #show()
    
    #figure(2)
    ion()
    data = imshow(s(im), vmin=0, vmax=1)
    #title('Daltonised, simulated')
    show()
    draw()
    
    
    gradx0 = dxp1(im0)
    grady0 = dyp1(im0)
    norm_im0 = sqrt(gradx0[:,:,0]**2+grady0[:,:,0]**2+gradx0[:,:,1]**2+grady0[:,:,1]**2+gradx0[:,:,2]**2+grady0[:,:,2]**2)

    dt = .1
    eps = .001
    delta_u = 0.1
    m,n,d = numpy.shape(im)
    while True:
        
        print "Next round"
        
        gradx = dxp1(s(im))
        grady = dyp1(s(im))
        normgrad_onelayer = numpy.sqrt(gradx[:,:,0]**2 + grady[:,:,0]**2+gradx[:,:,1]**2 + grady[:,:,1]**2+gradx[:,:,2]**2 + grady[:,:,2]**2) + eps
        
        func = normgrad_onelayer - norm_im0
        sign = numpy.sign(func)
        sign_3 = numpy.zeros((m,n,3))
        normgrad = numpy.zeros((m,n,3))
        for i in range(0,d):
            sign_3[:,:,i] = sign
            normgrad[:,:,i] = normgrad_onelayer
        
        #tv = dxm1(sign_3 * gradx / normgrad) + dym1(sign_3 * grady / normgrad)
        ###
        gradgradx = dxm1(sign_3*gradx / normgrad); gradgradx_arr = gradgradx.reshape((m*n,d))
        gradgrady = dym1(sign_3*grady / normgrad); gradgrady_arr = gradgrady.reshape((m*n,d))
        im_gradient = computeGradientperChannel(im,s,delta_u)
        
        tv = numpy.zeros((m,n,d))
        for i in range(0,d):
            gradient_i = im_gradient[i]
            gradient_i_arr = gradient_i.reshape((m*n,d))
            
            x_component_i_arr = numpy.sum(gradient_i_arr * gradgradx_arr,axis=1)
            x_component_i = x_component_i_arr.reshape((m,n))
            y_component_i_arr = numpy.sum(gradient_i_arr * gradgrady_arr,axis=1)
            y_component_i = y_component_i_arr.reshape((m,n))
            tv[:,:,i] = x_component_i + y_component_i
        
        ###
        im[1:-1, 1:-1] = im[1:-1, 1:-1] + dt*tv[1:-1, 1:-1]
        
        im[im < 0] = 0
        im[im > 1] = 1
        data.set_array(im)
        draw()

def computeTestName(dict):
    #print dict
    test_name = ''
    if dict['modus']==1: test_name += 'multi-scaling_'
    elif dict['modus']==2: test_name += 'smoothing_'
    else: test_name += 'simple_'
    
    if dict['global_unit_vectors']: test_name += 'glob-uv_'
    else: test_name += 'indi-uv_'
    
    if dict['chroma_data_attachment']: test_name += '-chroma-da'+str('{0:.2f}'.format(dict['sigma_chroma']))+'_'
    else: test_name += 'non-chroma-da_'
    
    if dict['skin_data_attachment']: test_name += '-skin-da'+str('{0:.2f}'.format(dict['sigma_skin']))+'_'
    else: test_name += 'non-skin-da_'
            
    if dict['constant_lightness']: test_name += 'constant-lightness_'
    else: test_name += 'neutral-gray_'
    
    if dict['chi_computations']==1: test_name += 'chi1_'
    elif dict['chi_computations']==2: test_name += 'chi2_'
        
    if dict['chi_sign']==1.: test_name += 'chired_'
    elif dict['chi_sign']==-1.: test_name += 'chigreen_'
    elif dict['chi_sign']==0: test_name += 'chiauto_'
    elif dict['chi_sign']==2.: test_name += 'chiindi_'
        
    if dict['ed_orthogonalization']: test_name += 'ed90_'
    else: test_name += 'edXX_'
        
    if dict['optimization']==1: test_name += 'poisson-optimization_'
    elif dict['optimization']==2: test_name += 'tv-optimization_'
    elif dict['optimization']==3: test_name += 'anisotropic-optimization-'+str(dict['anisotropic'])+'_'
    
    #print dict['boundary']
    if dict['boundary']==0: test_name += 'nobound_'  
    else: test_name += str(int(dict['boundary']))+'bound_'
            
    test_name += dict['yg_simulation_type']+'-'
    test_name += dict['coldef_type']
    return test_name

def tvdalt_engineeredgradient():
    
    images = []
    #images.append(('0340000000-mirr',{'modus':1}))
    #images.append(('0370000000',{'modus':1}))
    images.append(('0890000000',{'modus':1}))
    #images.append(('0340000000',{'modus':1}))
    #images.append(('0550000000',{'modus':1}))
    #images.append(('0520000000',{'modus':1}))
    #images.append(('0460000000',{'modus':1}))
    #images.append(('0460000000',{'modus':1}))
    #images.append(('berries2',{'modus':1}))
    #images.append(('0790000000',{'modus': 1}))
    #images.append(('0810000000',{'modus': 1}))
    #images.append(('0810000000-blurred',{'modus': 0}))
    #images.append(('berries2-inverted',{'modus':0}))
    #images.append(('berries2',{'modus':0}))
    #im_names.append('bananas1')
    #images.append(('berries1',{'modus': 0}))
    #im_names.append('0030000000')
    #im_names.append('berries2-gradient')
    for image in images:
        
        im_name = image[0]
        print im_name
        im = imread(os.path.join('../colordeficiency-images/',im_name+'.png'))
        im = im[...,0:3]
        im0 = im.copy()
        dict_list = []
        
        figure(0); ion(); show()
        data = imshow(im0, vmin=0, vmax=1)
        title("Daltonised"); draw()
        
        # Show different unit vectors
        # ed
        simulation_type = 'brettel'
        coldef_type = 'p'
        coldef_strength = 0.5
        dict_1 = image[1]
        dict_1.update({'constant_lightness': 1, # 1 - constant lightness, 0 - neutral gray
                       'chi_computations': 1, 
                       'ed_orthogonalization': 0,
                       'chi_sign': 0, # 1. - change red color, -1. - change green colors, 0 - automatic
                       'optimization': 1, # 1 - poisson, 2 - total variation, 3 - anisotropic
                       'boundary': 2, # 0 - pass, 1 - gradu[0] = gradu0[0] , 2 - gradu[0] = 0, 3 - gradgradu[0] = 0
                       'interp': "cubic",
                       'data': data,
                       'dt': .20,
                       'chroma_data_attachment': 1,
                       'skin_data_attachment': 0,
                       'sigma_chroma': .05,
                       'sigma_skin': 1.0,
                       #'data2': data2,
                       #'max_sigma': 10,
                       'max_its': 1000,
                       'cutoff': .00005,
                       'is_simulated': 0,
                       'yg_simulation_type': simulation_type,
                       'global_unit_vectors': 0, 
                       'coldef_type': coldef_type,
                       'coldef_strength': coldef_strength,
                       'im_name': im_name,
                       'path_out': os.path.join('/Users/thomas/Desktop/different-unit-vectors/ed'),
                        #'im_out': os.path.join('/Users/thomas/Desktop/different-unit-vectors',im_name)
                    })
        # el
        dict_3 = dict_1.copy(); dict_3.update({'path_out': os.path.join('/Users/thomas/Desktop/different-unit-vectors/el')})
        dict_4 = dict_3.copy(); dict_4.update({'ed_orthogonalization': 1})
        dict_5 = dict_3.copy(); dict_5.update({'constant_lightness': 0})
        dict_6 = dict_5.copy(); dict_6.update({'ed_orthogonalization': 1})
        # ec
        dict_7 = dict_1.copy(); dict_7.update({'path_out': os.path.join('/Users/thomas/Desktop/different-unit-vectors/ec')})
        dict_8 = dict_7.copy(); dict_8.update({'chi_sign': -1.})
        
        # Show different chi computatins
        dict_9 = dict_1.copy(); dict_9.update({'path_out': os.path.join('/Users/thomas/Desktop/chi-computations')})
        dict_10 = dict_9.copy(); dict_10.update({'chi_computations': 2})
        
        # Show different modi
        dict_11 = dict_1.copy(); dict_11.update({'path_out': os.path.join('/Users/thomas/Desktop/multi-scaling')})
        dict_12 = dict_11.copy()
        if dict_11['modus']: dict_12.update({'modus': 0})
        else: dict_12.update({'modus': 1})
        
        # Use different boundary computations
        dict_13 = dict_1.copy(); dict_13.update({'path_out': os.path.join('/Users/thomas/Desktop/boundaries')})
        dict_14 = dict_13.copy(); dict_14.update({'boundary': 1})
        dict_15 = dict_13.copy(); dict_15.update({'boundary': 2})
        dict_16 = dict_13.copy(); dict_16.update({'boundary': 3})
        
        # Use different optimization computations
        dict_18 = dict_1.copy(); dict_18.update({'path_out': os.path.join('/Users/thomas/Desktop/optimization')})
        dict_19 = dict_18.copy(); dict_19.update({'optimization': 2})
        dict_20 = dict_1.copy(); dict_20.update({'path_out': os.path.join('/Users/thomas/Desktop/anisotropic'), 
                                                 'anisotropic': 0,
                                                 'optimization': 3})
        dict_21 = dict_20.copy(); dict_21.update({'anisotropic': 3})
        
        dict_22 = dict_20.copy(); dict_22.update({'anisotropic': 1})
        dict_23 = dict_20.copy(); dict_23.update({'anisotropic': 2})
        
        # Use global or individual unit vectors
        dict_24 = dict_1.copy(); dict_24.update({'path_out': os.path.join('/Users/thomas/Desktop/different-unit-vectors'),
                                                 'global_unit_vectors': 0})
        dict_25 = dict_24.copy(); dict_25.update({'global_unit_vectors': 1})
                
        # Use global vs individual chi-sign computation
        dict_28 = dict_1.copy(); dict_28.update({'path_out': os.path.join('/Users/thomas/Desktop/chi-sign'),
                                                 'global_unit_vectors': 1,
                                                 'chi_sign': 0})
        dict_29 = dict_28.copy(); dict_29.update({'global_unit_vectors': 0,
                                                  'chi_sign': 0})
        
        # Use w/ or /wo chroma data attachment
        dict_30 = dict_1.copy(); dict_30.update({'path_out': os.path.join('/Users/thomas/Desktop/data-attachment')})
        dict_31 = dict_30.copy(); dict_31.update({'chroma_data_attachment': 1,
                                                  'sigma': .01})
        dict_32 = dict_31.copy(); dict_32.update({'sigma': .05})
        dict_33 = dict_31.copy(); dict_33.update({'sigma': .25})
        dict_34 = dict_31.copy(); dict_34.update({'sigma': .5})
        dict_35 = dict_31.copy(); dict_35.update({'sigma': .75})
        
        # Use skin data attachment
        dict_36 = dict_1.copy(); dict_36.update({'path_out': os.path.join('/Users/thomas/Desktop/skin-data-attachment')})
        dict_37 = dict_36.copy(); dict_37.update({'skin_data_attachment': 1,
                                                  'sigma_skin': 1.})
        #dict_list.append(dict_1)
        #dict_list.append(dict_2)
        #dict_list.append(dict_3)
        #dict_list.append(dict_4)
        #dict_list.append(dict_5)
        #dict_list.append(dict_6)
        #dict_list.append(dict_7)
        #dict_list.append(dict_8)
        #dict_list.append(dict_9)
        #dict_list.append(dict_10)
        #dict_list.append(dict_11)
        #dict_list.append(dict_12)
        
        #dict_list.append(dict_13)
        #dict_list.append(dict_14)
        #dict_list.append(dict_15)
        #dict_list.append(dict_16)
        #dict_list.append(dict_17)

        #dict_list.append(dict_18)
        #dict_list.append(dict_19)
        #dict_list.append(dict_20)
        #dict_list.append(dict_21)
        
        #dict_list.append(dict_24)
        #dict_list.append(dict_25)
        #dict_list.append(dict_26)
        #dict_list.append(dict_27)
        dict_list.append(dict_28)
        dict_list.append(dict_29)
        #dict_list.append(dict_30)
        #dict_list.append(dict_31)
        #dict_list.append(dict_32)
        #dict_list.append(dict_33)
        #dict_list.append(dict_34)
        #dict_list.append(dict_35)
        #dict_list.append(dict_36)
        #dict_list.append(dict_37)
        
        
        for dict_i in dict_list:
            
            dict_i.update({'test_name': computeTestName(dict_i)})
            im_dalt = daltonization_yoshi_042016(im,dict_i)
            
            if not os.path.isdir(dict_i['path_out']):
                os.makedirs(dict_i['path_out'])
                print "Caution: Created directory ["+dict_i['path_out']+']'
            
            if 1:
                print "Saving '"+dict_i['im_name']+"' image: "+dict_i['test_name']
                print dict_i['path_out']
                imsave(os.path.join(dict_i['path_out'],dict_i['im_name']+'_'+'_orig'+'.png'), im)
                imsave(os.path.join(dict_i['path_out'],dict_i['im_name']+'_'+'_orig-sim'+'.png'), simulate(simulation_type,im,coldef_type))
                imsave(os.path.join(dict_i['path_out'],dict_i['im_name']+'_'+dict_i['test_name']+'_dalt'+'.png'), im_dalt.copy())
                imsave(os.path.join(dict_i['path_out'],dict_i['im_name']+'_'+dict_i['test_name']+'_dalt-sim'+'.png'), simulate(simulation_type,im_dalt.copy(),coldef_type))  
        
    print "Ferdig"
    
tvdalt_engineeredgradient()
#tvdalt_colortv_anisotropic()
#tvdalt_colortv()