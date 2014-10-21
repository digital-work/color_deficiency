'''
Created on 24. feb. 2014

@author: joschua
'''

import os
import numpy

from scipy.interpolate import griddata
import settings 
from PIL import Image
import pylab
import sys
#import colour
#import scipy
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def convertToLuminanceImage(img_in):
    
    img_in = img_in.convert('RGB')
    img_in_array = numpy.asarray(img_in, dtype=float)
    
    """
    sRGBOriginal_arr = colour.data.Data(colour.space.srgb, img_array)
    LABOriginal_arr = sRGBOriginal_arr.get(colour.space.cielab)
    
    LABLuminance_arr = LABOriginal_arr.copy()
    LABLuminance_arr[:,:,1] = .0
    LABLuminance_arr[:,:,2] = .0
    
    LABLuminance_arr = colour.data.Data(colour.space.cielab,LABLuminance_arr)
    sRGBLuminance_arr = LABLuminance_arr.get(colour.space.srgb)
    """
    
    img_out_array = numpy.dot(img_in_array[...,:3], [0.299, 0.587, 0.144])
    img_out_array[img_out_array<0.0] = 0.0
    img_out_array[img_out_array>255.0] = 255.0
    
    img_out = Image.fromarray(numpy.uint8(img_out_array))
    
    return img_out
 


def makeSubplots(size,imgs_in,options={}):
    """
    Makes a plot including all subplots of size with input images in imgs
    Input:    size     - tuple (columns,rows), where m are the columns and n the rows.
              imgs_in  - images as dictionaries {img_in, title, font-size}  
    """
    
    if (not isinstance(size,tuple)) and not (numpy.shape(size)==2):
        print "Error: size must be a tuple of form (columns,row)."
        return None
    columns = size[0]
    rows = size[1]
    
    if options.has_key('output_name'):
        output_name = options['output_name']
    else:
        output_name = "./images/tmp/make-subplots-tmp.png"
    
    if options.has_key('dpi'):
        dpi = options['dpi']
    else:
        dpi = 300
    
    if options.has_key('size_inches'):
        size_inches = options['size_inches']
    else:
        size_inches = settings.A4
    
    i = 1
    fig = pylab.figure()
    fig.dpi = dpi
    
    for img in imgs_in:
        #print img
        if img.has_key('img_in'):
            img_in = img['img_in']
            
            a = numpy.shape(img_in)
            if a[0]<a[1]:
                pass
            else:
                size = tuple(numpy.array(size).transpose())
            fig.set_size_inches(size_inches)
            
            pylab.subplot(columns,rows,i)
            pylab.imshow(img_in)
            
            if img.has_key('title'):
                title = img['title']
            else:
                title = "No title."
                
            if img.has_key('fontsize'):
                fontsize = img['fontsize']
            else:
                fontsize = 10.
            pylab.title(title,fontsize=fontsize)
            pylab.axis("off")
        else:
            pass
        i = i+1
    
    pylab.savefig(output_name)
    pylab.close()
    
    return None
        
    
def makeComparisonFig(img_in,coldef_types,simulation_types,name):
    
    sys.stdout.write("Making comparison figure for: "+name)
    
    n_coldefs = numpy.shape(coldef_types)[0]
    n_simulations = numpy.shape(simulation_types)[0]
    
    fig = pylab.figure()
    a = numpy.shape(img_in)
    if a[0]>a[1]:
        fig.set_size_inches(23.4,33)
    else:
        fig.set_size_inches(33,23.4)
            
    fig.dpi = 600
    i = 1
        
    for coldef_type in coldef_types:  
        sys.stdout.write(' ... ' + coldef_type)
        #print simulation_type
        """  
        if a[0]>a[1]:    
            pylab.subplot(n_coldefs,n_simulations,i)
        else:
            pylab.subplot(n_simulations,n_coldefs,i)
        pylab.title("Original : " + name, fontsize=28.)
        pylab.imshow(img_in)
        pylab.axis("off")
        i = i+1
        """    
        for simulation_type in simulation_types: 
            #print coldef_type
            #print i
            img_in_sim = simulate(simulation_type, img_in, coldef_type)
            
            pylab.subplot(n_coldefs,n_simulations,i)
            """
            if a[0]>a[1]:    
                pylab.subplot(n_coldefs,n_simulations,i)
            else:
                pylab.subplot(n_simulations,n_coldefs,i)
            """
                
            pylab.title(simulation_type+" : "+coldef_type, fontsize=67.)
            pylab.imshow(img_in_sim)
            pylab.axis("off")
            i = i + 1
    sys.stdout.write('\n')


     

def simDaltAndSave(file_in, simulation_types=[], daltonization_types=[], coldef_types=[], dir_out=''):
    """
    Takes the input image and makes a quick simulation, a quick daltonization and a simulation of the daltonized images. Saves all computed and 
    Input:     file_in:                Path to the input image
               simulation_types:       Simulations that should be computed of the originaland the daltonized versions. No simulations as default.
               daltonization_types:    Daltonizations that should be computed. No daltonizations as default.
               dir_out:                Directory where the daltonized and simulation will be stored. Directory of file_in is default.
    
    """
    
    if coldef_types:
        dir_in = os.path.dirname(file_in)
        if dir_out == "":
            dir_out = dir_in
        
        basename = os.path.splitext(os.path.basename(file_in))
        name =  basename[0]
        extension =  basename[1]
        
        img_in = Image.open(file_in)
        file_format = img_in.format
        
        name_img_in = name+"-orig"+extension
        img_in.save(os.path.join(dir_out,name_img_in),file_format)  
        
        if simulation_types and daltonization_types:
            for coldef_type in coldef_types: # Do both daltonization and simulation
                print "STARTING daltonization and simulation for " + str(coldef_type),
                print ".. DALTONIZING",
                for daltonization_type in daltonization_types:
                    print ".. "+str(daltonization_type),
                    img_dalt_temp = daltonize(daltonization_type,img_in,coldef_type)
                    name_dalt_temp = name+"_dalt-"+str(daltonization_type)+"-"+str(coldef_type)+extension
                    img_dalt_temp.save(os.path.join(dir_out,name_dalt_temp),file_format)
                    print ".. SIMULATING",
                    for simulation_type in simulation_types:
                        print ".. "+str(simulation_type),
                        img_in_sim = simulate(simulation_type,img_in,coldef_type)
                        name_in_sim = name+"_sim-"+str(simulation_type)+"-"+str(coldef_type)+extension
                        img_in_sim.save(os.path.join(dir_out,name_in_sim), file_format)
                        
                        img_dalt_sim = simulate(simulation_type,img_dalt_temp,coldef_type)
                        name_dalt_sim = name+"_dalt-"+str(daltonization_type)+"_sim-"+str(simulation_type)+"-"+str(coldef_type)+extension
                        img_dalt_sim.save(os.path.join(dir_out,name_dalt_sim),file_format)
                print "... END"
                
        elif simulation_types and not daltonization_types:
            for coldef_type in coldef_types: # Do only simulation
                print "STARTING simulation for " + str(coldef_type),
                #simDaltAndSave("images/example80.jpg",["vienot-adjusted"], ['kotera'], ["p", "d", "t"], "images/bruno") 
                for simulation_type in simulation_types:
                    print ".. "+str(simulation_type),
                    img_in_sim = simulate(simulation_type,img_in,coldef_type)
                    name_in_sim = name+"_sim-"+str(simulation_type)+"-"+str(coldef_type)+extension
                    img_in_sim.save(os.path.join(dir_out,name_in_sim), file_format)
                print ".... END"
                    
        elif not simulation_types and daltonization_types: 
            for coldef_type in coldef_types: # Do only daltonization
                print "STARTING daltonization for " + str(coldef_type),
                for daltonization_type in daltonization_types:
                    print ".. "+str(daltonization_type),
                    img_dalt_temp = daltonize(daltonization_type,img_in,coldef_type)
                    name_dalt_temp = name+"_dalt-"+str(daltonization_type)+"-"+str(coldef_type)+extension
                    img_dalt_temp.save(os.path.join(dir_out,name_dalt_temp),file_format)
                print ".... END"
                    
        else:
            print "Error: No simulation or daltonization has been chosen"
            return
    else:
        print "Error: No color deficiency type has been chosen."
        return
 
def makeSimulationLookupTable(simulation_type, coldef_type,accuracy=5):
    
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
    #input_img.show()
    
    output_img = simulate(simulation_type, input_img, coldef_type)
    #output_img.show()
    output_array = numpy.asarray(output_img)
    output_tab = numpy.reshape(output_array,(mn,3))    
    
                
    return input_tab, output_tab

def lookup(img_in, input_tab, output_tab):
    
    input_arr = numpy.asarray(img_in)
    m,n,d = numpy.shape(img_in)
    
    input_vec = numpy.reshape(input_arr,(m*n,3))
    
    #im = rand(100,100,3)
    #inputdata = im.reshape(10000,3)
    if False:
        new_r = griddata(input_tab, output_tab[:,0], input_vec, 'linear')#.reshape(m,n)
        new_g = griddata(input_tab, output_tab[:,1], input_vec, 'linear')#.reshape(m,n)
        new_b = griddata(input_tab, output_tab[:,2], input_vec, 'linear')#.reshape(m,n)
    
        output_vec =numpy.array([new_r, new_g, new_b])
        print numpy.shape(output_vec)
        output_arr = numpy.reshape(output_vec.transpose(), (m,n,3))
    else:
        new_r = griddata(input_tab, output_tab[:,0], input_vec, 'linear').reshape(m,n)
        new_g = griddata(input_tab, output_tab[:,1], input_vec, 'linear').reshape(m,n)
        new_b = griddata(input_tab, output_tab[:,2], input_vec, 'linear').reshape(m,n)
        
        output_arr = input_arr.copy()
        output_arr[:,:,0] = new_r
        output_arr[:,:,1] = new_g
        output_arr[:,:,2] = new_b
    
    img_out = Image.fromarray(numpy.uint8(output_arr))
    
    return img_out

