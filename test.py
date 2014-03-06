'''
Created on 24. feb. 2014

@author: joschua
'''

from colordeficiency import *

def test6():
    best = [10,12,14,20,25,5,6] 
    
    name = "example"
    coldef_type = "d"
    simulation_type = "vienot-adjusted"
    daltonization_type = "kotera"
    size = 256, 256
    
    for b in best:
        name_tmp = name+str(b)
        
        img_in = Image.open("images/"+name_tmp+".jpg")
        
        img_in.thumbnail(size, Image.ANTIALIAS)
        
        img_sim = daltonize(daltonization_type,img_in,coldef_type)
        img_sim.show()



def test5():
    """
    Make example images to illustrate evaluation methods
    """
    
    best = [1,6,10,70] # Img 5 illustrates daltonization, img 10 illustrates the color deficiency verification experiments and , img 1 illustrates the daltonization evaluation experiment - visual search, img 60 illustrates the daltonization evaluation experiment - object recognition 
    
    name = "example"
    coldef_type = "d"
    simulation_type = "vienot-adjusted"
    daltonization_type = "kotera"
    size = 128, 128
    
    for b in best:
        name_tmp = name+str(b)
        
        img_in = Image.open("images/"+name_tmp+".jpg")
        
        #img_in.thumbnail(size, Image.ANTIALIAS)
        img_in.save("images/presentation/"+name_tmp+"orig.jpg", "JPEG")
        #img_in.show()
        img_in_sim = simulate(simulation_type, img_in, coldef_type)
        #img_in_sim.show()
        img_in_sim.save("images/presentation/"+name_tmp+"orig-sim.jpg", "JPEG")
        
        img_out = daltonize(daltonization_type, img_in, coldef_type)
        #img_out.show()
        img_out.save("images/presentation/"+name_tmp+"dalt.jpg", "JPEG")
        img_out_sim = simulate(simulation_type, img_out, coldef_type)
        #img_out_sim.show()
        img_out_sim.save("images/presentation/"+name_tmp+"dalt-sim.jpg", "JPEG")
    
def test4():
    best = [43,46,48,49,51,52,53,56]
    
    name = "example"
    coldef_type = "d"
    simulation_type = "vienot-adjusted"
    daltonization_type = "kotera"
    size = 128, 128
    
    for b in best:
        name_tmp = name+str(b)
        
        img_in = Image.open("images/"+name_tmp+".jpg")
        
        img_in.thumbnail(size, Image.ANTIALIAS)
        img_in.save("images/best/"+name_tmp+"orig.jpg", "JPEG")
        #img_in.show()
        img_in_sim = simulate(simulation_type, img_in, coldef_type)
        #img_in_sim.show()
        img_in_sim.save("images/best/"+name_tmp+"orig-sim.jpg", "JPEG")
        
        img_out = daltonize(daltonization_type, img_in, coldef_type)
        #img_out.show()
        img_out.save("images/best/"+name_tmp+"dalt.jpg", "JPEG")
        img_out_sim = simulate(simulation_type, img_out, coldef_type)
        #img_out_sim.show()
        img_out_sim.save("images/best/"+name_tmp+"dalt-sim.jpg", "JPEG")
                
        
def test3():
    
    name = "example10"
    simulation_type = "vienot-adjusted"
    coldef_type = "d"
    size = 512, 512
    
    input_tab, output_tab = makeSimulationLookupTable(simulation_type, coldef_type,4)
    #print input_tab, output_tab
    
    if True:
        img_in = Image.open("images/"+name+".jpg")
        img_in.thumbnail(size, Image.ANTIALIAS)
        #img_in.show()
        
        sRGB_in = colour.data.Data(colour.space.srgb,numpy.asarray(img_in)/255.)
        
        t = time.time()
        img_lut = lookup(img_in, input_tab, output_tab)
        print time.time()-t
        img_lut.show()
        
        sRGB_lut = colour.data.Data(colour.space.srgb,numpy.asarray(img_lut)/255.)
    
        t = time.time()
        img_sim = simulate(simulation_type,img_in,coldef_type)
        print time.time()-t
        img_sim.show()
        
        sRGB_sim = colour.data.Data(colour.space.srgb,numpy.asarray(img_sim)/255.)
        
        diff = colour.metric.dE_E(sRGB_in, sRGB_lut)
        print diff
        #print numpy.shape(diff)
        
        import pylab
        a = Image.fromarray(diff)
        #print numpy.max(diff)
        a.show()
        a = a.convert('RGB')
        a.save("./images/difference.jpg","JPEG")
        pylab.imshow(diff)
    else:
        import os
        size = 512,512
        directory = os.path.join("./images/test2/")
        for root,dirs,files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg'):
                    dir = os.path.join(directory,file)
                    print dir
                    img_in = Image.open(dir)        
                    t = time.time()
                    img_lut = lookup(img_in, input_tab, output_tab)
                    #print time.time()-t
                    img_lut.thumbnail(size, Image.ANTIALIAS)
                    img_lut.show()
    


def test2():
    
    name = "example48"
    coldef_type = "d"
    simulation_type = "IPT"
    daltonization_type = "kotera"
    size = 512, 512
    
    img_in = Image.open("images/"+name+".jpg")
    img_in.thumbnail(size, Image.ANTIALIAS)
    img_in.show()
    img_in_sim = simulate(simulation_type, img_in, coldef_type)
    img_in_sim.show()
    
    img_out = daltonize(daltonization_type, img_in, coldef_type)
    img_out.show()
    img_out_sim = simulate(simulation_type, img_out, coldef_type)
    img_out_sim.show()

test5()

def test1():
    name = "test18"
    
    im = Image.open("images/"+name+".jpg")
    #im.show()
    simulation_type = "vienot"
    coldef_strength = 1.0
    
    for coldef_type in coldef_types:
        im_sim = simulate(simulation_type,im,coldef_type,coldef_strength)
        #im_sim.show()
        im_sim.save(name+"-"+simulation_type+"-"+coldef_type+".jpg")
        print coldef_type + " simulation done"