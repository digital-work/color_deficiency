'''
Created on 24. feb. 2014

@author: joschua
'''

from colordeficiency import *
from tools import makeComparisonFig, makeSubplots
import os
import sys
import subprocess
import settings

def test11():
    
    #os.system("./stress -i ./images/example13.png")
    os.environ['PATH'] = os.environ['PATH'] + ":/usr/local/bin"
    print os.environ['PATH']
    os.system("./stress -i ./images/example13.jpg -o ./test13.png -g -ns 1 -ni 200")
    #subprocess.call("./stress -i ./images/example11.jpg -o ./test1.png -g -ns 1 -ni 200")

def test10():
    """
    Testing yoshi_c2g algorithm
    """
    coldef_types = [p,d,t]
    simulation_type = brettel
    daltonization_types = [yoshi_c2g,yoshi_c2g_only]
    size = 1024,1024
    
    enhances = [1,0]
    pts = 5
    its = 100
    
    names = ["images/database/trit9.jpg","images/database/ber3.jpg","images/database/wrest26.jpg","images/database/nat2.jpg","images/database/pap11.jpg"]
    #name = "images/example1.jpg"
    #name = "images/database/wrest30.jpg"
    for name in names:
        print name
        img_in = Image.open(name)
        for daltonization_type in daltonization_types:
            print daltonization_type
            for enhance in enhances:
                for coldef_type in coldef_types:
                    img_in.thumbnail(size)
                    img_in_sim = simulate(simulation_type,img_in,coldef_type)
                    
                    dict_dalt = {'daltonization_type':daltonization_type, 'coldef_type':coldef_type}
                    dict_dalt.update({'enhance':enhance, 'pts':pts, 'its':its})
                    img_out = daltonize(img_in,dict_dalt)
                    img_out_sim = simulate(simulation_type,img_out,coldef_type)
                    
                    size_plts = (2,2)
                    imgs = [{'img_in':img_in,'title':'Orig. image','fontsize':40.}]
                    imgs.append({'img_in':img_in_sim,'title':'Orig. sim:\"'+simulation_type+"\" | "+coldef_type,'fontsize':40.})
                    
                    if enhance:
                        imgs.append({'img_in':img_out,'title':'dalt:\"'+daltonization_type+"\" +\n pts:"+str(pts)+" its:"+str(its)+" | "+coldef_type,'fontsize':40.})
                        imgs.append({'img_in':img_out_sim,'title':'sim:\"'+simulation_type+"\' | dalt:\'"+daltonization_type+"\' +\n pts:"+str(pts)+" its:"+str(its)+" | "+coldef_type,'fontsize':40.})
                    else:
                        imgs.append({'img_in':img_out,'title':'dalt:\"'+daltonization_type+"\" | "+coldef_type,'fontsize':40.})
                        imgs.append({'img_in':img_out_sim,'title':'sim:\"'+simulation_type+"\' | dalt:\'"+daltonization_type+"\' | "+coldef_type,'fontsize':40.})
                    
                    if not enhance and daltonization_type=="yoshi_c2g":
                        output_name = './images/yoshi_c2g/subplots-'+os.path.basename(name)+"-"+coldef_type+"-"+daltonization_type+"-not-enhanced.png"
                    else:
                        output_name = './images/yoshi_c2g/subplots-'+os.path.basename(name)+"-"+coldef_type+"-"+daltonization_type+"-enhanced-pts-"+str(pts)+"-its-"+str(its)+".png"

                    options = {'output_name':output_name,'size_inches':A1}
                    fig = makeSubplots(size_plts,imgs,options)
    
def test9():
    
    name = "example"
    best = [7] #33
    coldef_types = [p,d,t]
    simulation_types = [vienot,vienot_adjusted,kotera,brettel]
    size = 1024,1024
    
    for b in best:
        name_tmp = name+str(b)
        sys.stdout.write("Computing \'"+name_tmp+"\'.")       
        img_in = Image.open("images/buginbrettel/"+name_tmp+".png")
        img_in.thumbnail(size)     
        for simulation_type in simulation_types:
            sys.stdout.write(str(simulation_type).upper()+".")
            for coldef_type in coldef_types:
                sys.stdout.write(coldef_type+".")
                pylab.figure()
                pylab.subplot(121)
                pylab.title("Original image")
                pylab.axis("off")
                pylab.imshow(img_in)
                
                img_out = simulate(simulation_type,img_in,coldef_type)
                name_sim = "images/buginbrettel/tritanoper/"+name_tmp+"-"+simulation_type+"-"+coldef_type+"-sim.jpg"
                img_out.save(name_sim)
                
                pylab.subplot(122)
                pylab.title("Sim.: "+str(simulation_type)+" | ColDef.: "+str(coldef_type))
                pylab.imshow(img_out)
                pylab.axis("off")
                name_comparison = "images/buginbrettel/tritanoper/"+name_tmp+"-"+simulation_type+"-"+coldef_type+"-comparison.jpg"
                pylab.savefig(name_comparison)
                pylab.close()
                
                
        sys.stdout.write("ok.\n")
        #makeComparisonFig(img_in,coldef_types,simulation_types,name_tmp)   
        #pylab.savefig(name_tmp + "-comparison2.png")
    #pylab.show()
    sys.stdout.write("Done, gurl ;)")

def test8():
    print "Started"
    #fileList = os.listdir("images/database")
    fileList = ["wrest30.jpg","wrest30.jpg","pupsi.jpg"]
    coldef_types = [p,d]
    simulation_types = [vienot,kotera,brettel]
    size = 1024,1024#1024,1024#512,512
    n_files = numpy.shape(fileList)
    i=0
    
    for name in fileList:
        #print name
        i = i+1
        #print name
        save_name = "images/database/"+ name + "-comparison.png"
        if not os.path.isfile(save_name) and "comparison" not in name:
            try: 
                img_in = Image.open("images/database/"+name)
                img_in.thumbnail(size)
                
                sys.stdout.write(str(i) + "/"+str(n_files[0])+" --- ")
                
                makeComparisonFig(img_in, coldef_types,simulation_types,name)
                pylab.savefig(save_name)
                pylab.close()
                #img_in.show()
            except IOError:
                sys.stdout.write(str(i) + "/"+str(n_files[0])+" --- Error: Could not load image: \'" + name + "\'\n")
                
                pass
    print "Finished!"
    pylab.show()

def test7():
    
    name = "example"
    best = [10,15,17,18,19,33] #33
    coldef_types = [t]
    simulation_types = [brettel]
    size = 1024,1024
    
    for b in best:
        name_tmp = name+str(b)       
        img_in = Image.open("images/"+name_tmp+".jpg")
        img_in.thumbnail(size)        
        #img_in.show()
        
        makeComparisonFig(img_in,coldef_types,simulation_types,name_tmp)   
        pylab.savefig(name_tmp + "-comparison2.png")
    #pylab.show()
    
def test6():
    #best = [10,12,14,20,25,5,6] 
    #best = [101,102]
    best = [5]
    
    name = "example"
    coldef_types = ["p","d"]
    simulation_types = ["vienot","kotera"]
    daltonization_type = "anagnatopoulos"
    size = 256, 256
    
    #import pylab
    for b in best:
        print 1
        for simulation_type in simulation_types:
            print 2
            for coldef_type in coldef_types:
                print "starting with " + coldef_type
                #pylab.figure()
                name_tmp = name+str(b)
                
                img_in = Image.open("images/egne/"+name_tmp+".jpg")
                
                img_in.thumbnail(size, Image.ANTIALIAS)
                
                img_in_sim = simulate(simulation_type,img_in,coldef_type)
                #                 img_in_sim.save("images/"+name_tmp+"_"+str(coldef_type)+"_sim-"+str(simulation_type)+".jpg")
                img_in_sim.show()
                #                 
                #                 img_dalt = daltonize(daltonization_type,img_in,coldef_type)
                #                 img_dalt.save("images/"+name_tmp+"_"+str(coldef_type)+"_dalt-"+str(daltonization_type)+".jpg")
                #                 img_dalt.show()
                #                 
                #                 img_dalt_sim = simulate(simulation_type,img_dalt,coldef_type)
                #                 img_dalt_sim.save("images/"+name_tmp+"_"+str(coldef_type)+"_dalt-"+str(daltonization_type)+"_sim-"+str(simulation_type)+".jpg")
                #                 img_dalt_sim.show()
                
                #pylab.savefig(name_tmp+"-"+coldef_type+"-costs-fig.png")
                #img_sim.show()
    pylab.show()

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