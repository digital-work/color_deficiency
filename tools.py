'''
Created on 24. feb. 2014

@author: joschua
'''

from colordeficiency import simulate, daltonize
from PIL import Image
import os

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
    
simDaltAndSave("images/example80.jpg",["vienot-adjusted"], ['kotera'], ["p", "d", "t"], "images/bruno") 