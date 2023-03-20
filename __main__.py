'''
  Created on 6. mar 2023.
  
  Last updated on 6. mars. 2023.
  
  @author. joschua
  
  This package is used to simulate and daltonize images and arrays.
  

'''

#from os import path
import sys
#print(sys.executable)
#sys.path.append(path.abspath('../colourspace'))
#sys.path.append('..\colourspace')
#print(path)
import colour

import argparse
import os

from PIL import Image


from colordeficiency.colordeficiency import simulate

coldef_types = {
      'p': 'protan',
      'd': 'deutan',
      't': 'tritan'
   }

sim_methods = {
      'v': "vienot", 
      'va': 'vienot-adjusted', 
      'k': 'kotera',
      'b': 'brettel',
      'd': 'dummy'
   }

def colordeficiency():
   
   parser = argparse.ArgumentParser("A command line script to simulate color deficiency or daltonize images.")
 
   parser.add_argument('-i', '--img', help="Url to image file to be enhanced.")
   
   parser.add_argument('--type', default='p', const='p', nargs='?', choices=['p','d','t'])
   parser.add_argument('--method', default='va', const='va', nargs='?', choices=['v','va', 'k','b','d'])
 
   args = parser.parse_args()
   
   print(vars(args))
   
   # Retrieve the coldef type.
   coldef_type = args.type
   print("The chosen coldef type is {}.".format(coldef_types[coldef_type]))
   
   # Retrieve the simulaiton method.
   sim_method = args.method
   print("The chosen simulation method is {}.".format(sim_methods[sim_method]))
   
   # Retrieve original image.
   img = args.img
   if not img:
      print("No image has been chosen.")
      return
   elif not os.path.exists(img):
      print("The chosen image ({}) does not exist.".format(img))
   else:
      
      # Show the original image.
      im_orig = Image.open(img)
      im_orig.show()
      
      # Simulatte the original image.
      im_sim = simulate(sim_methods[sim_method],im_orig,coldef_type)
      
      # Show the simulated image.
      im_sim.show()

if __name__=='__main__':
  colordeficiency()