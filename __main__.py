'''
  Created on 6. mar 2023.
  
  Last updated on 6. mars. 2023.
  
  @author. joschua
  
  This package is used to simulate and daltonize images and arrays.
  

'''


import argparse
import os

from PIL import Image

from colordeficiency.colordeficiency import simulate


def colordeficiency():
   
   parser = argparse.ArgumentParser("A command line script to simulate color deficiency or daltonize images.")
 
   parser.add_argument('-i', '--img', help="Url to image file to be enhanced.")
 
   args = parser.parse_args()
   
   # Retrieve original image
   
   img = args.img
   if not img:
      print("No image has been chosen.")
   elif not os.path.exists(img):
      print("The chosen image ({}) does not exist.".format(img))
   else:
      
      im = Image.open(img)
      im.show()

if __name__=='__main__':
  colordeficiency()