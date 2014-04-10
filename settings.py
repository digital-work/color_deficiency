'''
Created on 9. apr. 2014

Contains all the basic default settings

@author: joschua
'''

from PIL import Image

simulation_types = ["vienot", "vienot-adjusted", "kotera", "brettel"]
brettel = "brettel"
vienot = "vienot"
vienot_adjusted = "vienot-adjusted"
kotera = "kotera"

daltonization_types = ["anagnostopoulos", "kotera", "yoshi_c2g", "yoshi_c2g_only"]
anagnostopoulos = "anagnostopoulos"
kotera = "kotera"
yoshi_c2g = "yoshi_c2g"
yoshi_c2g_only = "yoshi_c2g_only"

coldef_types = ["d","p","t"]
d = "d"
p = "p"
t = "t"

pts_default = 10
its_default = 100

A1 = (23.4,33.1)
A4 = (8.267,11.692)

img_in = Image.open("images/example1.jpg")