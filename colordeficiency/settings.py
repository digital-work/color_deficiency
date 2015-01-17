'''
Created on 9. apr. 2014

Contains all the basic default settings

@author: joschua
'''

#from PIL import Image
import os

version="v0.1.2"

simulation_types = ["vienot", "vienot-adjusted", "kotera", "brettel","dummy"]
vienot = "vienot"
vienot_adjusted = "vienot-adjusted"
kotera = "kotera"
brettel = "brettel"
sim2ID = {  'none':             0,
            'vienot':           1,
            'vienot-adjusted':  2,
            'kotera':           3,
            'brettel':          4,
            'dummy':            99}
id2Sim = {  0: "none",
            1: "vienot",
            2: "vienot-adjusted",
            3: "kotera",
            4: "brettel",
            99: "dummy"}

daltonization_types = ["anagnostopoulos", "kotera", "kuhn", "huan","yoshi-simone","yoshi-alsam","dummy"]
anagnostopoulos = "anagnostopoulos"
kotera = "kotera"
kuhn = "kuhn"
huan = "huan"
yoshi_c2g = "yoshi_c2g"
yoshi_c2g_only = "yoshi_c2g_only"
dalt2ID = {'none':              0,
           'anagnostopoulos':   1,
           'kotera':            2,
           'kuhn':              3,
           'huan':              4,
           'yoshi-simone-only': 5,
           'yoshi-alsam-only':  6,
           'yoshi-simone':      7,
           'yoshi-alsam':       8,
           'dummy':             99}
id2Dalt = {0:   "none",
           1:   "anagnostopoulos",
           2:   "kotera",
           3:   "kuhn",
           4:   "huan",
           5:   "yoshi-simone-only",
           6:   "yoshi-alsam-only",
           7:   "yoshi-simone",
           8:   "yoshi-alsam",
           99:  "dummy"}

dict_dalt_ids = {'anagnastapoulos'}

coldef_types = ["d","p","t"]
coldef_types_long = ['deuteranopia', 'protanopia', 'tritanopia']
d = "d"
p = "p"
t = "t"
id2ColDef = {0: "normal",
             1: 'protanopia',
             2: 'deuteranopia',
             3: 'tritanopia'}

colDefLong2ID = {'normal':          0,
                 'protanopia':      1,
                 'deuteranopia':    2,
                 'tritanopia':      3}

colDef2ID = {'p': 1,
             'd': 2,
             't': 3}

default = {'simulation_type': 'brettel',
           'daltonization_type': 'anagnastopoulos',
           'coldef_type': 'p'}
module_path = os.path.dirname(os.path.abspath(__file__))

pts_default = 10
its_default = 100

size_default = 512,512

A1 = (23.4,33.1)
A4 = (8.267,11.692)

test_functions = [8]

#img_in = Image.open("images/0010000000.png")
#img_in.thumbnail(size_default)
#img_in.save("images/0020000000.png")
#img_in.show()