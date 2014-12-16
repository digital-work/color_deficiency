import colour
from colordeficiency.colordeficiency import simulate, daltonize
from PIL import Image
import settings

im = Image.open('colordeficiency-images/0010000000.png')
img_in = im.copy()
size = 256,256
img_in.thumbnail(size)
#im.show()
if 1:
    for sim_type in settings.simulation_types:
        im_sim = simulate(sim_type, img_in, 'p')
        im_sim.show()
else:
    for dalt_type in settings.daltonization_types:
        print dalt_type
        if 1:#(dalt_type == 'kuhn'): 
            options = {'daltonization_type':dalt_type, 'coldef_type': 'p', 'alpha': 1.0, 'beta': 0.25}
            im_dalt = daltonize(img_in,options)
            im_dalt.show()