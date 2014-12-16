'''
Created on 25. juni 2014

@author: joschua
'''
# wech
# whole file can be deleted

# from test import simType2Int, getStatsFromFilename, setStatsToFilename
# #from colordeficiency_old import simulate
# from PIL import Image
# import sys
# import os
# import settings
# from test import getAllXXXInPath
# from analysis import extracVisualSearchData
# from colordeficiency_old import simulate, daltonize

# #wech
# def simulate_pics(images,options):
#     """
#     Same as in colordeficiency_old.py, is not actually called up. 
#     """
#     if options.has_key('path_out'):
#         path_out = options['path_out']
#     else:
#         path_out = ""
#     
#     if not options.has_key('simulation_types'):
#         print "Caution: No daltonization_type chosen. Default value will be chosen: " + ", ".join(settings.simulation_types)
#         options['simulation_types'] = settings.simulation_types
#     
#     if not options.has_key('coldef_types'):
#         print 'Caution: No coldef_type chosen. Default value will be chosen: '+", ".join(settings.coldef_types)
#         options['coldef_types'] = settings.coldef_types
#     
#     for img in images:
#         path_in_tmp, basename_tmp = os.path.split(img)
#         file_name_in_tmp,file_ext = os.path.splitext(basename_tmp)
#         dict_in = getStatsFromFilename(file_name_in_tmp)
#         
#         if not path_out:
#             path_out = path_in_tmp
#         sys.stdout.write("Computing "+str(file_name_in_tmp))
#         
#         for simulation_type in options['simulation_types']:
#             sys.stdout.write( "..." + simulation_type)
#             for coldef_type in options['coldef_types']:
#                 sys.stdout.write("."+coldef_type)
#                 
#                 if (bool(dict_in['sim'])):
#                     sys.stdout.write('.Cannot simulate already simulated images')
#                 
#                 elif (bool(dict_in['dalt']) and not (int(dict_in['coldef_type'])) == settings.colDef2ID[coldef_type]):
#                     sys.stdout.write('.Color deficiency type of daltonization and simulation has to be the same')
#                 else:
#                     
#                     img_in = Image.open(img)
#                     img_sim =simulate(simulation_type, img_in, coldef_type)
#                     
#                     file_name_out_tmp = setStatsToFilename(
#                                                        dict_in['img_id'],
#                                                        dict_in['dalt'],
#                                                        dict_in['dalt_id'],
#                                                        True,
#                                                        simType2Int(simulation_type),
#                                                        settings.colDef2ID[coldef_type]
#                                                        )
#                 
#                     file_path_out_tmp = os.path.join(path_out,file_name_out_tmp)
#                 #print file_path_out_tmp
#         sys.stdout.write("\n")             
                
#images = ['images/IMT6131/0341000002.png','images/IMT6131/0430000000.png','images/IMT6131/0810000000.png']
#options = {'coldef_types':['p','d'], 'simulation_types': ['brettel']}
#simulate_pics(images,options)         
            