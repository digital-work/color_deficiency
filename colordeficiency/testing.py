from pylab import *
import os

from colordeficiency import *

def computeTestName(dict):
    #print dict
    test_name = ''
    if dict['multi_scaling']:
        test_name += 'multi-scaling_'
    else:
        test_name += 'simple_'
        
    if dict['img_PCA']:
        test_name += 'image_PCA_'
    else:
        test_name += 'gradient-PCA_'
        
    if dict['constant_lightness']:
        test_name += 'constant-lightness_'
    else:
        test_name += 'neutral-gray_'
    
    if dict['chi_computations']==1:
        test_name += 'chi1_'
    elif dict['chi_computations']==2:
        test_name += 'chi2_'
        
    if dict['chi_red']:
        test_name += 'chired_'
    else:
        test_name += 'chigreen_'
        
    if dict['ed_orthogonalization']:
        test_name += 'ed90_'
    else:
        test_name += 'edXX_'
        
    if dict['optimization']==1:
        test_name += 'poisson-optimization_'
    elif dict['optimization']==2:
        test_name += 'tv-optimization_'
        
    if dict['boundary']==3:
        test_name += 'nobound_'  
    else:
        test_name += str(int(dict['boundary']))+'bound_'
        
    test_name += dict['simulation_type']+'-'
    test_name += dict['coldef_type']
    return test_name

def tvdalt_engineeredgradient():
    
    im_names = []
    #im_names.append('0340000000')
    im_names.append('berries2')
    
    #im_names.append('berries2-inverted')
    #im_names.append('bananas1')
    #im_names.append('berries1')
    #im_names.append('0030000000')
    #im_names.append('berries2-gradient')
    
    #images = [im1,im2]
    dict_list = []
    for im_name in im_names:
        im = imread(os.path.join('../colordeficiency-images/',im_name+'.png'))
        im0 = im.copy()
        
        figure(0); ion()
        data = imshow(im0, vmin=0, vmax=1)
        title("Daltonised"); show(); draw()
        
        # Show different unit vectors
        simulation_type = 'brettel'
        coldef_type = 'p'
        dict_1 = {  'constant_lightness': 1, # 1 - constant lightness, 0 - neutral gray'multi_scaling': 0,
                    'multi_scaling': 0,
                    'img_PCA': 1,
                    'chi_computations': 1, 
                    'ed_orthogonalization': 0,
                    'chi_red': 1, # 1 - change red color, 0 - change green colors
                    'optimization': 1, # 1 - poisson, 2 - total variation
                    'boundary': 3,
                    'interp': "cubic",
                    'im0': im0,
                    'data': data,
                    'dt': .25,
                    #'data2': data2,
                    'cutoff': .01,
                    'is_simulated': 0,
                    'simulation_type': simulation_type,
                    'coldef_type': coldef_type,
                    'im_name': im_name,
                    'path_out': os.path.join('/Users/thomas/Desktop/different-unit-vectors/ed'),
                    #'im_out': os.path.join('/Users/thomas/Desktop/different-unit-vectors',im_name)
                    }
        test_name = computeTestName(dict_1)
        dict_1['test_name'] = test_name
        #dict_list.append(dict_1)
        
        dict_2 = dict_1.copy()
        dict_2.update({'img_PCA': 0 })
        test_name = computeTestName(dict_2)
        dict_2['test_name'] = test_name
        #dict_list.append(dict_2)
        
        dict_3 = dict_1.copy()
        dict_3.update({'ed_orthogonalization': 1})
        test_name = computeTestName(dict_3)
        dict_3['test_name'] = test_name
        #dict_list.append(dict_3)
        
        dict_4 = dict_1.copy()
        dict_4.update({'ed_orthogonalization': 0})
        test_name = computeTestName(dict_4)
        dict_4['test_name'] = test_name
        #dict_list.append(dict_4)
        
        dict_5 = dict_1.copy()
        dict_5.update({'path_out': os.path.join('/Users/thomas/Desktop/different-unit-vectors/el')})
        test_name = computeTestName(dict_5)
        dict_5['test_name'] = test_name
        #dict_list.append(dict_5)
        
        dict_6 = dict_5.copy()
        dict_6.update({'constant_lightness': 1})
        test_name = computeTestName(dict_6)
        dict_6['test_name'] = test_name
        #dict_list.append(dict_6)
        
        dict_7 = dict_1.copy()
        dict_7.update({'path_out': os.path.join('/Users/thomas/Desktop/different-unit-vectors/ec')})
        test_name = computeTestName(dict_7)
        dict_7['test_name'] = test_name
        #dict_list.append(dict_7)
        
        dict_8 = dict_7.copy()
        dict_8.update({'chi_red': 0})
        test_name = computeTestName(dict_8)
        dict_8['test_name'] = test_name
        #dict_list.append(dict_8)
        
        dict_9 = dict_1.copy()
        dict_9.update({'path_out': os.path.join('/Users/thomas/Desktop/chi-computations')})
        test_name = computeTestName(dict_9)
        dict_9['test_name'] = test_name
        #dict_list.append(dict_9)
        
        dict_10 = dict_9.copy()
        dict_10.update({'chi_computations': 2})
        test_name = computeTestName(dict_10)
        dict_10['test_name'] = test_name
        #dict_list.append(dict_10)

        dict_11 = dict_1.copy()
        dict_11.update({'path_out': os.path.join('/Users/thomas/Desktop/multi-scaling')})
        test_name = computeTestName(dict_11)
        dict_11['test_name'] = test_name
        #dict_list.append(dict_11)
        
        dict_12 = dict_11.copy()
        dict_12.update({'multi_scaling': 1})
        test_name = computeTestName(dict_12)
        dict_12['test_name'] = test_name
        #dict_list.append(dict_12)
        
        dict_13 = dict_1.copy()
        dict_13.update({'path_out': os.path.join('/Users/thomas/Desktop/boundaries')})
        test_name = computeTestName(dict_13)
        dict_13['test_name'] = test_name
        
        dict_14 = dict_13.copy()
        dict_14.update({'boundary': 1})
        test_name = computeTestName(dict_14)
        dict_14['test_name'] = test_name
        
        dict_15 = dict_13.copy()
        dict_15.update({'boundary': 2})
        test_name = computeTestName(dict_15)
        dict_15['test_name'] = test_name
        
        dict_16 = dict_13.copy()
        dict_16.update({'boundary': 3})
        test_name = computeTestName(dict_16)
        dict_16['test_name'] = test_name
        
        dict_17 = dict_1.copy()
        dict_17.update({'path_out': os.path.join('/Users/thomas/Desktop/optimization')})
        test_name = computeTestName(dict_17)
        dict_17['test_name'] = test_name
        
        dict_18 = dict_17.copy()
        dict_18.update({'optimization': 2})
        test_name = computeTestName(dict_18)
        dict_18['test_name'] = test_name
        
        #dict_list.append(dict_13)
        #dict_list.append(dict_14)
        #dict_list.append(dict_15)
        #dict_list.append(dict_16)
        #dict_list.append(dict_17)
        dict_list.append(dict_18)
    
        for dict_i in dict_list:
            im_dalt = daltonization_yoshi_042016(im,dict_i)
            #im2_dalt = multiscaling(im2,daltonization_yoshi_gradient,dict_i)
            
            if not os.path.isdir(dict_i['path_out']):
                os.makedirs(dict_i['path_out'])
                print "Caution: Created directory ["+dict_i['path_out']+']'
            
            if 1:
                print "Saving '"+dict_i['im_name']+"' image: "+dict_i['test_name']
                imsave(os.path.join(dict_i['path_out'],dict_i['im_name']+'_'+dict_i['test_name']+'_orig'+'.png'), im)
                imsave(os.path.join(dict_i['path_out'],dict_i['im_name']+'_'+dict_i['test_name']+'_orig-sim'+'.png'), simulate(simulation_type,im,coldef_type))
                imsave(os.path.join(dict_i['path_out'],dict_i['im_name']+'_'+dict_i['test_name']+'_dalt'+'.png'), im_dalt)
                imsave(os.path.join(dict_i['path_out'],dict_i['im_name']+'_'+dict_i['test_name']+'_dalt-sim'+'.png'), simulate(simulation_type,im_dalt,coldef_type))  
        
    print "Ferdig"
    
tvdalt_engineeredgradient()