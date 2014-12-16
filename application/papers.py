'''
Created on 22. nov. 2014

@author: joschua
'''

from PIL import Image
import os
from colordeficiency.colordeficiency import simulate, daltonize
from test import setStatsToFilename
import settings
import pandas
from colordeficiency.analysis import vsplots67thru70, s2mplots29and30,organizeArray,writeMetaData,analyzeSample2MatchData,analyzeVisualSearchData, vsplots71thru74  
import operator

def EIXX2015_SChaRDa():
    path_out = "/Users/thomas/Dropbox/01_NZT/01_PhD/02_Conferences/EI-2015_Color-Imaging-XX/01_Artikler/01_Simple-daltonization-methods/images/00_top/"
    dalt_types = ['yoshi-alsam','yoshi-simone']
    sim_id = 'brettel'
    size = 250,250
    
    if 0:
        path_out = "/Users/thomas/Dropbox/01_NZT/01_PhD/02_Conferences/EI-2015_Color-Imaging-XX/01_Artikler/01_Simple-daltonization-methods/images"
        image_ids = [34,40,43,81,95] # Image IDs in reference to the color deficiency data base
        coldef_types = ['p','d']
        dalt_types = ['yoshi-alsam','yoshi-simone']
        sim_id = 'brettel'
        
        alphas = [.9]
        
        for image_id in image_ids:
            filename_in = str(image_id).zfill(3) +"0000000.png"
            img_in = Image.open(os.path.join('../colordeficiency-images/',filename_in))
            img_in.thumbnail(size)
            
            for coldef_type in coldef_types:
                img_in_sim = simulate(sim_id, img_in, coldef_type)
                filename_in_sim = setStatsToFilename(image_id, 0, 0, 1, settings.sim2ID[sim_id], settings.colDef2ID[coldef_type]);
                #img_in_sim.save(os.path.join(path_out,filename_in_sim))
                
                for alpha in alphas:
                    beta = .25
                    for dalt_type in dalt_types:
                        img_out = daltonize(img_in, {'coldef_type': coldef_type, 'daltonization_type': dalt_type, 'alpha': alpha, 'beta': beta })
                        filename_out = setStatsToFilename(image_id, 1, settings.dalt2ID[dalt_type], 0, 0, settings.colDef2ID[coldef_type]);
                        img_out.save(os.path.join(path_out,'alpha-beta',str(alpha)+'-'+str(beta),filename_out))
                        
                        img_out_sim = simulate(sim_id, img_out, coldef_type)
                        filename_out_sim = setStatsToFilename(image_id, 1, settings.dalt2ID[dalt_type], 1, settings.sim2ID[sim_id], settings.colDef2ID[coldef_type]);
                        img_out_sim.save(os.path.join(path_out,'alpha-beta',str(alpha)+'-'+str(beta),filename_out_sim))
    elif 0:
        image_ids = [34,43] # Image IDs in reference to the color deficiency data base
        coldef_types = ['d']
        
        for image_id in image_ids:
            filename_in = str(image_id).zfill(3) +"0000000.png"
            img_in = Image.open(os.path.join('../colordeficiency-images/',filename_in))
            img_in.save(os.path.join(path_out,str(image_id).zfill(3)+'-original.png'))
            
            for coldef_type in coldef_types:
                img_in_sim = simulate(sim_id, img_in, coldef_type)
                filename_in_sim = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-original-'+sim_id+'-simulation.png'#setStatsToFilename(image_id, 0, 0, 1, settings.sim2ID[sim_id], settings.colDef2ID[coldef_type]);
                img_in_sim.save(os.path.join(path_out,filename_in_sim))
                
                for dalt_type in dalt_types:
                    img_out = daltonize(img_in, {'coldef_type': coldef_type, 'daltonization_type': dalt_type})
                    filename_out = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-'+dalt_type+'-daltonization.png'#setStatsToFilename(image_id, 1, settings.dalt2ID[dalt_type], 0, 0, settings.colDef2ID[coldef_type]);
                    img_out.save(os.path.join(path_out,filename_out))
                    
                    img_out_sim = simulate(sim_id, img_out, coldef_type)
                    filename_out_sim = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-'+dalt_type+'-daltonization-'+sim_id+'-simulation.png'#setStatsToFilename(image_id, 1, settings.dalt2ID[dalt_type], 0, 0, settings.colDef2ID[coldef_type]);
                    img_out_sim.save(os.path.join(path_out,filename_out_sim))
        
        image_ids = [81,95] # Image IDs in reference to the color deficiency data base
        coldef_types = ['p']
        
        for image_id in image_ids:
            filename_in = str(image_id).zfill(3) +"0000000.png"
            img_in = Image.open(os.path.join('../colordeficiency-images/',filename_in))
            img_in.save(os.path.join(path_out,str(image_id).zfill(3)+'-original.png'))
            
            for coldef_type in coldef_types:
                img_in_sim = simulate(sim_id, img_in, coldef_type)
                filename_in_sim = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-original-'+sim_id+'-simulation.png'#setStatsToFilename(image_id, 0, 0, 1, settings.sim2ID[sim_id], settings.colDef2ID[coldef_type]);
                img_in_sim.save(os.path.join(path_out,filename_in_sim))
                
                for dalt_type in dalt_types:
                    img_out = daltonize(img_in, {'coldef_type': coldef_type, 'daltonization_type': dalt_type})
                    filename_out = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-'+dalt_type+'-daltonization.png'#setStatsToFilename(image_id, 1, settings.dalt2ID[dalt_type], 0, 0, settings.colDef2ID[coldef_type]);
                    img_out.save(os.path.join(path_out,filename_out))
                    
                    img_out_sim = simulate(sim_id, img_out, coldef_type)
                    filename_out_sim = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-'+dalt_type+'-daltonization-'+sim_id+'-simulation.png'#setStatsToFilename(image_id, 1, settings.dalt2ID[dalt_type], 0, 0, settings.colDef2ID[coldef_type]);
                    img_out_sim.save(os.path.join(path_out,filename_out_sim))
    else:
        image_ids = [34] # Image IDs in reference to the color deficiency data base
        coldef_types = ['d']
        
        alphas = [.0,.33,.67]
        
        for image_id in image_ids:
            filename_in = str(image_id).zfill(3) +"0000000.png"
            img_in = Image.open(os.path.join('../colordeficiency-images/',filename_in))
            img_in.save(os.path.join(path_out,str(image_id).zfill(3)+'-original.png'))
            
            for coldef_type in coldef_types:
                img_in_sim = simulate(sim_id, img_in, coldef_type)
                filename_in_sim = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-original-'+sim_id+'-simulation.png'#setStatsToFilename(image_id, 0, 0, 1, settings.sim2ID[sim_id], settings.colDef2ID[coldef_type]);
                img_in_sim.save(os.path.join(path_out,filename_in_sim))
                
                for alpha in alphas:
                    beta = 1-alpha
                    for dalt_type in dalt_types:
                        img_out = daltonize(img_in, {'coldef_type': coldef_type, 'daltonization_type': dalt_type, 'alpha': alpha, 'beta': beta })
                        filename_out = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-'+dalt_type+'-daltonization-alpha-'+str(int(alpha*100))+'-beta-'+str(int(beta*100))+'.png'#setStatsToFilename(image_id, 1, settings.dalt2ID[dalt_type], 0, 0, settings.colDef2ID[coldef_type]);
                        img_out.save(os.path.join(path_out,filename_out))
                        
                        img_out_sim = simulate(sim_id, img_out, coldef_type)
                        filename_out_sim = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-'+dalt_type+'-daltonization-alpha-'+str(int(alpha*100))+'-beta-'+str(int(beta*100))+'-'+sim_id+'-simulation.png'#setStatsToFilename(image_id, 1, settings.dalt2ID[dalt_type], 0, 0, settings.colDef2ID[coldef_type]);
                        img_out_sim.save(os.path.join(path_out,filename_out_sim))
        
        image_ids = [95] # Image IDs in reference to the color deficiency data base
        coldef_types = ['p']
        
        alpha = .9
        betas = [.33,.67,1.]
        
        for image_id in image_ids:
            filename_in = str(image_id).zfill(3) +"0000000.png"
            img_in = Image.open(os.path.join('../colordeficiency-images/',filename_in))
            img_in.save(os.path.join(path_out,str(image_id).zfill(3)+'-original.png'))
            
            for coldef_type in coldef_types:
                img_in_sim = simulate(sim_id, img_in, coldef_type)
                filename_in_sim = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-original-'+sim_id+'-simulation.png'#setStatsToFilename(image_id, 0, 0, 1, settings.sim2ID[sim_id], settings.colDef2ID[coldef_type]);
                img_in_sim.save(os.path.join(path_out,filename_in_sim))
                
                for beta in betas:
                    for dalt_type in dalt_types:
                        img_out = daltonize(img_in, {'coldef_type': coldef_type, 'daltonization_type': dalt_type, 'alpha': alpha, 'beta': beta })
                        filename_out = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-'+dalt_type+'-daltonization-alpha-'+str(int(alpha*100))+'-beta-'+str(int(beta*100))+'.png'#setStatsToFilename(image_id, 1, settings.dalt2ID[dalt_type], 0, 0, settings.colDef2ID[coldef_type]);
                        img_out.save(os.path.join(path_out,filename_out))
                        
                        img_out_sim = simulate(sim_id, img_out, coldef_type)
                        filename_out_sim = str(image_id).zfill(3)+'-'+settings.id2ColDef[settings.colDef2ID[coldef_type]]+'-'+dalt_type+'-daltonization-alpha-'+str(int(alpha*100))+'-beta-'+str(int(beta*100))+'-'+sim_id+'-simulation.png'#setStatsToFilename(image_id, 1, settings.dalt2ID[dalt_type], 0, 0, settings.colDef2ID[coldef_type]);
                        img_out_sim.save(os.path.join(path_out,filename_out_sim))

def EIXX2015_SaMSEM_ViSDEM():
    path_img = "/Users/thomas/Dropbox/01_NZT/01_PhD/02_Conferences/EI-2015_Color-Imaging-XX/01_Artikler/02_Evaluation-methods/images/"
    path_data = "/Users/thomas/Dropbox/01_NZT/01_PhD/02_Conferences/EI-2015_Color-Imaging-XX/01_Artikler/02_Evaluation-methods/data/"
    read_data = 0
    plot_data = 1
    meta_data = 1
    y_lim = [.65,1.]
    
    ##################################################
    #####
    #####     Read SaMSEM and VisDEM data
    #####
    ##################################################
    
    if read_data:
        
        analyzeSample2MatchData({'path_in': os.path.join(path_data,'SaMSEM'), 'path_out': path_data})
        analyzeVisualSearchData({'path_in': os.path.join(path_data,'ViSDEM'), 'path_out': path_data})
    
    if plot_data:
    
        ##################################################
        #####
        #####     Plots for SaMSEM
        #####
        ##################################################
        
        sample2MatchDataPath = os.path.join(path_data,'samsem-data.csv')
        sample2match_data = pandas.read_csv(sample2MatchDataPath,index_col=False,sep=';')
        
        # Take out dummy "daltonization"
        whatArr_tmp = [['sim_id',operator.ne,99]];howArr_tmp=[]
        sample2match_data = organizeArray(sample2match_data,whatArr_tmp,howArr_tmp)
        
        #print sample2match_data
        
        
        dict = {'result_id': '', 'coldef_type': 1, 'obs_title': '', 'filename': 'figure-3ab-samsem-protanopia', 'y_lim':y_lim }; 
        s2mplots29and30(sample2match_data, path_img, dict) #Res#29
        dict = {'result_id': '', 'coldef_type': 2, 'obs_title': '', 'filename': 'figure-3cd-samsem-deuteranopia', 'y_lim':y_lim }; 
        s2mplots29and30(sample2match_data, path_img, dict) #Res#30
        
        if meta_data:
            file_path = os.path.join(path_data,'samsem-meta')
            writeMetaData(sample2match_data, {'exp_type': "samsem", 'path_out':file_path})    
        
        print
        
        ##################################################
        #####
        #####     Plots for ViSDEM
        #####
        ##################################################
        
        visualSearchDataPath = os.path.join(path_data,'visdem-data.csv')
        visual_search_data = pandas.read_csv(visualSearchDataPath,index_col=False,header=0,sep=';')
        
        # Excluding "dummy" simulation
        whatArr_tmp = [['dalt_id',operator.ne,99]];howArr_tmp=[]
        visual_search_data = organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        #print visual_search_data
        
        # Excluding Group B from Set 2
        indices_all_obs_groupB_set2 = (visual_search_data['obsGroup']=='B')&(visual_search_data['set_id']==2)
        visual_search_data[indices_all_obs_groupB_set2] = visual_search_data[(indices_all_obs_groupB_set2==False)]
        #visual_search_data[indices_all_obs_groupB_set2].is_correct #= visual_search_data[indices_all_obs_groupB_set2].is_correct#==False)
        
        #visual_search_data.to_csv(os.path.join(path_data,'visdem-data-bruno.csv'),sep=';')
        
        # ViSDEM Res#67+70
        set_ids = [1,2,3,4,5,6,8] # Sets that are difficult for protanopes and deuteranopes
        dict = {'result_id': '', 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': '', 'filename': 'figure-4ab-visdem-normal-observers', 'sets': set_ids, 'y_lim':y_lim }; 
        vsplots67thru70(visual_search_data, path_img, dict) #Res#67
        dict = {'result_id': '', 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': '', 'filename': 'figure-4cd-visdem-deutan-observers', 'sets': set_ids, 'y_lim':y_lim }; 
        vsplots67thru70(visual_search_data, path_img, dict) #Res#70
        
        
        
        # ViSDEM Res#71+74
        #dict = {'result_id': '', 'obs_coldef_type': str(0), 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All sets for all norm.sight.obs.', 'filename': 'visdem-normal-observers-test', 'sets': set_ids }; 
        #vsplots71thru74(visual_search_data, path_img, dict) #Res#67
        #dict = {'result_id': '', 'obs_coldef_type': str(2), 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All sets for all deut.obs.', 'filename': 'visdem-deutan-observers-test', 'sets': set_ids }; 
        #vsplots71thru74(visual_search_data, path_img, dict) #Res#70
        
        if meta_data:
            file_path = os.path.join(path_data,'visdem-meta')
            writeMetaData(visual_search_data, {'exp_type': "visdem", 'path_out':file_path})
         
#print "hier simmer"           
EIXX2015_SaMSEM_ViSDEM()        
        