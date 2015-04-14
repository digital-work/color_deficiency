'''
Created on 26. aug. 2014

@author: joschua
'''

import copy, json, math, matplotlib.pyplot as plt, numpy, operator, os, pandas, scipy, sys
#import matplotlib.pyplot as plt
#import pandas
#import os
#import sys
#import operator
#import json
#import math
#from colordeficiency import colordeficiency_tools
#from colordeficiency_tools import getAllXXXinPath, getStatsFromFilenamet

from analysis_tools import writePandastoLatex, makePearsonChi2Contingency, makePearsonChi2Contingency2x2Test, plotQQPlot, plotResidualPlots, plotHistogram, plotAccuracyGraphs, plotCIAverageGraphs, plotRTGraphs, getSetFromScene, getAccuracy, getCIAverage, organizeArray, extractDataFromPsychoPyXLSX
from colordeficiency import settings
from colordeficiency_tools import getAllXXXinPath, getStatsFromFilename
from scipy import stats
#print colordeficiency.settings.coldef_toolbox_path

def extractExperimentData(dataSheet,dataArray=pandas.DataFrame(),testArray=pandas.DataFrame()):
    """
    This function takes the url an xlsx file containing the result data from a sample-to-match experiment, extracts the data and returns the contained information as numpy array.
    Is this the same as analyze SaMSEMData? What does it do?
    """
    
    # Get all the sheet names for the excel file
    
    sheet_names = pandas.ExcelFile(dataSheet).sheet_names
    # Create empty array for colordeficiency_tools and relevatn data. Create empty dictionary for the extra data
    if dataArray.empty: dataArray = pandas.DataFrame();
    if testArray.empty: testArray = pandas.DataFrame();
    extraDataDict = {}
    
    # Read file sheet by sheet
    for sheet_name in sheet_names:
        if not "~" in sheet_name:
            if ("roundTrials" in sheet_name) or ('practSets' in sheet_name) or ('sets' in sheet_name): pass # Ignore data about the order of set and samples for now
            else:
                # Read excel and extract data
                excel_sheet = pandas.read_excel(dataSheet, sheet_name)
                array_tmp, extraDataDict_tmp = extractDataFromPsychoPyXLSX(excel_sheet)
                #Update dictionary containing extra data
                extraDataDict.update(extraDataDict_tmp)
                # Append colordeficiency_tools or relevant data to respective data array
                if ("testTrials" in sheet_name) or ("practTrials" in sheet_name): testArray = pandas.concat([array_tmp, testArray]) if not testArray.empty else array_tmp
                else: dataArray = pandas.concat([array_tmp,dataArray]) if not dataArray.empty else array_tmp
    
    dataArray = dataArray.reset_index()
    testArray = testArray.reset_index()
            
    return dataArray, testArray, extraDataDict
    

def analyzeSaMSEMData(dict):
    """
    This functions analyzes the data from the sample-2-match experiment as can be found in the path.
    Before: analyzeSample2MatchData
    """
    
    if dict.has_key('path_in'):
        path_in = dict['path_in']
    else:
        print "Caution: No path for input folder containing the data has been defined. Please define path to folder by dict['path_in']=path_in"    
        return
    
    path_out_default = '../colordeficiency-data/'    
    if dict.has_key('path_out'):
        path_out = dict['path_out']
    else:
        print "Caution: No path for output folder where the data should be stored has been defined. Using default output path instead: "+str(path_out_default)
        path_out = path_out_default
    
    # 0. Step: Get all the relevant information, i.e. obs_col_defs etc.
    observer_ids = "../colordeficiency-data/observer_ids.csv"
    obs_ids_sheet = pandas.read_csv(observer_ids,sep=";")
    
    # 1. Step: Read all the XLSX data in the path
    ext = 'xlsx'; xlsx_files = getAllXXXinPath(path_in,ext)
    dataArray = pandas.DataFrame()
    i=1
    for xlsx_file in xlsx_files:
        if not '~' in xlsx_file:
            sys.stdout.write(xlsx_file)
            dataArray_tmp, testArray, extraDataDict = extractExperimentData(os.path.join(path_in,xlsx_file))
            
            experiment_type = extraDataDict['expName'] if extraDataDict.has_key('expName') else 'none'
            if experiment_type == "sample-2-match":
                newDataArray = dataArray_tmp[['sim_id','coldef_type','resp.corr_raw','resp.rt_raw','origFile']]
            
            if extraDataDict.has_key('0. Participant ID'):
                obsID = int(extraDataDict['0. Participant ID'])
                newDataArray['observer_id'] = obsID
                obs_coldef_type = obs_ids_sheet.loc[obs_ids_sheet['observer_id']==obsID,['observer_coldef_type']]
                newDataArray['observer_coldef_type'] = int(obs_coldef_type['observer_coldef_type'])
            
            if extraDataDict.has_key("2. Session"):
                sessionID = int(extraDataDict['2. Session'])
                newDataArray['session_id'] = sessionID
            
            dataArray = pandas.concat([dataArray, newDataArray])
            sys.stdout.write(' . ')
            if (i%5)==0: sys.stdout.write('\n')
            i+=1
    sys.stdout.write('\n')
    
    dataArray = dataArray.reset_index()
    
    # 2.Step: Adapt values to programstandards
    for item in settings.colDefLong2ID:
        dataArray.loc[dataArray['coldef_type'] == item, ['coldef_type']] = settings.colDefLong2ID[item]
    
    if experiment_type == "sample-2-match":
        for item in settings.sim2ID:
            dataArray.loc[dataArray['sim_id'] == item, ['sim_id']] = settings.sim2ID[item]
        
        dataArray = dataArray.rename(columns={'sim_id': 'sim_id',
                              'coldef_type': 'coldef_type',
                              'resp.corr_raw': 'is_correct',
                              'resp.rt_raw': 'resp_time',
                              'origFile': 'filepath'})
            
        for index, row in dataArray.iterrows():
            path_tmp = row['filepath']
            filename = os.path.basename(path_tmp).split('.')[0]
            dict_tmp = getStatsFromFilename(filename)
            imgID_tmp = int(dict_tmp['img_id'])
            dataArray.at[index,'image_id'] = int(imgID_tmp)
        dataArray.image_id = dataArray.image_id.astype(int)
        
        dataArray = dataArray[['image_id','sim_id','coldef_type','is_correct','resp_time','observer_id','observer_coldef_type','session_id','filepath']]
    elif experiment_type == "visual-search":
        pass
    dataArray.is_correct = dataArray.is_correct.astype(bool) 
    
    # 3. Saving data to file
    try:
        sys.stdout.write("Starting to save ... ")
        if experiment_type == "sample-2-match":
            dataArray.to_csv(os.path.join(path_out,'samsem-data.csv'),sep=";")
            sys.stdout.write("Success: Sample-to-match data successfully saved in '"+str(path_out)+"'.\n")
            print
        elif experiment_type == "visual-search":
            dataArray.to_csv(os.path.join(path_out,'visdem-data.csv'),sep=";")
            sys.stdout.write("Visual-search data successfully saved.")
        else:
            sys.stdout.write("Caution: No data saved.")
    except Exception as e:
        print e 

def analyzeViSDEMData(dict):
    """
    This function analyzes the data from the visual searach experiment as can be found in the path.
    Before: analyseVisualSearchData
    
    """
    
    if dict.has_key('path_in'):
        path_in = dict['path_in']
    else:
        print "Caution: No path for input folder containing the data has been defined. Please define path to folder by dict['path_in']=path_in"    
        return
    
    path_out_default = '../colordeficiency-data/'    
    if dict.has_key('path_out'):
        path_out = dict['path_out']
    else:
        print "Caution: No path for output folder where the data should be stored has been defined. Using default output path instead: "+str(path_out_default)
        path_out = path_out_default
        
    # 0. Step: Get all the relevant information, i.e. scene_ids, obs_col_defs etc.
    visualsearch_ids = "../colordeficiency-data/visualsearccolordeficiency-colordeficiency-data/.csv"
    vs_ids_sheet = pandas.read_csv(visualsearch_ids,sep=';')    
    
    observer_ids = "../colordeficiency-data/observer_ids.csv"
    obs_ids_sheet = pandas.read_csv(observer_ids,sep=';')
    
    
    # 1. Step: Read all the XLSX data in the path
    ext = 'xlsx'; xlsx_files = getAllXXXinPath(path_in,ext)
    dataArray = pandas.DataFrame()
    i=1
    for xlsx_file in xlsx_files:
        sys.stdout.write(xlsx_file)
        dataArray_tmp, testArray, extraDataDict = extractExperimentData(os.path.join(path_in,xlsx_file))
        
        newDataArray = dataArray_tmp[['dalt_id','coldef_type','resp.corr_raw','resp.rt_raw','stimFile']]
        
        if extraDataDict.has_key("2. Session"):
            sessionID = int(extraDataDict['2. Session'])
        newDataArray['session_id'] = sessionID
        
        if extraDataDict.has_key('group'):
            obsGroup = str(extraDataDict['group'])
        newDataArray['obsGroup'] = obsGroup
            
        if extraDataDict.has_key('0. Participant ID'):
            obsID = int(extraDataDict['0. Participant ID'])
            
        newDataArray['observer_id'] = obsID
        obs_coldef_type = obs_ids_sheet.loc[obs_ids_sheet['observer_id']==obsID,['observer_coldef_type']]
        newDataArray['observer_coldef_type'] = int(obs_coldef_type['observer_coldef_type'])
        
        dataArray = pandas.concat([dataArray, newDataArray])
        sys.stdout.write(' . ')
        if (i%5)==0: sys.stdout.write('\n')
        i+=1
    sys.stdout.write('\n')
    
    # 2. Step: Adapt values to programstandards
    for item in settings.colDefLong2ID:
        dataArray.loc[dataArray['coldef_type'] == item, ['coldef_type']] = settings.colDefLong2ID[item]
    
    for item in settings.dalt2ID:
        dataArray.loc[dataArray['dalt_id'] == item, ['dalt_id']] = settings.dalt2ID[item]
        
    dataArray = dataArray.rename(columns={'dalt_id': 'dalt_id',
                              'coldef_type': 'coldef_type',
                              'resp.corr_raw': 'is_correct',
                              'resp.rt_raw': 'resp_time',
                              'stimFile': 'filepath'})
    dataArray = dataArray.reset_index()
    
    # Adding set_id, scene_id and version_id to each file
    for index, row in dataArray.iterrows():
        path_tmp = row['filepath']
        filename = os.path.basename(path_tmp).split('.')[0]
        dict_tmp = getStatsFromFilename(filename)
        imgID_tmp = int(dict_tmp['img_id'])
        
        tempVSDataArray = vs_ids_sheet.loc[vs_ids_sheet['image_id']==imgID_tmp,['set_id','scene_id','version_id']]
        
        dataArray.at[index,'image_id'] = imgID_tmp
        dataArray.ix[index,'set_id'] = int(tempVSDataArray['set_id'])
        dataArray.ix[index,'scene_id'] = int(tempVSDataArray['scene_id'])
        dataArray.ix[index,'version_id'] = int(tempVSDataArray['version_id'])
        #dataArray.ix[index,'observer_coldef_type'] = obs_coldef_type_tmp
    dataArray.image_id = dataArray.image_id.astype(int)
    dataArray.set_id = dataArray.set_id.astype(int)
    dataArray.scene_id = dataArray.scene_id.astype(int)
    dataArray.version_id = dataArray.version_id.astype(int)
    
    dataArray = dataArray[['image_id','set_id','scene_id','version_id','dalt_id','coldef_type','is_correct','resp_time','observer_id','observer_coldef_type','session_id','filepath','obsGroup']]
    
    # 3. Saving data to file
    try:
        dataArray.to_csv(os.path.join(path_out, 'visdem-data.csv'),sep=";")
        sys.stdout.write("Success: ViSDEM data successfully saved in '"+str(path_out)+"'.\n")
    except Exception as e:
        print e  

def writeMetaDataOfExperiments(experimentData,path,dict):
    """
    This function returns the meta data of a SaMSEM or ViSDEM experiment like for example number of sessions, sets, simulations or daltonizations used etc.
    The information is stored inside a text file that is saved at the path defined by path_out.
    Input: experimentData:    * A pandas array with either the SaMSEM or ViSDEM data.
           dict:              * A dictionary containing all options for the computation of the meta data. 
                              Requiree are: ** exp_type: Whether the experiment was SaMSEM or ViSDEM.
                              Optional are: ** path_out: Path to the folder, where the text file containing the meta data should be stored.
    """
    
    if dict.has_key('exp_type'):
        exp_type = dict['exp_type']
    else:
        print "Error: No experiment type has been chosen. Choose either ViSDEM or SaMSEM."
        return

    path_out_default = '../colordeficiency-data/'
    if dict.has_key('path_out'):
        path_out = dict['path_out']
    else:
        print "Caution: No output path has been chosen for the meta data file. Using default value for output path instead: '"+str(path_out_default)+"'."
        path_out = os.path.join(path_out_default,'metadata-generic')
    
    if dict.has_key('filename'):
        filename = dict['filename']
    else:
        filename = 'meta-data'
    path_out = os.path.join(path, filename)
        
    if exp_type == "visdem":
        f = open(path_out,'w+')
        f.write("ViSDEM meta data\n")
        sessions = sorted(set(experimentData['session_id']))
        f.write('... # of sessions: '+str(len(sessions))+' ; '+str(sessions)+'\n')
        sets = sorted(set(experimentData['set_id']))
        f.write('... # of sets: '+str(len(sets))+' ; '+str(sets)+'\n')
        daltonizations = sorted(set(experimentData['dalt_id']))
        f.write('... # of daltonization ids: '+str(len(daltonizations))+' ; '+str(daltonizations)+'\n')
        observers = sorted(set(experimentData['observer_id']))
        f.write("... # of observers: " +str(len(observers))+' ; '+str(observers)+'\n')
        observers_norm = sorted(set(experimentData[experimentData['observer_coldef_type']==0]['observer_id']))
        f.write("...... # of normal observers: "+str(len(observers_norm))+' ; '+str(observers_norm)+'\n')
        observers_prot = sorted(set(experimentData[experimentData['observer_coldef_type']==1]['observer_id']))
        f.write("...... # of protan observers: "+str(len(observers_prot))+' ; '+str(observers_prot)+'\n')  
        observers_deut = sorted(set(experimentData[experimentData['observer_coldef_type']==2]['observer_id']))
        f.write("...... # of deutan observers: "+str(len(observers_deut))+' ; '+str(observers_deut)+'\n')         
        f.close()
    elif exp_type == 'samsem':
        f = open(path_out,'w+')
        f.write("SaMSEM meta data\n")
        sessions = sorted(set(experimentData['session_id']))
        f.write('... # of sessions: '+str(len(sessions))+' ; '+str(sessions)+'\n')
        images = sorted(set(experimentData['image_id']))
        f.write('... # of images: '+str(len(images))+' ; '+str(images)+'\n')
        simulations = sorted(set(experimentData['sim_id']))
        f.write('... # of simulations: '+str(len(simulations))+' ; '+str(simulations)+'\n')
        observers = sorted(set(experimentData['observer_id']))
        f.write("... # of observers: " +str(len(observers))+' ; '+str(observers)+'\n')
        observers_norm = sorted(set(experimentData[experimentData['observer_coldef_type']==0]['observer_id']))
        f.write("...... # of normal observers: "+str(len(observers_norm))+' ; '+str(observers_norm)+'\n')
        observers_prot = sorted(set(experimentData[experimentData['observer_coldef_type']==1]['observer_id']))
        f.write("...... # of protan observers: "+str(len(observers_prot))+' ; '+str(observers_prot)+'\n')
        observers_deut = sorted(set(experimentData[experimentData['observer_coldef_type']==2]['observer_id']))
        f.write("...... # of deutan observers: "+str(len(observers_deut))+' ; '+str(observers_deut)+'\n')
    else:
        print "Error: No valid experiment format has been chosen. Choose either visdem or samsem."
        return

##################################################
###
###  The following functions contain the results images as defined on: 
###  https://docs.google.com/document/d/1w305_EUYiReLrQ34H-teJwsNICXvRK-cv7E6TgVfukM/
###
##################################################

###
###  SaMSEM result plots
###

def samsemPlots1thru4(samsem_data,path,dict):
    """
    Plots the SaMSEM Res#1-4, and effectively also SaMSEM Res#9-12. 
    Meaning that all images are collapsed together for different simulations and different groups of normal sighted or color deficient people.
    On the x-axis, there are the different algorithm ids. The result images are stored in (subfolders) defined by path input.
    Input: * samsem_data:    SaMSEM data that is to be analyzed
           * path:           Path to the folder, where the result images should be stored. 
           * dict:           A dictionary containing different options.
                             ** result_id:    ID of the result images according to the meeting from 18/09/2014. 
                             ** coldef_type:  The color deficiency type of the simulation being used. 
                             ** obs_operator: Color deficiency of the observer group as array in the form: ['observer_coldef_type',operator.eq,1]
    Before: s2mplots1thru4       
    """
    
    # Defining the subfolder for the results
    path_res = path
    if dict.has_key('result_id'):
        sys.stdout.write("Starting Res#"+str(dict['result_id'])+'.')
        path_res = os.path.join(path,str(dict['result_id']))    
        if not os.path.exists(path_res): os.makedirs(path_res)
        
    if dict.has_key('subfolder'):
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    # 1. Retrieving data from data set
    coldef_type = dict['coldef_type']
    #whatArr_tmp = [['coldef_type',operator.eq,coldef_type],dict['obs_operator']];howArr_tmp = []; # Only observations that use simulations of the chosen color deficiency are needed 
    #samsem_data_reduced =  organizeArray(samsem_data, whatArr_tmp,howArr_tmp)
    # HDEG: get the missing data for deuteranopes from method 3 and 99.
    
    
    boxes = []; labels = []; accuracies = {}; mean_cis = {}; order = {}; i=0
    if dict.has_key('method_ids'):
        method_ids = dict['method_ids']
    else:
        method_ids = set(samsem_data['sim_id'].values.astype(int))
        if 99 in method_ids: method_ids.remove(99)
        
    if dict.has_key('plot_types'):
        plot_types = dict['plot_types']
    else:
        plot_types = ['RT_boxplots', 'RT_means', 'ACC_CIs', 'median']
        
    method_names = []
    # 2. Retrieving data from each algorithm
    for method_id in method_ids:
        method_names.append(settings.id2Sim[method_id])
        
        if (method_id != 3) and (method_id != 99):
            whatArr_tmp = [['coldef_type',operator.eq,coldef_type],dict['obs_operator'],['sim_id',operator.eq,method_id]];howArr_tmp=[];
        else:
            whatArr_tmp = [dict['obs_operator'],['sim_id',operator.eq,method_id]];howArr_tmp=[];
        relevant_data_tmp = organizeArray(samsem_data,whatArr_tmp,howArr_tmp)   
        
        # 3. Get response time data
        alg_values = relevant_data_tmp[relevant_data_tmp['is_correct']==True]['resp_time'].values*1000; 
        if 'RT_boxplots' in plot_types:
            boxes.append(alg_values)
        labels.append(settings.id2Sim[method_id]) if alg_values.size else labels.append(str(method_id) + ' - No data'); 
            
        # 4. Get CI of RT means
        if 'RT_means' in plot_types:
            alg_mean = getCIAverage(alg_values);
            alg_mean.append(labels[i-1])
            mean_cis.update({method_id: alg_mean})
            
        # 5. Get accuracy data
        if 'ACC_CIs' in plot_types:
            alg_acc = getAccuracy(relevant_data_tmp)
            alg_acc.append(labels[i-1])
            accuracies.update({method_id: alg_acc})
            
        order.update({i:method_id})
        i += 1
    
    if 'RT_boxplots' in plot_types:
        # 6. Plot response time data
        plotRTGraphs(boxes,labels,path_res,dict,order)
    
    if 'RT_means' in plot_types:
        # 7. Plot CI means of RT data
        plotCIAverageGraphs(mean_cis,path_res,dict,order)
        
    if 'ACC_CIs' in plot_types:
        # 8. Plot accuracy data
        plotAccuracyGraphs(accuracies,path_res,dict,order)
        
    if 'median-test' in plot_types:
        # 9. Make median test
        dict.update({'filename': dict['filename']+"-RT"})
        makeMedianTest(numpy.array(boxes), path_res, method_names,dict)
    
def samsemPlots9thru12(samsem_data,path,dict):
    """
    Literally identical to samsemPlots1thru4. Check documentation there.
    """
    
    print "Starting SAMSEM_RES#9+12: Analyzing data for " + str(settings.id2ColDefLong[dict['coldef_type']]) + " simulation methods."
    samsemPlots1thru4(samsem_data,path,dict)

def samsemPlots17thru20(samsem_data, path, dict):
    """
    Plots the SaMSEM Res#17-30
    Meaning that for each algorithm individually simulating one specific color deficiency results for all images are plotted.
    On the x-axis, we have the image ids. The images are stored in (subfolder) defined by the path input.
    """
    
    path_res = path
    
    if dict.has_key('subfolder'):
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    coldef_type = dict['coldef_type']
    print "Starting SAMSEM_RES#17+20: Analyzing data of all images for " + str(settings.id2ColDefLong[dict['coldef_type']]) + " simulation methods."
    
    if dict.has_key('plot_types'):
        plot_types = dict['plot_types']
    else:
        plot_types = ['RT_boxplots', 'RT_means', 'ACC_CIs', 'median']
    
    method_ids = sorted(set(samsem_data['sim_id'].values.astype(int)))
    image_ids = sorted(set(samsem_data['image_id'].values.astype(int)))
    #print method_ids
    
    for method_id in method_ids:
        if (method_id != 3) and (method_id != 99):
            whatArr_sim = [['sim_id',operator.eq,method_id],['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq, coldef_type]]
        else:
            whatArr_sim = [['sim_id',operator.eq,method_id],['observer_coldef_type',operator.eq,coldef_type]]
        samsem_data_sim = organizeArray(samsem_data, whatArr_sim)
        
        boxes = []; labels = []; accuracies = {}; mean_cis = {}; order = {}; i=1
        
        for image_id in image_ids:
            whatArr_image = [['image_id', operator.eq, image_id]]
            samsem_data_image = organizeArray(samsem_data_sim,whatArr_image)
            
            dict.update({'filename':str(settings.id2ColDef[coldef_type])+'-simulation-method-'+str(settings.id2Sim[method_id])+'_images'})
            
            # 3. Get response time data
            alg_values = samsem_data_image[samsem_data_image['is_correct']==True]['resp_time'].values*1000; 
            if ('RT_boxplots' in plot_types) or ('median-test' in plot_types):
                boxes.append(alg_values)
            labels.append(image_id) if alg_values.size else labels.append(str(image_id) + ' - No data'); 
                
            # 4. Get CI of RT means
            if 'RT_means' in plot_types:
                alg_mean = getCIAverage(alg_values);
                alg_mean.append(labels[i-1])
                mean_cis.update({image_id: alg_mean})
            
            # 5. Get accuracy data
            if 'ACC_CIs' in plot_types:
                alg_acc = getAccuracy(samsem_data_image)
                alg_acc.append(labels[i-1])
                accuracies.update({image_id: alg_acc})
            
            order.update({i:image_id})
            i += 1
        if 'RT_boxplots' in plot_types:
            # 6. Plot response time data
            plotRTGraphs(boxes,labels,path_res,dict,order)
        
        if 'RT_means' in plot_types:
            # 7. Plot CI means of RT data
            plotCIAverageGraphs(mean_cis,path_res,dict,order)
        
        if 'ACC_CIs' in plot_types:
            # 8. Plot accuracy data
            plotAccuracyGraphs(accuracies,path_res,dict,order)
        
        if 'median-test' in plot_types:
            # 9. Make median test
            dict.update({'filename': dict['filename']+"-RT"})
            makeMedianTest(numpy.array(boxes), path_res, image_ids,dict)
        
def samsemPlots29and30(samsem_data,path,dict):
    """
    Plots the SaMSEM Res#29+30.
    Meaning that all images simulating one specific color deficiency are collapsed together for all algorithms.
    On the x-axis, we have the the observer groups with different color deficiency types. The result images are stored in (subfolders) defined by the path input.
    Input: * samsem_data:    SaMSEM data that is to be analyzed
           * path:           Path to the folder, where the result images should be stored. 
           * dict:           A dictionary containing different options.
                             ** result_id:    ID of the result images according to the meeting from 18/09/2014. 
                             ** coldef_type:  The color deficiency type of the simulation being used. 
                             ** filename:     Name of the file as which the result image should be stored.
    Before: s2mplots29and30
    """
    
    # Defining the subfolder for the results
    path_res = path    
    if dict.has_key('subfolder'):
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    filename = dict['filename']
    coldef_type = dict['coldef_type']
    print "Starting SAMSEM_RES#29+30: Analyzing data of all observer groups for " + str(settings.id2ColDefLong[dict['coldef_type']]) + " simulation methods."
    
    # Ignore dummy algorithm
    #whatArr_tmp = [['sim_id',operator.ne,99],];howArr_tmp=[]
    #samsem_data = organizeArray(samsem_data,whatArr_tmp,howArr_tmp)
    #sim_ids = set(samsem_data['sim_id'].values.astype(int))
    
    if dict.has_key('method_ids'):
        method_ids = dict['method_ids']
    else:
        method_ids = set(samsem_data['sim_id'].values.astype(int))
        if 99 in method_ids: method_ids.remove(99)
        
    if dict.has_key('plot_types'):
        plot_types = dict['plot_types']
    else:
        plot_types = ['RT_boxplots', 'ACC_CIs', 'median']
    
    rel_data = pandas.DataFrame()
    for method_id in method_ids:
        if (method_id != 3) and (method_id != 99):
            whatArr_tmp = [['sim_id',operator.eq,method_id],['coldef_type',operator.eq,coldef_type]]
        else:
            whatArr_tmp = [['sim_id',operator.eq,method_id]]
        rel_data_tmp = organizeArray(samsem_data,whatArr_tmp)
        rel_data = pandas.concat([rel_data_tmp, rel_data])
    samsem_data_adj = rel_data.reset_index()
    
    # 1. Retrieving data for the three observer groups
    # Data for normal sighted observers
    whatArr_tmp = [['observer_coldef_type',operator.eq,0]]
    normal_data = organizeArray(samsem_data_adj,whatArr_tmp)
    # Data for protan observers
    whatArr_tmp = [['observer_coldef_type',operator.eq,1]]
    protan_data = organizeArray(samsem_data_adj,whatArr_tmp)
    # Data for deutan observers
    whatArr_tmp = [['observer_coldef_type',operator.eq,2]]
    deutan_data = organizeArray(samsem_data_adj,whatArr_tmp)
    
    pandas_dict = {'normal':normal_data, 'protan':protan_data, 'deutan': deutan_data}
    order_dict = {0: 'normal', 1: 'protan', 2: 'deutan'}
    observer_group_names = ['normal', 'protan', 'deutan']
    
    # 2. Make response time boxplots
    if 'RT_boxplots' in plot_types:
        boxes, labels = preparePandas4RTPlots(pandas_dict, order_dict)
        plotRTGraphs(boxes,labels,path_res, dict)
    
    # 3. Make median test
    if 'median-test' in plot_types:
        boxes, labels = preparePandas4RTPlots(pandas_dict, order_dict)
        makeMedianTest(boxes, path_res, labels, dict)
    
    # 4. Plot accuracy confidence intervals        
    if 'ACC_CIs' in plot_types:
        accuracies = preparePandas4AccuracyPlots(pandas_dict)
        plotAccuracyGraphs(accuracies,path_res,dict,order_dict)

def samsemPlots31and32(samsem_data,path,dict):
    """
    Plots the SaMSEM Res#29+30.
    Each image simulating one specific color deficiency is analyzed individually collapsed together for all algorithms.
    On the x-axis, we have the the observer groups with different color deficiency types. The result images are stored in (subfolders) defined by the path input.
    Input: * samsem_data:    SaMSEM data that is to be analyzed
           * path:           Path to the folder, where the result images should be stored. 
           * dict:           A dictionary containing different options.
                             ** result_id:    ID of the result images according to the meeting from 18/09/2014. 
                             ** coldef_type:  The color deficiency type of the simulation being used.
                             ** filename:     Name of the file as which the result image should be stored.
    """
    
    result_id = dict['result_id']   
    if not os.path.exists(os.path.join(path,str(result_id))): os.makedirs(os.path.join(path,str(result_id)))
    
    intro_string = result_id if result_id else dict['filename']  # filename does not make any sense at this point
    sys.stdout.write("Starting Res#"+str(intro_string)+'.') 
    
     # Ignore dummy algorithm
    whatArr_tmp = [['sim_id',operator.ne,99],];howArr_tmp=[]
    samsem_data = organizeArray(samsem_data,whatArr_tmp,howArr_tmp)
    
    coldef_type = dict['coldef_type']
    
    # 1. Retrieving data for the three observer groups
    # Data for normal sighted observers
    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq,0]];howArr_tmp=[]
    normal_data = organizeArray(samsem_data,whatArr_tmp,howArr_tmp)
    # Data for protan observers
    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq,1]];howArr_tmp=[]
    protan_data = organizeArray(samsem_data,whatArr_tmp,howArr_tmp)
    # Data for deutan observers
    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq,2]];howArr_tmp=[]
    deutan_data = organizeArray(samsem_data,whatArr_tmp,howArr_tmp)
    
    obs_title = dict['obs_title']
    
    image_ids = set(samsem_data['image_id'].values.astype(int))
    for image_id in image_ids:
        filename_tmp = 'C-99'+str(coldef_type)+'999'+'9'+str(image_id).zfill(3)
        dict.update({'filename':filename_tmp})
        
        obs_title_tmp = str(image_id).zfill(3)+' - '+obs_title
        dict.update({'obs_title':obs_title_tmp}) 
        
        boxes = []; accuracies = {}; mean_cis = {}; labels = []
        
        whatArr_tmp = [['image_id',operator.eq,image_id],];howArr_tmp=[]
        
        # 2. Get response time data
        # For normal sighted observers
        normal_data_tmp = organizeArray(normal_data,whatArr_tmp,howArr_tmp)
        normal_response_values = normal_data_tmp[normal_data_tmp['is_correct']==True]['resp_time'].values*1000; labels.append('Norm.obs.') if normal_response_values.size else labels.append('Norm.obs. - No data'); 
        boxes.append(normal_response_values)
        # For protan sighted observers
        protan_data_tmp = organizeArray(protan_data,whatArr_tmp,howArr_tmp)
        protan_response_values = protan_data_tmp[protan_data_tmp['is_correct']==True]['resp_time'].values*1000; labels.append('Prot.obs.') if protan_response_values.size else labels.append('Prot.obs. - No data'); 
        boxes.append(protan_response_values)
        # For deutan sighted observers
        deutan_data_tmp = organizeArray(deutan_data,whatArr_tmp,howArr_tmp)
        deutan_response_values = deutan_data_tmp[deutan_data_tmp['is_correct']==True]['resp_time'].values*1000; labels.append('Deut.obs.') if deutan_response_values.size else labels.append('Deut.obs. - No data'); 
        boxes.append(deutan_response_values)
        
        # 3. Get CI data
        # For normal sighted observers
        mean_normal = getCIAverage(normal_response_values);
        mean_normal.append(labels[0])
        # For protan observers
        mean_protan = getCIAverage(protan_response_values);
        mean_protan.append(labels[1])
        # For deutan observers
        mean_deutan = getCIAverage(deutan_response_values);
        mean_deutan.append(labels[2])
    
        mean_cis.update({'norm': mean_normal,
                         'protan': mean_protan,
                         'deutan': mean_deutan})
        
        # 4. Get accuracy data
        # For normal sighted observers
        norm_acc = getAccuracy(normal_data)
        norm_acc.append(labels[0])
        # For protan observers
        prot_acc = getAccuracy(protan_data)
        prot_acc.append(labels[1])
        # For deutan observers
        deut_acc = getAccuracy(deutan_data)
        deut_acc.append(labels[2])
        
        accuracies.update({'norm': norm_acc,
                           'protan': prot_acc,
                           'deutan': deut_acc})
        
        
        # 5. Plot response time data
        plotRTGraphs(boxes,labels,path,dict)
        # 6. Plot mean data
        order={1:'norm', 2:'protan', 3:'deutan'}
        plotCIAverageGraphs(mean_cis,path,dict,order)
        # 7. Plot accrucay data
        plotAccuracyGraphs(accuracies, path, dict, order)
        
def samsemPlots35and36(samsem_data,path,dict):
    """
    Plots the SaMSEM Res#35+36. Similar to checkNormality.
    Each image showing distribution for one algorithm simulating one specific color deficiency for all images collapsed together.
    On the x-axis, ???.
    The result images are stored in (subfolders) defined by the path input.
    Input: * samsem_data:    SaMSEM data that is to be analyzed
           * path:           Path to the folder, where the result images should be stored. 
           * dict:           A dictionary containing different options.
                             ** result_id:    ID of the result images according to the meeting from 18/09/2014. 
                             ** coldef_type:  The color deficiency type of the simulation being used.
                             ** filename:     Name of the file as which the result image should be stored.
                             ** plot_type:    Type of plot that is being used. Can be either histogram, residual-plot, aller qq-plot.
    """
    
    path_res = path
    if dict.has_key('result_id'):
        result_id = dict['result_id']   
        if not os.path.exists(os.path.join(path,str(result_id))): os.makedirs(os.path.join(path,str(result_id)))
    
    if dict.has_key('subfolder'):
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    #intro_string = result_id if result_id else dict['filename']  # filename does not make any sense at this point
    #sys.stdout.write("Starting Res#"+str(intro_string)+'.') 
    coldef_type = dict['coldef_type']
    if dict.has_key('plot_types'):
        plot_types = dict['plot_types']
    else:
        plot_types = ['histogramm', 'residual', 'q-q']
    print "Starting SaMSEM_RES#35+36: Computing normality plot ("+str(plot_types)+") for "+str(settings.id2ColDefLong[coldef_type])+" simulation methods."
    
    boxes = []; labels = []
    
    sim_ids = set(samsem_data['sim_id'].values.astype(int))
    for sim_id in sim_ids:
        
        filename = str(settings.id2ColDefLong[coldef_type])+"-method-"+str(settings.id2Sim[sim_id])+"-RT"
        dict.update({'filename': filename})
        
        if (sim_id != 3) and (sim_id != 99):
            whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq,coldef_type],['sim_id',operator.eq,sim_id]]
        else:
            whatArr_tmp = [['observer_coldef_type',operator.eq,coldef_type],['sim_id',operator.eq,sim_id]] # All dummy simulations are only noted for protanopia
        alg_data = organizeArray(samsem_data,whatArr_tmp)
        
        alg_RT_values = (alg_data[alg_data['is_correct']==True]['resp_time'].values*1000)
        
        if "histogram" in plot_types:
            plotHistogram(alg_RT_values, path_res, dict)
        if "residual" in plot_types:
            boxes.append(alg_RT_values)
            labels.append(settings.id2Sim[sim_id]) if alg_RT_values.size else labels.append(str(sim_id) + ' - No data');
        if "q-q" in plot_types:
            plotQQPlot(alg_RT_values, path_res, dict)
    
    if "residual" in plot_types:
        filename = "99"+str(coldef_type)+"999"+str(coldef_type)+"999"
        dict.update({'filename': filename})
        plotResidualPlots(boxes, labels, path_res, dict)

def samsemPlots37and38(samsem_data,path,dict):
    """
    Plots the SaMSEM Res#37-38. Similar to checkNormality.
    Each image showing distribution for one observer group and simulation method simulating one specific color deficiency for all images collapsed together.
    On the x-axis, ???.
    The result images are stored in (subfolders) defined by the path input.
    Input: * samsem_data:    SaMSEM data that is to be analyzed
           * path:           Path to the folder, where the result images should be stored. 
           * dict:           A dictionary containing different options.
                             ** result_id:    ID of the result images according to the meeting from 18/09/2014. 
                             ** coldef_type:  The color deficiency type of the simulation being used.
                             ** filename:     Name of the file as which the result image should be stored.
                             ** plot_type:    Type of plot that is being used. Can be either histogram, residual-plot, aller qq-plot.
    """
    
    path_res = path
    if dict.has_key('result_id'):
        result_id = dict['result_id']   
        if not os.path.exists(os.path.join(path,str(result_id))): os.makedirs(os.path.join(path,str(result_id)))
    
    if dict.has_key('subfolder'):
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    #intro_string = result_id if result_id else dict['filename']  # filename does not make any sense at this point
    #sys.stdout.write("Starting Res#"+str(intro_string)+'.') 
    coldef_type = dict['coldef_type']
    if dict.has_key('plot_types'):
        plot_types = dict['plot_types']
    else:
        plot_types = ['histogramm', 'q-q']
    print "Starting SaMSEM_RES#37+38: Computing normality plot ("+str(plot_types)+") of observer groups for "+str(settings.id2ColDefLong[coldef_type])+" simulation methods."
    
    if dict.has_key('filename'):
        filename = dict['filename']
    else:
        filename = "test-pupsi"
    
    boxes = []; labels = []
    
    method_ids = set(samsem_data['sim_id'].values.astype(int))
    rel_data = pandas.DataFrame()
    for method_id in method_ids:
        if (method_id != 3) and (method_id != 99):
            whatArr_tmp = [['sim_id',operator.eq,method_id],['coldef_type',operator.eq,coldef_type]]
        else:
            whatArr_tmp = [['sim_id',operator.eq,method_id]]
        rel_data_tmp = organizeArray(samsem_data,whatArr_tmp)
        rel_data = pandas.concat([rel_data_tmp, rel_data])
    samsem_data_adj = rel_data.reset_index()
    
    # 1. Retrieving data for the three observer groups
    # Data for normal sighted observers
    whatArr_tmp = [['observer_coldef_type',operator.eq,0]]
    normal_data = organizeArray(samsem_data_adj,whatArr_tmp)
    # Data for protan observers
    whatArr_tmp = [['observer_coldef_type',operator.eq,1]]
    protan_data = organizeArray(samsem_data_adj,whatArr_tmp)
    # Data for deutan observers
    whatArr_tmp = [['observer_coldef_type',operator.eq,2]]
    deutan_data = organizeArray(samsem_data_adj,whatArr_tmp)
    
    # 2. Get response time data
    # For normal sighted observers
    normal_response_values = normal_data[normal_data['is_correct']==True]['resp_time'].values*1000; labels.append('Normal') if normal_response_values.size else labels.append('Normal - No data'); 
    # For protan observers
    protan_response_values = protan_data[protan_data['is_correct']==True]['resp_time'].values*1000; labels.append('Protan') if protan_response_values.size else labels.append('Protan - No data'); 
    # For deutan observers
    deutan_response_values = deutan_data[deutan_data['is_correct']==True]['resp_time'].values*1000; labels.append('Deutan') if deutan_response_values.size else labels.append('Protan - No data'); 
    
    if "histogram" in plot_types:
        filename_tmp = filename + "-normal-RT"
        plotHistogram(normal_response_values, path_res, dict.update({'filename': filename_tmp}))
        filename_tmp = filename + "-protan-RT"
        plotHistogram(protan_response_values, path_res, dict.update({'filename': filename_tmp}))
        filename_tmp = filename + "-deutan-RT"
        plotHistogram(deutan_response_values, path_res, dict.update({'filename': filename_tmp}))
    
    if "q-q" in plot_types:
        filename_tmp = filename + "-normal-RT"
        dict.update({'filename': filename_tmp})
        plotQQPlot(normal_response_values, path_res, dict)

        filename_tmp = filename + "-protan-RT"
        dict.update({'filename': filename_tmp})
        plotQQPlot(protan_response_values, path_res, dict)
        
        filename_tmp = filename + "-deutan-RT"
        dict.update({'filename': filename_tmp})
        plotQQPlot(deutan_response_values, path_res, dict)


def samsemPlots41and42(samsem_data,path,dict):
    """
    Making pearson chi2-contingency test for all algorithms samsem data.
    """
    coldef_type = dict['coldef_type']
    print "Starting SAMSEM_RES#41+42: Computing of Chi2 for simulation methods of " + str(settings.id2ColDefLong[coldef_type])+"."
    
    if dict.has_key('subfolder'):
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    # Ignore dummy algorithm
    #whatArr_tmp = [['sim_id',operator.ne,99]];howArr_tmp=[]
    #samsem_data = organizeArray(samsem_data,whatArr_tmp,howArr_tmp)
    
    corrArr = []; uncorrArr = []; data = {}; sim_names = []; chi2_pandas = {}
    
    sim_ids = sorted(set(samsem_data['sim_id'].values.astype(int)))
    for sim_id in sim_ids:
        sim_name_tmp = settings.id2Sim[sim_id]
        sim_names.append(sim_name_tmp)
        if (sim_id !=3) and (sim_id != 99):
            whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq,coldef_type],['sim_id',operator.eq,sim_id]];howArr_tmp=[]
        else:
            whatArr_tmp = [['observer_coldef_type',operator.eq,coldef_type],['sim_id',operator.eq,sim_id]]
        alg_data_tmp = organizeArray(samsem_data,whatArr_tmp)
        
        #pandas_dict
        chi2_pandas.update({sim_name_tmp: alg_data_tmp})
    
    # Make Chi2 contingency test
    obs_array, obs_pandas = preparePandas4Chi2(chi2_pandas,{0: 'vienot', 1: 'vienot-adjusted', 2: 'kotera', 3: 'brettel', 4: 'dummy'})
    
    start = 0; end = 4
    obs_adj = obs_array[:,start:end]
    chi2, p, dof, ex  = stats.chi2_contingency(obs_adj) # Compare only simulation methods
    
    res_str = ""
    res_str = res_str + "Simulation methods and observations:\n" + str(obs_pandas)
    res_str = res_str + "\n\nSimulation methods included in test:\n" + str(sim_names[start:end])
    res_str = res_str + "\nChi2: %f, p-value: %E, dof: %i, expect: " % (chi2, p, dof) + "\n"+str(ex)
    text_file = open(os.path.join(path_res,settings.id2ColDefLong[coldef_type]+"-methods-ACC_pearson-chi2-contingency-test_p-value.txt"), "w+")
    text_file.write(res_str)
    text_file.close()
    
    writePandastoLatex(obs_pandas, os.path.join(path_res,settings.id2ColDefLong[coldef_type]+"-methods-ACC_observations.tex"))
    
    # Make Chi2 contingency 2x2 test matrix
    dict.update({'filename': dict['filename']+'-ACC'})
    makePearsonChi2Contingency2x2Test(obs_array, path_res, sim_names, dict)


    
def samsemPlots43and44(samsem_data,path,dict):
    """
    Making chi2-contingency test for each observer group of the samsem data.
    """
    
    coldef_type = dict['coldef_type']
    print "Starting SAMSEM_RES#43+44: Computing of Chi2 for observer groups for " + str(settings.id2ColDefLong[coldef_type]) + " simulation methods."
    
    
    if dict.has_key('subfolder'):
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    obs_groups = ['normal','protan','deutan']
    
    # Ignore dummy algorithm
    whatArr_tmp = [['sim_id',operator.ne,99]]
    samsem_data_wo99 = organizeArray(samsem_data,whatArr_tmp)
    
    method_ids = set(samsem_data['sim_id'].values.astype(int))
    rel_data = pandas.DataFrame()
    for method_id in method_ids:
        if (method_id != 3) and (method_id != 99):
            whatArr_tmp = [['sim_id',operator.eq,method_id],['coldef_type',operator.eq,coldef_type]]
        elif (method_id == 3):
            whatArr_tmp = [['sim_id',operator.eq,method_id]]
        rel_data_tmp = organizeArray(samsem_data,whatArr_tmp)
        rel_data = pandas.concat([rel_data_tmp, rel_data])
    samsem_data_adj = rel_data.reset_index()
    
    # 1. Retrieving data for the three observer groups
    # Data for normal sighted observers
    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq,0]]
    normal_data = organizeArray(samsem_data_adj,whatArr_tmp)
    
    # Data for protan observers
    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq,1]]
    protan_data = organizeArray(samsem_data_adj,whatArr_tmp)
    
    # Data for deutan observers
    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq,2]]
    deutan_data = organizeArray(samsem_data_adj,whatArr_tmp)
    
    # Make Chi2 contingency test
    obs_array, obs_pandas = preparePandas4Chi2({'normal': normal_data, 'protan': protan_data, 'deutan': deutan_data},{0: 'normal', 1: 'protan', 2: 'deutan'})
    
    chi2, p, dof, ex  = stats.chi2_contingency(obs_array) # Compare only simulation methods
    res_str = ""
    res_str = res_str + "Observer groups and observations:\n" + str(obs_pandas)
    res_str = res_str + "\n\nChi2: %f, p-value: %E, dof: %i, expect: " % (chi2, p, dof) + "\n"+str(ex)
    text_file = open(os.path.join(path_res,settings.id2ColDefLong[coldef_type]+"-methods_obs-groups-ACC_pearson-chi2-contingency-test_p-value.txt"), "w+")
    text_file.write(res_str)
    text_file.close()
    
    writePandastoLatex(obs_pandas, os.path.join(path_res,settings.id2ColDefLong[coldef_type]+"-methods_obs-groups-ACC_observations.tex"))
    
    # Make Chi2 contingency test matrix
    dict.update({'filename': dict['filename']+'-ACC'})
    makePearsonChi2Contingency2x2Test(obs_array, path_res, obs_groups, dict)

            

###
###  ViSDEM result plots
###

def vsplots1thru4(visual_search_data,path,dict):
    sys.stdout.write("Starting Res#"+str(dict['result_id'])+':')    
    if not os.path.exists(os.path.join(path,str(dict['result_id']))): os.makedirs(os.path.join(path,str(dict['result_id'])))
    
    set_ids = set(visual_search_data['set_id'].values.astype(int))        
    for set_id in set_ids:
        sys.stdout.write("Set#"+str(set_id)+'.')
        # extract only DN and DE images for norm.sigh.obs.
        filename_tmp = '000999'+dict['obs_coldef_type']+str(set_id).zfill(3)+'9999-v'
            
        whatArr_tmp = [['version_id',operator.eq,0],dict['obs_operator'],['set_id',operator.eq,set_id],['is_correct',operator.eq,True]];howArr_tmp = []
        DNData =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
            
        whatArr_tmp = [['version_id',operator.eq,2],['dalt_id',operator.eq,0],dict['obs_operator'],['set_id',operator.eq,set_id],['is_correct',operator.eq,True]];howArr_tmp = []
        DEData = organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
            
        boxes_tmp = [];labels_tmp = []
        DND_values = DNData['resp_time'].values; labels_tmp.append('DN') if DND_values.size else labels_tmp.append('DN - No data'); boxes_tmp.append(DND_values)
        DED_values = DEData['resp_time'].values; labels_tmp.append('DE') if DED_values.size else labels_tmp.append('DE - No data'); boxes_tmp.append(DED_values)
            
        plt.figure(); plt.boxplot(boxes_tmp, notch=1)
        plt.xticks([1,2],labels_tmp); plt.title('Set '+str(set_id)+': '+dict['obs_title']); plt.ylabel('response time'); plt.ylim([0,5])
        plt.savefig(os.path.join(path,str(dict['result_id']),filename_tmp+".png")); plt.close()
    
    sys.stdout.write('\n')
  
def vsplots5thru8(visual_search_data,path,dict):
    sys.stdout.write("Starting Res#"+str(dict['result_id'])+':')    
    if not os.path.exists(os.path.join(path,str(dict['result_id']))): os.makedirs(os.path.join(path,str(dict['result_id'])))
    
    scene_ids = set(visual_search_data['scene_id'].values.astype(int))        
    for scene_id in scene_ids:
        sys.stdout.write("Scene#"+str(scene_id)+'.')
        # extract only DN and DE images for norm.sigh.obs.
        filename_tmp = '000999'+dict['obs_coldef_type']+str(getSetFromScene(scene_id)).zfill(3)+str(scene_id).zfill(3)+'9-v'
            
        whatArr_tmp = [['version_id',operator.eq,0],dict['obs_operator'],['scene_id',operator.eq,scene_id],['is_correct',operator.eq,True]];howArr_tmp = []
        DNData =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
            
        whatArr_tmp = [['version_id',operator.eq,2],['dalt_id',operator.eq,0],dict['obs_operator'],['scene_id',operator.eq,scene_id],['is_correct',operator.eq,True]];howArr_tmp = []
        DEData = organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
            
        boxes_tmp = [];labels_tmp = []
        DND_values = DNData['resp_time'].values; labels_tmp.append('DN') if DND_values.size else labels_tmp.append('DN - No data'); boxes_tmp.append(DND_values)
        DED_values = DEData['resp_time'].values; labels_tmp.append('DE') if DED_values.size else labels_tmp.append('DE - No data'); boxes_tmp.append(DED_values)
            
        plt.figure(); plt.boxplot(boxes_tmp, notch=1)
        plt.xticks([1,2],labels_tmp); plt.title('Scene '+str(scene_id)+': '+dict['obs_title']); plt.ylabel('response time'); plt.ylim([0,5])
        plt.savefig(os.path.join(path,str(dict['result_id']),filename_tmp+".png")); plt.close()
    
    sys.stdout.write('\n')


    
def visdemPlots67thru70(visdem_data,path,dict):
    
    print "Starting SAMSEM_RES#67-70: Plotting of ACC and RT data, and computying of Chi2 of ACC and median test of RT for observer group " + dict['filename']+"."
    
    path_res = path
    if dict.has_key('subfolder'):
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    dict.update({'investigated-item': 'daltonization method'})
    
    DNData = pandas.DataFrame(); DEData = pandas.DataFrame(); DCData = pandas.DataFrame()
    if dict.has_key('sets'):
        set_ids = dict['sets']
    else:
        set_ids = sorted(set(visdem_data['set_id'].values.astype(int)))
    
    visdem_data_adj = pandas.DataFrame()
    for set_id in set_ids:
        # Getting data from all variants for all normal sighted observers
        whatArr_tmp = [['set_id',operator.eq,set_id]]
        rel_data_tmp =  organizeArray(visdem_data,whatArr_tmp)
        visdem_data_adj = pandas.concat([rel_data_tmp, visdem_data_adj])
    visdem_data_adj.reset_index()
    
    # 1. Retrieving data from the image versions
    whatArr_tmp = [['version_id',operator.eq,0],['dalt_id',operator.eq,0],dict['obs_operator']]
    DNData =  organizeArray(visdem_data_adj,whatArr_tmp)
            
    whatArr_tmp = [['version_id',operator.eq,2],['dalt_id',operator.eq,0],dict['obs_operator']]
    DEData = organizeArray(visdem_data_adj,whatArr_tmp)
        
    whatArr_tmp = [['version_id',operator.eq,1],['dalt_id',operator.eq,0],dict['obs_operator']]
    DCData = organizeArray(visdem_data_adj,whatArr_tmp)
    
    pandas_dict = {'DN': DNData, 'DC': DCData, 'DE': DEData}
    order_dict = {0: 'DN', 1: 'DC', 2: 'DE'}
    
    # 2. Make response time plots
    boxes, labels = preparePandas4RTPlots(pandas_dict, order_dict)
    plotRTGraphs(boxes, labels, path_res, dict)
    
    # 3. Make Accuracy plots
    accuracies = preparePandas4AccuracyPlots(pandas_dict)
    plotAccuracyGraphs(accuracies, path_res, dict, order_dict)
    
    # 4. Make median test 
    dict.update({'filename': dict['filename']+'-RT'})
    makeMedianTest(boxes, path_res, labels, dict)
              
    # 5. Make Chi2 contingency test
    obs_array, obs_pandas = preparePandas4Chi2(pandas_dict, order_dict)
        
    dict.update({'filename': dict['filename']+'-ACC'})
    makePearsonChi2Contingency(obs_array, obs_pandas, labels, path_res, dict)

   
    # 6. Make Chi2 contingency test matrix
    makePearsonChi2Contingency2x2Test(obs_array, path_res, labels, dict)
    

def visdemPlots53thru60(visdem_data,path,dict):
    
    print "Starting SAMSEM_RES#53-60: Plotting of ACC and RT data, and computying of Chi2 of ACC and median test of RT for daltonization methods of observer group " + dict['filename']+"."
    
    
    path_res = path
    if dict.has_key('subfolder'):
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    filename_orig = dict['filename']
    
    coldef_type = dict['coldef_type']
    dict.update({'investigated-item': 'daltonization method'})
    
    if dict.has_key('sets'):
        set_ids = dict['sets']
    else:
        set_ids = sorted(set(visdem_data['set_id'].values.astype(int)))
        
    visdem_data_adj = pandas.DataFrame()
    for set_id in set_ids:
        # Getting data from all variants for all normal sighted observers
        whatArr_tmp = [['set_id',operator.eq,set_id]]
        rel_data_tmp =  organizeArray(visdem_data,whatArr_tmp)
        visdem_data_adj = pandas.concat([rel_data_tmp, visdem_data_adj])
    visdem_data_adj.reset_index()
    
    dalt_method_ids = sorted(set(visdem_data['dalt_id'].values.astype(int)))
    
    pandas_dict = {}; i = 0; order_dict = {}; dalt_names = []
    for dalt_method_id in dalt_method_ids:
        
        dalt_method = settings.id2Dalt[dalt_method_id]
        whatArray = [['dalt_id', operator.eq, dalt_method_id],['version_id',operator.eq,1],dict['obs_operator']]
        if (dalt_method_id != 0)  and (dalt_method_id != 99):
            whatArray.append(['coldef_type',operator.eq,coldef_type])
        relevant_data_tmp = organizeArray(visdem_data_adj, whatArray)
        
        pandas_dict.update({dalt_method: relevant_data_tmp})
        order_dict.update({i: dalt_method}); i += 1
        dalt_names.append(dalt_method)
    
    # Make response time plots
    boxes, labels = preparePandas4RTPlots(pandas_dict, order_dict)
    plotRTGraphs(boxes, labels, path_res, dict)
    
    # Make Accuracy plots
    accuracies = preparePandas4AccuracyPlots(pandas_dict)
    plotAccuracyGraphs(accuracies, path_res, dict, order_dict)
    
    # Make median test
    dict.update({'filename': filename_orig+'-RT'})
    makeMedianTest(boxes, path_res, labels, dict)
    
    # Make Chi2 contingency test
    obs_array, obs_pandas = preparePandas4Chi2(pandas_dict, order_dict)
    
    start = 1; end = 5
    res = [start,end]
    dict.update({'filename': filename_orig+'-ACC'})
    
    makePearsonChi2Contingency(obs_array, obs_pandas, labels, path_res, dict, res)
    
    # Make Chi2 contingency test matrix
    makePearsonChi2Contingency2x2Test(obs_array, path_res, labels, dict)   

def visdemPlots53thru60Paired(visdem_data,path,dict):
    
    print "Starting SAMSEM_RES#53-60-paired: Plotting Paired RT data, and computying of Chi2 of ACC and median test of RT for daltonization methods of observer group " + dict['filename']+"."
    
    compute_paired_data = 1
    filename_orig = dict['filename'] 
    
    path_res = path
    if dict.has_key('subfolder'):
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    orig_filenmae = dict['filename']
    
    coldef_type = dict['coldef_type']
    
    if dict.has_key('sets'):
        set_ids = dict['sets']
    else:
        set_ids = sorted(set(visdem_data['set_id'].values.astype(int)))
        
    visdem_data_adj = pandas.DataFrame()
    for set_id in set_ids:
        # Getting data from all variants for all normal sighted observers
        whatArr_tmp = [['set_id',operator.eq,set_id]]
        rel_data_tmp =  organizeArray(visdem_data,whatArr_tmp)
        visdem_data_adj = pandas.concat([rel_data_tmp, visdem_data_adj])
    visdem_data_adj.reset_index()
    
    dalt_method_ids = sorted(set(visdem_data['dalt_id'].values.astype(int)))
    
    pandas_dict = {}; i = 0; order_dict = {}; dalt_names = []
    for dalt_method_id in dalt_method_ids:
        
        dalt_method = settings.id2Dalt[dalt_method_id]
        whatArray = [['dalt_id', operator.eq, dalt_method_id],['version_id',operator.eq,1],dict['obs_operator']]
        if (dalt_method_id != 0)  and (dalt_method_id != 2) and (dalt_method_id != 99):
            whatArray.append(['coldef_type',operator.eq,coldef_type])
        relevant_data_tmp = organizeArray(visdem_data_adj, whatArray)
        
        pandas_dict.update({dalt_method: relevant_data_tmp})
        order_dict.update({i: dalt_method}); i += 1
        dalt_names.append(dalt_method)
    
    if compute_paired_data:
        makePairedDataForViSDEM(visdem_data_adj, dict['path_data'], dict)
    
    visdem_data_paired_path = os.path.join(dict['path_data'],dict['filename']+'_visdem-data-RT-paired.csv')
    visdem_data_paired = pandas.read_csv(visdem_data_paired_path,index_col=False,sep=';')    
    
    f = open(os.path.join(dict['path_data'],dict['filename']+'_visdem-data-RT-paired_meta-data.txt'), 'r')
    b = json.load(f)
    f.close()
    
    start = 0; end = 5
    visdem_data_paired_array, labels = makePairedRTData(visdem_data_paired, b, 'dalt_id')
    
    # Remove NaN values from array
    visdem_data_paired_wonan_array = []
    for a in visdem_data_paired_array:
        visdem_data_paired_wonan_array.append(a[numpy.isnan(a)==False])
    
    # Plot response times
    dict.update({'filename': filename_orig+'-paired'})
    plotRTGraphs(visdem_data_paired_wonan_array[start:end], labels[start:end], path_res, dict)
    
    # Make median test
    makeMedianTest(visdem_data_paired_wonan_array[start:end], path_res, labels[start:end], dict)
    
    #print visdem_data_paired_wonan_array
    #x[~numpy.isnan(x)]
    
     
def vsplots71thru74(visual_search_data,path,dict):
    
    result_id = dict['result_id']
    intro_string = result_id if result_id else dict['filename'] 
    sys.stdout.write("Starting Res#"+str(intro_string)+' -> ')    
    if not os.path.exists(os.path.join(path,str(dict['result_id']))): os.makedirs(os.path.join(path,str(dict['result_id'])))
    
    if dict.has_key('sets'):
        set_ids = dict['sets']
    else:
        set_ids = []
    
    beforeData = pandas.DataFrame(); afterData = pandas.DataFrame();
    
    for set_id in set_ids:
        # get all the data before daltonization
        whatArr_tmp = [['dalt_id',operator.eq,0],['version_id',operator.eq,1],dict['obs_operator'],];howArr_tmp = []
        beforeData_tmp =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        beforeData = pandas.concat([beforeData_tmp,beforeData])
        
        # get all the data after daltonization
        whatArr_tmp = [['dalt_id',operator.eq,2],['version_id',operator.eq,1],dict['obs_operator'],];howArr_tmp = []
        afterData_tmp =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        afterData = pandas.concat([afterData_tmp,afterData])
        
    boxes_tmp = [];labels_tmp = []
    before_values = beforeData[beforeData['is_correct']==True]['resp_time'].values; labels_tmp.append('before daltonization') if before_values.size else labels_tmp.append('before - No data'); boxes_tmp.append(before_values)
    after_values = afterData[afterData['is_correct']==True]['resp_time'].values; labels_tmp.append('after daltonization') if after_values.size else labels_tmp.append('after - No data'); boxes_tmp.append(after_values)
    
    plt.figure(); plt.boxplot(boxes_tmp, notch=1)
    plt.xticks([1,2,3],labels_tmp); plt.title(dict['obs_title']); plt.ylabel('response time'); plt.ylim([0,3])
    plt.grid(axis='y')
    #plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-R.eps"));
    plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-R.pdf")); 
    plt.close()
    
    # 3. Accuracy plots
    before_acc = getAccuracy(beforeData)
    before_acc.append(labels_tmp[0])
    after_acc = getAccuracy(afterData)
    after_acc.append(labels_tmp[1])
    accuracies = {'before': before_acc,
                  'after': after_acc}
    plotAccuracyGraphs(accuracies,path,dict,order=['before','after'])

def visdemPlot75(visual_search_data,path,dict):
    
    
    if dict.has_key('filename'):
        filename = dict['filename']
    else:    
        filename = "visdem-normal-deutan-observers"
    
    if dict.has_key('obs_title'):
        obs_title = dict['obs_title']
    else:
        obs_title = ''
    
    if dict.has_key('fontsize'):
        fontsize = dict['fontsize']
    else:
        fontsize = 18
        
    if dict.has_key('y_lim_ACC'):
        y_lim_ACC = dict['y_lim_ACC']
    else:
        y_lim_ACC = [.0,1.] 
        
    if dict.has_key('y_lim_RT'):
        y_lim_RT = dict['y_lim_RT']
    else:
        y_lim_RT = [0,3000]
    
    if dict.has_key('result_id'):
        result_id = dict['result_id']
    else:
        result_id = ''
    intro_string = result_id if result_id else filename
         
    sys.stdout.write("Starting Res#"+str(intro_string))    
    if not os.path.exists(os.path.join(path,str(result_id))): os.makedirs(os.path.join(path,str(result_id)))
    
    Data_norm = pandas.DataFrame()
    Data_deut = pandas.DataFrame()
    if dict.has_key('sets'):
        set_ids = dict['sets']
    else:
        set_ids = []
    
    # 1. Retrieving data from the image versions
    for set_id in set_ids:
        # Getting data from all variants for all normal sighted observers
        whatArr_tmp = [['dalt_id',operator.eq,0],['observer_coldef_type',operator.eq,0],['set_id',operator.eq,set_id]];howArr_tmp = []
        Data_norm_tmp =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        Data_norm = pandas.concat([Data_norm_tmp,Data_norm])
        
        # Getting data from all variant for all deutan color deficient observers
        whatArr_tmp = [['dalt_id',operator.eq,0],['observer_coldef_type',operator.eq,2],['set_id',operator.eq,set_id]];howArr_tmp = []
        Data_deut_tmp =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        Data_deut = pandas.concat([Data_deut_tmp,Data_deut])
    
    # 2. Plot reponse times
    # Normal sighted observers
    boxes_tmp = []; labels_tmp = []
    Data_norm_values = Data_norm[Data_norm['is_correct']==True]['resp_time'].values*1000; 
    labels_tmp.append('Normal') if Data_norm_values.size else labels_tmp.append('Normal - No data'); 
    boxes_tmp.append(Data_norm_values)
    
    # Deutan color deficient observers
    Data_deut_values = Data_deut[Data_deut['is_correct']==True]['resp_time'].values*1000; 
    labels_tmp.append('Deutan') if Data_deut_values.size else labels_tmp.append('Deutan - No data'); 
    boxes_tmp.append(Data_deut_values)
             
    # Let's start plotting!
    plt.figure(); 
    plt.boxplot(boxes_tmp, notch=1)
    plt.xticks([1,2],labels_tmp,fontsize=fontsize); plt.title(obs_title, fontsize=fontsize); plt.ylabel('Response Time (ms)',fontsize=fontsize); 
    plt.ylim(y_lim_RT); plt.grid(axis='y')
    plt.savefig(os.path.join(path,str(result_id),filename+"-R.pdf")); 
    plt.close()
    
    # 3. Run median test on RT data
    test_array = []
    test_array.append(Data_norm_values)
    test_array.append(Data_deut_values)
    test_array = numpy.array(test_array)
    print
    makeMedianTest(test_array,path,labels_tmp,dict)
     
    # 4. Accuracy plots
    # Normal sighted
    norm_acc = getAccuracy(Data_norm)
    norm_acc.append(labels_tmp[0])
    # Deutan color deficient
    deut_acc = getAccuracy(Data_deut)
    deut_acc.append(labels_tmp[1])
    accuracies_norm = {'Normal': norm_acc,
                       'Deutan': deut_acc}
    
    # Lets start plotting
    dict.update({'fontsize':fontsize,'filename': filename})
    plotAccuracyGraphs(accuracies_norm,path,dict,order={1:'Normal', 2:'Deutan'})
        
    sys.stdout.write(".")

 
    
def visdemPlot76(visual_search_data,path,dict):
    
    filename_norm = "visdem-normal-observers"
    filename_deut = "visdem-deutan-observers"
    filename_norm_deut = "visdem-normal-deutan-observers"
    if dict.has_key('fontsize'):
        fontsize = dict['fontsize']
    else:
        fontsize = 18
    
    result_id = dict['result_id']
    intro_string = result_id if result_id else dict['filename'] 
    sys.stdout.write("Starting Res#"+str(intro_string))    
    if not os.path.exists(os.path.join(path,str(dict['result_id']))): os.makedirs(os.path.join(path,str(dict['result_id'])))
    
    DNData_norm = pandas.DataFrame(); DCData_norm = pandas.DataFrame()
    DNData_deut = pandas.DataFrame(); DCData_deut = pandas.DataFrame()
    if dict.has_key('sets'):
        set_ids = dict['sets']
    else:
        set_ids = []
    
    # 1. Retrieving data from the image versions
    for set_id in set_ids:
        whatArr_tmp = [['version_id',operator.eq,0],['dalt_id',operator.eq,0],['observer_coldef_type',operator.eq,0],['set_id',operator.eq,set_id]];howArr_tmp = []
        DNData_norm_tmp =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        DNData_norm = pandas.concat([DNData_norm_tmp,DNData_norm])
        
        whatArr_tmp = [['version_id',operator.eq,1],['dalt_id',operator.eq,0],['observer_coldef_type',operator.eq,0],['set_id',operator.eq,set_id]];howArr_tmp = []
        DCData_norm_tmp = organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        DCData_norm = pandas.concat([DCData_norm_tmp,DCData_norm])
        
        whatArr_tmp = [['version_id',operator.eq,0],['dalt_id',operator.eq,0],['observer_coldef_type',operator.eq,2],['set_id',operator.eq,set_id]];howArr_tmp = []
        DNData_deut_tmp =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        DNData_deut = pandas.concat([DNData_deut_tmp,DNData_deut])
        
        whatArr_tmp = [['version_id',operator.eq,1],['dalt_id',operator.eq,0],['observer_coldef_type',operator.eq,2],['set_id',operator.eq,set_id]];howArr_tmp = []
        DCData_deut_tmp = organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        DCData_deut = pandas.concat([DCData_deut_tmp,DCData_deut])
    
    # 2. Plot reponse times
    boxes_norm_tmp = []; boxes_deut_tmp = []; labels_tmp = []
    DND_norm_values = DNData_norm[DNData_norm['is_correct']==True]['resp_time'].values*1000; 
    labels_tmp.append('DN') if DND_norm_values.size else labels_tmp.append('DN - No data'); 
    boxes_norm_tmp.append(DND_norm_values)
    DCD_norm_values = DCData_norm[DCData_norm['is_correct']==True]['resp_time'].values*1000; 
    labels_tmp.append('DC') if DCD_norm_values.size else labels_tmp.append('DC - No data'); 
    boxes_norm_tmp.append(DCD_norm_values) 
    
    DND_deut_values = DNData_deut[DNData_deut['is_correct']==True]['resp_time'].values*1000; 
    labels_tmp.append('DN') if DND_deut_values.size else labels_tmp.append('DN - No data'); 
    boxes_deut_tmp.append(DND_deut_values)
    DCD_deut_values = DCData_deut[DCData_deut['is_correct']==True]['resp_time'].values*1000; 
    labels_tmp.append('DC') if DCD_deut_values.size else labels_tmp.append('DC - No data'); 
    boxes_deut_tmp.append(DCD_deut_values) 
             
    plt.figure(); 
    plt.boxplot(boxes_norm_tmp, notch=1)
    plt.xticks([1,2],labels_tmp,fontsize=fontsize); plt.title(dict['obs_title']); plt.ylabel('Response Time (ms)',fontsize=fontsize); 
    plt.ylim([0,3000]); plt.grid(axis='y')
    #plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-R.eps"));
    plt.savefig(os.path.join(path,str(dict['result_id']),filename_norm+"-R.pdf")); 
    plt.close()
    
    plt.figure(); 
    plt.boxplot(boxes_deut_tmp, notch=1)
    plt.xticks([1,2],labels_tmp,fontsize=fontsize); plt.title(dict['obs_title']); plt.ylabel('Response Time (ms)',fontsize=fontsize); 
    plt.ylim([0,3000]); plt.grid(axis='y')
    #plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-R.eps"));
    plt.savefig(os.path.join(path,str(dict['result_id']),filename_deut+"-R.pdf")); 
    plt.close()
     
    # 4. Accuracy plots
    dn_norm_acc = getAccuracy(DNData_norm)
    dn_norm_acc.append(labels_tmp[0])
    dc_norm_acc = getAccuracy(DCData_norm)
    dc_norm_acc.append(labels_tmp[1])
    accuracies_norm = {'DN': dn_norm_acc,
                       'DC': dc_norm_acc}
    
    dn_deut_acc = getAccuracy(DNData_deut)
    dn_deut_acc.append(labels_tmp[0])
    dc_deut_acc = getAccuracy(DCData_deut)
    dc_deut_acc.append(labels_tmp[1])
    accuracies_deut = {'DN': dn_deut_acc,
                       'DC': dc_deut_acc}
    
    dict.update({'fontsize':fontsize,'filename': filename_norm_deut})
    plt.figure()
    dict.update({'fmt': 'or', 'color': 'red'})
    plotAccuracyGraphs(accuracies_norm,path,dict,order={1:'DN', 2:'DC'})
    dict.update({'fmt': 'D', 'color': 'blue'})
    plotAccuracyGraphs(accuracies_deut,path,dict,order={1:'DN', 2:'DC'})
    plt.close()
        
    sys.stdout.write(".")

# if 0:
#     test_path = "/Users/thomas/Desktop/tull/colordeficiency_tools/visual-search/colordeficiency-data/"
#     test_file = "/Users/thomas/Desktop/tull/colordeficiency_tools/visual-search/colordeficiency-data/11_2014_mai_16_1356.xlsx"
#     #dataArray, testArray, extraDataDict = extractVisualSearchData(test_file)
#     #dataArray, testArray, extraDataDict = extractExperimentData(test_file)
#     #print dataArray
#     #print testArray
#     #print extraDataDict
#     analyzeViSDEMData(test_path)
#     #plotVisualSearchData(os.path.join("/Users/thomas/Desktop/tull/colordeficiency_tools/results",'VS-plots'),'../colordeficiency-data/visual-search-data.csv')   
# elif 0:
#     test_path = "/Users/thomas/Desktop/tull/colordeficiency_tools/sample-2-match/colordeficiency-data/"
#     test_file = "/Users/thomas/Desktop/tull/colordeficiency_tools/sample-2-match/colordeficiency-data/11_2014_mai_16_1307.xlsx"
#     #dataArray, testArray, extraDataDict = extractExperimentData(test_file)
#     #print dataArray
#     #print testArray
#     #print extraDataDict
#     analyzeSaMSEMData(test_path)
#     #plotSample2MatchData(os.path.join("/Users/thomas/Desktop/tull/colordeficiency_tools/results/",'S2M-plots'), '../colordeficiency-data/sample-2-match-data.csv')
# #plotEI2015_SaMSEM_ViSDEM()

def makePairedDataForSimulation(samsem_data,path,dict):
    """
    Make paired data for all observers and all images collapsed of all algorithms individually.
    The result images are stored in (subfolders) defined by the path input.
    Input: * samsem_data:    SaMSEM data that is to be analyzed
           * path:           Path to the folder, where the result images should be stored.
    """
    
    # 1. Restrict data to only one type of observer color deficiency
    coldef_type = dict['coldef_type']
    whatArr_tmp = [['observer_coldef_type',operator.eq,coldef_type]];howArr_tmp=[]
    samsem_data_restr = organizeArray(samsem_data,whatArr_tmp,howArr_tmp)

    sim_ids = sorted(set(samsem_data_restr['sim_id'].values.astype(int)))
    observer_ids = sorted(set(samsem_data_restr['observer_id'].values.astype(int)))
    image_ids = sorted(set(samsem_data_restr['image_id'].values.astype(int)))
    
    columns=['observer_id', 'observer_coldef_type', 'image_id', 'coldef_type']
    for sim_id in sim_ids:
        col_tmp = "sim_id_"+str(sim_id).zfill(2)
        columns.append(col_tmp)
        
    samsem_data_template = pandas.DataFrame(columns=columns)
    samsem_data_RT_paired = samsem_data_template.copy()
    samsem_data_ACC_paired = samsem_data_template.copy()
    index = 0
    
    for observer_id in observer_ids:
        for image_id in image_ids:
            
            whatArr_tmp = [['observer_id',operator.eq,observer_id],['image_id',operator.eq,image_id]]; howArr_tmp = []
            samsem_data_tmp = organizeArray(samsem_data_restr,whatArr_tmp,howArr_tmp)
            
            pandas_RT_tmp = pandas.DataFrame({'observer_id': observer_id,
                                          'observer_coldef_type': coldef_type,
                                          'image_id': image_id,
                                          'coldef_type': coldef_type},[index])
            pandas_ACC_tmp = pandas.DataFrame({'observer_id': observer_id,
                                          'observer_coldef_type': coldef_type,
                                          'image_id': image_id,
                                          'coldef_type': coldef_type},[index])
            
            for sim_id in sim_ids:
                if sim_id not in [3,99]:
                    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['sim_id',operator.eq,sim_id]];
                    field = organizeArray(samsem_data_tmp,whatArr_tmp,howArr_tmp)
                else:
                    # For the Kotera and the dummy method, both protan and deutan version are the same
                    whatArr_tmp = [['sim_id',operator.eq,sim_id]];
                    field = organizeArray(samsem_data_tmp,whatArr_tmp,howArr_tmp)
                RT_tmp = float('NaN') if field.empty else field['resp_time'].values[0]*1000
                ACC_tmp = float('NaN') if field.empty else field['is_correct'].values[0]
                pandas_RT_tmp["sim_id_"+str(sim_id).zfill(2)]= float(RT_tmp)#str(observer_id).zfill(2)+'-'+str(image_id).zfill(2)+'-'+str(sim_id)+"-"+str(int(RT_tmp)).zfill(4)
                pandas_ACC_tmp["sim_id_"+str(sim_id).zfill(2)]= bool(ACC_tmp)
            samsem_data_RT_paired = samsem_data_RT_paired.append(pandas_RT_tmp)
            samsem_data_ACC_paired = samsem_data_ACC_paired.append(pandas_ACC_tmp)
            index += 1
    # Layout RT and ACC for storage in path
    samsem_data_RT_paired =  samsem_data_RT_paired[columns]
    samsem_data_RT_paired.observer_id = samsem_data_RT_paired.observer_id.astype(int)
    samsem_data_RT_paired.observer_coldef_type = samsem_data_RT_paired.observer_coldef_type.astype(int)
    samsem_data_RT_paired.image_id = samsem_data_RT_paired.image_id.astype(int)
    samsem_data_RT_paired.coldef_type = samsem_data_RT_paired.coldef_type.astype(int)
    samsem_data_RT_paired.to_csv(os.path.join(path,'samsem-data-RT-paired.csv'),sep=";")
    
    samsem_data_ACC_paired =  samsem_data_ACC_paired[columns]
    samsem_data_ACC_paired.observer_id = samsem_data_ACC_paired.observer_id.astype(int)
    samsem_data_ACC_paired.observer_coldef_type = samsem_data_ACC_paired.observer_coldef_type.astype(int)
    samsem_data_ACC_paired.image_id = samsem_data_ACC_paired.image_id.astype(int)
    samsem_data_ACC_paired.coldef_type = samsem_data_ACC_paired.coldef_type.astype(int)
    samsem_data_ACC_paired.to_csv(os.path.join(path,'samsem-data-ACC-paired.csv'),sep=";")
    
    f = open(os.path.join(path,'meta-data.txt'), 'w')
    json.dump(sim_ids, f)
    f.close()

def makePairedDataForViSDEM(visdem_data,path,dict):
    """
    Make paired data for all observers and all images collapsed of all algorithms individually.
    The result images are stored in (subfolders) defined by the path input.
    Input: * samsem_data:    SaMSEM data that is to be analyzed
           * path:           Path to the folder, where the result images should be stored.
    """
    
    # 1. Restrict data to only one type of observer color deficiency
    coldef_type = dict['coldef_type']
    whatArr_tmp = [dict['obs_operator'],['version_id',operator.eq,1]]
    visdem_data_restr = organizeArray(visdem_data,whatArr_tmp)

    dalt_ids = sorted(set(visdem_data_restr['dalt_id'].values.astype(int)))
    observer_ids = sorted(set(visdem_data_restr['observer_id'].values.astype(int)))
    
    columns=['observer_id', 'observer_coldef_type', 'coldef_type', 'set_id', 'scene_id', 'image_id']
    for dalt_id in dalt_ids:
        col_tmp = "dalt_id_"+str(dalt_id).zfill(2)
        columns.append(col_tmp)
    
    observer_coldef_types = set(visdem_data_restr['observer_coldef_type'].values)
    #print observer_coldef_types
        
    visdem_data_template = pandas.DataFrame(columns=columns)
    visdem_data_RT_paired = visdem_data_template.copy()
    index = 0
    
    
    for observer_id in observer_ids:
        whatArray_tmp = [['observer_id',operator.eq,observer_id]]
        visdem_data_restr_obs = organizeArray(visdem_data_restr, whatArray_tmp)
        
        set_ids = sorted(set(visdem_data_restr_obs['set_id'].values.astype(int)))
        
        for set_id in set_ids:
            whatArray_tmp = [['set_id', operator.eq, set_id]]
            visdem_data_restr_set = organizeArray(visdem_data_restr_obs, whatArray_tmp)
            scene_ids = sorted(set(visdem_data_restr_set['scene_id'].values.astype(int)))

            
            for scene_id in scene_ids:
                whatArray_tmp = [['scene_id',operator.eq,scene_id]]
                visdem_data_restr_scene =organizeArray(visdem_data_restr_set, whatArray_tmp)
                
                pandas_RT_tmp = pandas.DataFrame({'observer_id': observer_id,
                                                  'observer_coldef_type': coldef_type,
                                                  'coldef_type': coldef_type,
                                                  'set_id': set_id,
                                                  'scene_id': scene_id,
                                                  'image_id': 'nix'
                                                  },[index])
                
                for dalt_id in dalt_ids:
                    if dalt_id not in [0,99]:
                        whatArray_tmp = [['dalt_id',operator.eq,dalt_id],['coldef_type',operator.eq,coldef_type]]
                    else: 
                        whatArray_tmp = [['dalt_id',operator.eq,dalt_id]]
                    field = organizeArray(visdem_data_restr_scene, whatArray_tmp).reset_index().loc[0]
                    if not field.empty:
                        RT_tmp = field['resp_time']*1000 if bool(field['is_correct']) else float('NaN')
                        #RT_tmp = field['resp_time']*1000
                        pandas_RT_tmp["dalt_id_"+str(dalt_id).zfill(2)]= float(RT_tmp)
                        pandas_RT_tmp['image_id'] = field['image_id']
                        pandas_RT_tmp['coldef_type'] = field['coldef_type']
                    else:
                        RT_tmp = float('NaN')
                        pandas_RT_tmp["dalt_id_"+str(dalt_id).zfill(2)]= float(RT_tmp)
                    
                visdem_data_RT_paired = visdem_data_RT_paired.append(pandas_RT_tmp)
                index += 1
    # Layout RT for storage in path
    visdem_data_RT_paired = visdem_data_RT_paired[columns]
    visdem_data_RT_paired.observer_id = visdem_data_RT_paired.observer_id.astype(int)
    visdem_data_RT_paired.observer_coldef_type = visdem_data_RT_paired.observer_coldef_type.astype(int)
    visdem_data_RT_paired.set_id = visdem_data_RT_paired.set_id.astype(int)
    visdem_data_RT_paired.scene_id = visdem_data_RT_paired.scene_id.astype(int)
    visdem_data_RT_paired.image_id = visdem_data_RT_paired.image_id.astype(int)
    visdem_data_RT_paired.coldef_type = visdem_data_RT_paired.coldef_type.astype(int)
    visdem_data_RT_paired.to_csv(os.path.join(path,dict['filename']+'_visdem-data-RT-paired.csv'),sep=";")
    
    f = open(os.path.join(path,dict['filename']+'_visdem-data-RT-paired_meta-data.txt'), 'w')
    json.dump(dalt_ids, f)
    f.close()


#import copy

def makePairedRTData(data_RT_paired, methods,id_label):
    """
    Input: * RT Pandas data frame paired
           * IDs of methods to compare
           * Label of columns being compared
    Output: * All possible combinations
            * Labels for the combinations
    """
    
    method_counter = copy.copy(methods)
    
    comparisons = []; labels = [];
    
    for method in methods:
        method_counter.pop(0)
        if method_counter:
            col_tmp = id_label+"_"+str(method).zfill(2)
            values_RT_tmp = data_RT_paired[col_tmp].values
            
            for to_method in method_counter:
                label_tmp = str(method).zfill(2)+'-to-'+str(to_method).zfill(2)
                
                to_col_tmp = id_label+"_"+str(to_method).zfill(2)
                to_values_RT_tmp = data_RT_paired[to_col_tmp].values
                
                difference_paired = values_RT_tmp - to_values_RT_tmp
                comparisons.append(difference_paired)
                labels.append(label_tmp)
    return comparisons, labels

def checkNormality(distributions,labels,path,options,project_str="",plot_types={'q-q','hist','boxplot','residuals'}):
    """
    Similar to samsemPlots35thru37
    """
    if project_str:
        folder_name = project_str.lower().replace(" ","-")+"_normality-check"
        project = project_str
    else:
        folder_name = "new-normality-check"
        project = "New normality check"
    print "* Start checking normality of \'"+str(project)+"\'"
    
    # 1. Check that dimensions of distribtions and labels are the same
    distr_size = numpy.shape(distributions)
    labels_size = numpy.shape(labels)
    try:
        pass
    except:
        print "Error: Dimensions of labels and distributions have to match."
        return
    
    # 2. Check that all plot types are accepted
    try:
        plot_types_acc = plot_types
    except:
        print "Caution: Plot type has to be one of the following: \'q-q\', \'hist\', \'boxplot\' or \'residuals\'"
    
    # 4. Create folder if necessary
    save_to_path = os.path.join(path,folder_name)
    if not os.path.exists(save_to_path): os.makedirs(save_to_path)
        
    # 3. Make q-q plot
    if 'q-q' in plot_types_acc:
        sys.stdout.write("** Making Q-Q plots ")
        save_to_tmp = os.path.join(save_to_path,'q-q')
        if not os.path.exists(save_to_tmp): os.makedirs(save_to_tmp)
        
        i = 0;
        for distribution in distributions:
            nonnan_distribution =  distribution[~numpy.isnan(distribution)]
            sys.stdout.write('.')
            plotQQPlot(nonnan_distribution, save_to_tmp, {"filename":labels[i]})
            i+=1
        
        sys.stdout.write('\n')
    
    # . Make histograms
    if 'hist' in plot_types_acc:
        sys.stdout.write("** Making histograms ")
        save_to_tmp = os.path.join(save_to_path,'hist')
        if not os.path.exists(save_to_tmp): os.makedirs(save_to_tmp)
        
        
        i = 0;
        for distribution in distributions:
            nonnan_distribution =  distribution[~numpy.isnan(distribution)]
            sys.stdout.write('.')
            options.update({"filename":labels[i]})
            plotHistogram(nonnan_distribution, save_to_tmp,options)
            i+=1
        
        sys.stdout.write('\n')
    
    # . Make boxplots
    if 'boxplot' in plot_types_acc:
        
        sys.stdout.write("** Making boxplots .")
        save_to_tmp = os.path.join(save_to_path,'boxplots')
        if not os.path.exists(save_to_tmp): os.makedirs(save_to_tmp)
        
        
        options.update({'filename': project_str.lower().replace(" ","-")})
        plotRTGraphs(distributions, labels, save_to_tmp,options)
    
        sys.stdout.write('\n')
    
    # . Make residuals
    if 'residuals' in plot_types_acc:
        sys.stdout.write("** Making boxplots of the residuals .")
        save_to_tmp = os.path.join(save_to_path,'residuals')
        if not os.path.exists(save_to_tmp): os.makedirs(save_to_tmp)
        
        
        options.update({'filename': project_str.lower().replace(" ","-")})
        plotResidualPlots(distributions, labels, save_to_tmp, options)
        sys.stdout.write('\n')
    
    print "* Stop checking normality"

#import scipy

def signTest(data,path,methods):
    """
    Input: * data:    Pandas data frame with relevant data that should be compared
           * path:    Path, to which the results should be save to as matrix
           * labels:  Name of the columns in the pandas data set, which should be analyzed
    Output
    """
    
    num_methods = numpy.shape(methods)[0]
    
    # Remove all rows that contain nan
    data_wonan = data.dropna()
    
    method_counter = copy.copy(methods)
    result_array_template = numpy.chararray(num_methods)
    result_array_template[:] = "x"
    result_array_template = numpy.array(result_array_template)
    
    template = pandas.DataFrame(columns=methods)
    template.loc[0] = result_array_template
    matrix = pandas.DataFrame(columns=methods)
    
    for method in methods:
        
        method_counter.pop(0)
        values = data_wonan[method].values
        
        curr_row = pandas.DataFrame.copy(template)
        if method_counter:
            for to_method in method_counter:
                
                # Get current to_values
                to_values = data_wonan[to_method].values
                
                diff_values = values-to_values # Make difference
                num_positive = sum(diff_values>0) # Find all the values that are greater than zero
                num_total = numpy.shape(diff_values)[0]
                
                p_value = stats.binom_test(num_positive,num_total)
                curr_row[to_method] = p_value
        matrix = matrix.append(curr_row)
        #sys.stdout.write("\n")
    matrix.index = methods
    matrix.to_csv(os.path.join(path,"sign_test_p-values.csv"),sep=';')

def makeKruskalWallisTest(data,path,methods):
    """
    Input: * data:    Pandas data frame with relevant data that should be compared
           * path:    Path, to which the results should be save to as matrix
           * methods:  Name of the columns in the pandas data set, which should be analyzed
    Output
    """
    
    num_methods = numpy.shape(methods)[0]
    
    # Remove all rows that contain nan
    data_wonan = data.dropna()
    
    method_counter = copy.copy(methods)
    result_array_template = numpy.chararray(num_methods)
    result_array_template[:] = "x"
    result_array_template = numpy.array(result_array_template)
    template = pandas.DataFrame(columns=methods)
    template.loc[0] = result_array_template
    
    matrix = pandas.DataFrame(columns=methods)
    
    for method in methods:
        
        method_counter.pop(0)
        values = data_wonan[method].values
        
        curr_row = pandas.DataFrame.copy(template)
        if method_counter:
            for to_method in method_counter: # Get current to_values
                to_values = data_wonan[to_method].values
                H,p_value = stats.mstats.kruskalwallis(values,to_values)
                curr_row[to_method] = p_value
        matrix = matrix.append(curr_row)
        #sys.stdout.write("\n")
    matrix.index = methods
    matrix.to_csv(os.path.join(path,"kruskal-wallis-test_p-values.csv"),sep=';')

def makeMedianTest(data,path,methods,dict):
    """
    Input: * data:    Numpy data array with relevant data that should be compared
           * path:    Path, to which the results should be save to as matrix
           * labels:  Name of the columns in the pandas data set, which should be analyzed
    Output
    """
    
    num_methods = numpy.shape(methods)[0]
    num_columns = numpy.shape(data)[0]
    if not num_methods == num_columns:
        print "Error: Number of columns does not match the labels for the median test. " + "Expected columns: %i, actual columns %i" % (num_methods, num_columns)
        return
    
    if dict.has_key('filename'):
        filename_csv = dict['filename']+"_median-test_p-values.csv"
        filename_latex = dict['filename']+"_median-test_p-values.tex"
    else:
        filename_csv = "_median-test_p-values.csv"
    
    range_methods = range(num_methods)
    
    method_counter = copy.copy(range_methods)
    result_array_template = numpy.chararray(num_methods)
    result_array_template[:] = "x"
    result_array_template = numpy.array(result_array_template)
    
    template = pandas.DataFrame(columns=methods)
    template.loc[0] = result_array_template
    matrix = pandas.DataFrame(columns=methods)
    
    for method in range_methods:
        method_counter.pop(0)
        values = data[method]
        
        curr_row = pandas.DataFrame.copy(template)
        if method_counter:
            for to_method in method_counter: # Get current to_values
                to_values = data[to_method]
                
                if (len(values) !=0) and (len(to_values) != 0):
                    stat, p_value, m, table =  stats.median_test(values,to_values)
                else:
                    p_value = "Nix"
                curr_row[methods[to_method]] = p_value
        matrix = matrix.append(curr_row)
    matrix.index = methods
    matrix = matrix.drop(matrix.index[[num_methods-1]])
    matrix = matrix[methods[1:num_methods]]
    matrix.to_csv(os.path.join(path,filename_csv),sep=';')

    writePandastoLatex(matrix,os.path.join(path,filename_latex))

def preparePandas4AccuracyPlots(pandas_dict):
    
    accuracies = {}
    
    for key in pandas_dict:
        ACC_tmp = getAccuracy(pandas_dict[key])
        ACC_tmp.append(key)
        accuracies.update({key: ACC_tmp})
        
    return accuracies
        
def preparePandas4Chi2(pandas_dict,order_dict):
    data = {}; corrArr = []; uncorrArr = []; labels_array = []
    for i in range(len(order_dict)):
        key = order_dict[i]
        corr_tmp = numpy.shape(pandas_dict[key][pandas_dict[key]['is_correct']==True]['resp_time'].values)[0]
        uncorr_tmp = numpy.shape(pandas_dict[key][pandas_dict[key]['is_correct']==False]['resp_time'].values)[0]
        corrArr.append(corr_tmp)
        uncorrArr.append(uncorr_tmp)
        labels_array.append(key)
        data.update({key: numpy.array([corr_tmp, uncorr_tmp]).astype(int)})
    
    obs_array = numpy.array([corrArr, uncorrArr])
    obs_pandas = pandas.DataFrame(data=data, index=['correct','uncorrect'])[labels_array]
    
    return obs_array, obs_pandas



def preparePandas4RTPlots(pandas_dict,order_dict):
    
    boxes = []; labels = []
    for i in range(len(order_dict)):
        key = order_dict[i]
        
        values_tmp = pandas_dict[key][pandas_dict[key]['is_correct']==True]['resp_time'].values*1000; 
        labels.append(key) if values_tmp.size else labels.append(key + ' - No data'); 
        boxes.append(values_tmp)
        
    return boxes, labels