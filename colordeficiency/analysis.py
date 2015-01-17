'''
Created on 26. aug. 2014

@author: joschua
'''

import numpy
import matplotlib.pyplot as plt
import pandas
import os
import sys
import operator
#from colordeficiency import colordeficiency_tools
#from colordeficiency_tools import getAllXXXinPath, getStatsFromFilenamet

from analysis_tools import plotAccuracyGraphs, plotCIAverageGraphs, plotRTGraphs, getSetFromScene, getAccuracy, getCIAverage, organizeArray, extractDataFromPsychoPyXLSX
from colordeficiency import settings
#print colordeficiency.settings.coldef_toolbox_path

def extractExperimentData(dataSheet,dataArray=pandas.DataFrame(),testArray=pandas.DataFrame()):
    """
    This function takes the url an xlsx file containing the result data from a sample-to-match experiment, extracts the data and returns the contained information as numpy array.
    Is this the same as analyze SaMSEMData? What does it do?
    """
    
    # Get all the sheet names for the excel file
    
    sheet_names = pandas.ExcelFile(dataSheet).sheet_names
    #print sheet_names
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
    ext = 'xlsx'; xlsx_files = colordeficiency_tools.getAllXXXinPath(path_in,ext)
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
    for item in colordeficiency.settings.colDefLong2ID:
        dataArray.loc[dataArray['coldef_type'] == item, ['coldef_type']] = colordeficiency.settings.colDefLong2ID[item]
    
    if experiment_type == "sample-2-match":
        for item in colordeficiency.settings.sim2ID:
            dataArray.loc[dataArray['sim_id'] == item, ['sim_id']] = colordeficiency.settings.sim2ID[item]
        
        dataArray = dataArray.rename(columns={'sim_id': 'sim_id',
                              'coldef_type': 'coldef_type',
                              'resp.corr_raw': 'is_correct',
                              'resp.rt_raw': 'resp_time',
                              'origFile': 'filepath'})
            
        for index, row in dataArray.iterrows():
            path_tmp = row['filepath']
            filename = os.path.basename(path_tmp).split('.')[0]
            dict_tmp = colordeficiency_tools.getStatsFromFilename(filename)
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
    ext = 'xlsx'; xlsx_files = colordeficiency_tools.getAllXXXinPath(path_in,ext)
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
    #print testArray    
    
    # 2. Step: Adapt values to programstandards
    for item in colordeficiency.settings.colDefLong2ID:
        dataArray.loc[dataArray['coldef_type'] == item, ['coldef_type']] = colordeficiency.settings.colDefLong2ID[item]
    
    for item in colordeficiency.settings.dalt2ID:
        dataArray.loc[dataArray['dalt_id'] == item, ['dalt_id']] = colordeficiency.settings.dalt2ID[item]
        
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
        #print filename
        dict_tmp = colordeficiency_tools.getStatsFromFilename(filename)
        imgID_tmp = int(dict_tmp['img_id'])
        #print index
        
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

def writeMetaDataOfExperiments(experimentData,dict):
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
        
    if exp_type == "visdem":
        f = open(path_out,'w+')
        f.write("ViSDEM meta data\n")
        sessions = set(experimentData['session_id'])
        f.write('... # of sessions: '+str(len(sessions))+' ; '+str(sessions)+'\n')
        sets = set(experimentData['set_id'])
        f.write('... # of sets: '+str(len(sets))+' ; '+str(sets)+'\n')
        daltonizations = set(experimentData['dalt_id'])
        f.write('... # of daltonization ids: '+str(len(daltonizations))+' ; '+str(daltonizations)+'\n')
        observers = set(experimentData['observer_id'])
        f.write("... # of observers: " +str(len(observers))+' ; '+str(observers)+'\n')
        observers_norm = set(experimentData[experimentData['observer_coldef_type']==0]['observer_id'])
        f.write("...... # of normal observers: "+str(len(observers_norm))+' ; '+str(observers_norm)+'\n')
        observers_prot = set(experimentData[experimentData['observer_coldef_type']==1]['observer_id'])
        f.write("...... # of protan observers: "+str(len(observers_prot))+' ; '+str(observers_prot)+'\n')  
        observers_deut = set(experimentData[experimentData['observer_coldef_type']==2]['observer_id'])
        f.write("...... # of deutan observers: "+str(len(observers_deut))+' ; '+str(observers_deut)+'\n')         
        f.close()
    elif exp_type == 'samsem':
        f = open(path_out,'w+')
        f.write("SaMSEM meta data\n")
        sessions = set(experimentData['session_id'])
        f.write('... # of sessions: '+str(len(sessions))+' ; '+str(sessions)+'\n')
        images = set(experimentData['image_id'])
        f.write('... # of images: '+str(len(images))+' ; '+str(images)+'\n')
        simulations = set(experimentData['sim_id'])
        f.write('... # of simulations: '+str(len(simulations))+' ; '+str(simulations)+'\n')
        observers = set(experimentData['observer_id'])
        f.write("... # of observers: " +str(len(observers))+' ; '+str(observers)+'\n')
        observers_norm = set(experimentData[experimentData['observer_coldef_type']==0]['observer_id'])
        f.write("...... # of normal observers: "+str(len(observers_norm))+' ; '+str(observers_norm)+'\n')
        observers_prot = set(experimentData[experimentData['observer_coldef_type']==1]['observer_id'])
        f.write("...... # of protan observers: "+str(len(observers_prot))+' ; '+str(observers_prot)+'\n')
        observers_deut = set(experimentData[experimentData['observer_coldef_type']==2]['observer_id'])
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
    
    sys.stdout.write("Starting Res#"+str(dict['result_id'])+'.')    
    if not os.path.exists(os.path.join(path,str(dict['result_id']))): os.makedirs(os.path.join(path,str(dict['result_id'])))
    
    # 1. Retrieving data from data set
    coldef_type = dict['coldef_type']
    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],dict['obs_operator']];howArr_tmp = []; # Only observations that use simulations of the chosen color deficiency are needed 
    samsem_data_reduced =  organizeArray(samsem_data, whatArr_tmp,howArr_tmp)
    
    boxes = []; labels = []; accuracies = {}; labels_acc_tmp = []; mean_cis = {}; order = {}; i=1
    algorithm_ids = set(samsem_data['sim_id'].values.astype(int))
    
    # 2. Retrieving data from each algorithm
    for algorithm_id in algorithm_ids:
        
        if algorithm_id != 99:
            whatArr_tmp = [['sim_id',operator.eq,algorithm_id]];howArr_tmp=[];
            relevant_data_tmp = organizeArray(samsem_data_reduced,whatArr_tmp,howArr_tmp)   
            
            # 3. Get response time data
            alg_values = relevant_data_tmp[relevant_data_tmp['is_correct']==True]['resp_time'].values*1000; 
            boxes.append(alg_values)
            
            labels.append(settings.id2Sim[algorithm_id]) if alg_values.size else labels.append(str(algorithm_id) + ' - No data'); 
            
            # 4. Get CI data
            alg_mean = getCIAverage(alg_values);
            alg_mean.append(labels[i-1])
            mean_cis.update({algorithm_id: alg_mean})
            
            # 5. Get accuracy data
            alg_acc = getAccuracy(relevant_data_tmp)
            alg_acc.append(labels[i-1])
            accuracies.update({algorithm_id: alg_acc})
            
            order.update({i:algorithm_id})
            i += 1
    
    # 6. Plot response time data
    dict.update({'y_lim': [0,1750]})
    plotRTGraphs(boxes,labels,path, dict, order)
    # 7. Plot mean data
    plotCIAverageGraphs(mean_cis,path,dict,order)
    # 8. Plot accuracy data
    dict.update({'y_lim': [.55,.9]})
    plotAccuracyGraphs(accuracies,path,dict,order)
    
def samsemPlots9thru12(samsem_data,path,dict):
    """
    Literally identical to samsemPlots1thru4. Check documentation there.
    """
    samsemPlots1thru4(samsem_data,path,dict)

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
    
    result_id = dict['result_id']   
    if not os.path.exists(os.path.join(path,str(result_id))): os.makedirs(os.path.join(path,str(result_id)))
    
    intro_string = result_id if result_id else dict['filename'] 
    sys.stdout.write("Starting Res#"+str(intro_string)+'.') 
    
    boxes = []; accuracies = {}; mean_cis = {}; labels = []
    coldef_type = dict['coldef_type']
    
    # Ignore dummy algorithm
    whatArr_tmp = [['sim_id',operator.ne,99],];howArr_tmp=[]
    samsem_data = organizeArray(samsem_data,whatArr_tmp,howArr_tmp)
    sim_ids = set(samsem_data['sim_id'].values.astype(int))
    print sim_ids
    
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
    
    # 2. Get response time data
    # For normal sighted observers
    normal_response_values = normal_data[normal_data['is_correct']==True]['resp_time'].values*1000; labels.append('Norm.obs.') if normal_response_values.size else labels.append('Norm.obs. - No data'); 
    boxes.append(normal_response_values)
    # For protan observers
    protan_response_values = protan_data[protan_data['is_correct']==True]['resp_time'].values*1000; labels.append('Prot.obs.') if protan_response_values.size else labels.append('Prot.obs. - No data'); 
    boxes.append(protan_response_values)
    # For deutan observers
    deutan_response_values = deutan_data[deutan_data['is_correct']==True]['resp_time'].values*1000; labels.append('Deut.obs.') if deutan_response_values.size else labels.append('Prot.obs. - No data'); 
    boxes.append(deutan_response_values)
    
    # 4. Get CI data
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
    
    # 5. Get accuracy data
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
    
    # 6. Save data in file
    for_bruno = pandas.DataFrame(boxes)
    for_bruno = for_bruno.rename({0:labels[0],1:labels[1],2:labels[2]})
    #for_bruno.to_csv(os.path.join(path,str(dict['result_id']),dict['filename']+".csv"),sep=';')
    
    # 7. Plot response time data
    dict.update({'y_lim': [0,3000]})
    plotRTGraphs(boxes,labels,path, dict)
    # 8. Plot mean data
    order={1:'norm', 2:'protan', 3:'deutan'}
    plotCIAverageGraphs(mean_cis,path,dict,order)
    # 9. Plot accuracy data
    plotAccuracyGraphs(accuracies,path,dict,order)

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
    
    #print samsem_data
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


    
def vsplots67thru70(visual_search_data,path,dict):
    
    result_id = dict['result_id']
    intro_string = result_id if result_id else dict['filename'] 
    sys.stdout.write("Starting Res#"+str(intro_string))    
    if not os.path.exists(os.path.join(path,str(dict['result_id']))): os.makedirs(os.path.join(path,str(dict['result_id'])))
    
    DNData = pandas.DataFrame(); DEData = pandas.DataFrame(); DCData = pandas.DataFrame()
    if dict.has_key('sets'):
        set_ids = dict['sets']
    else:
        set_ids = []
    
    # 1. Retrieving data from the image versions
    for set_id in set_ids:
        whatArr_tmp = [['version_id',operator.eq,0],['dalt_id',operator.eq,0],dict['obs_operator'],['set_id',operator.eq,set_id]];howArr_tmp = []
        DNData_tmp =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        DNData = pandas.concat([DNData_tmp,DNData])
            
        whatArr_tmp = [['version_id',operator.eq,2],['dalt_id',operator.eq,0],dict['obs_operator'],['set_id',operator.eq,set_id]];howArr_tmp = []
        DEData_tmp = organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        DEData = pandas.concat([DEData_tmp,DEData])
        
        whatArr_tmp = [['version_id',operator.eq,1],['dalt_id',operator.eq,0],dict['obs_operator'],['set_id',operator.eq,set_id]];howArr_tmp = []
        DCData_tmp = organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        DCData = pandas.concat([DCData_tmp,DCData])
    
    # 2. Plot reponse times
    boxes_tmp = [];labels_tmp = []
    DND_values = DNData[DNData['is_correct']==True]['resp_time'].values*1000; labels_tmp.append('DN') if DND_values.size else labels_tmp.append('DN - No data'); boxes_tmp.append(DND_values)
    DCD_values = DCData[DCData['is_correct']==True]['resp_time'].values*1000; labels_tmp.append('DC') if DCD_values.size else labels_tmp.append('DC - No data'); boxes_tmp.append(DCD_values) 
    DED_values = DEData[DEData['is_correct']==True]['resp_time'].values*1000; labels_tmp.append('DE') if DED_values.size else labels_tmp.append('DE - No data'); boxes_tmp.append(DED_values)
            
    plt.figure(); plt.boxplot(boxes_tmp, notch=1)
    plt.xticks([1,2,3],labels_tmp); plt.title(dict['obs_title']); plt.ylabel('Response time (ms)'); 
    plt.ylim([0,3000]); plt.grid(axis='y')
    #plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-R.eps"));
    plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-R.pdf")); 
    plt.close()
    
    # 3. Save data in file
    for_bruno = pandas.DataFrame(boxes_tmp)
    for_bruno = for_bruno.rename({0:labels_tmp[0],1:labels_tmp[1],2:labels_tmp[2]})
    #for_bruno.to_csv(os.path.join(path,str(dict['result_id']),dict['filename']+".csv"),sep=';')
    
     
    # 4. Accuracy plots
    dn_acc = getAccuracy(DNData)
    dn_acc.append(labels_tmp[0])
    dc_acc = getAccuracy(DCData)
    dc_acc.append(labels_tmp[1])
    de_acc = getAccuracy(DEData)
    de_acc.append(labels_tmp[2])
    accuracies = {'DN': dn_acc,
                  'DC': dc_acc,
                  'DE': de_acc}
    plotAccuracyGraphs(accuracies,path,dict,order={1:'DN', 2:'DC', 3:'DE'})
        
    sys.stdout.write(".")

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
    


  
    


# def plotSample2MatchData(path,sample2MatchDataPath):
#     
#     """
#     Input:  * path                 - Path where the plots should be saved
#             * sample2MatchDataPath - Path to the CSV-file containing Pandas array with the sample-2-match data.
#     """
#     
#     sample2match_data = pandas.read_csv(sample2MatchDataPath,sep=';')
#     """
#     dict = {'result_id': 1, 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All norm.sight.obs. on prot.', 'filename': '9919990999-a' }; s2mplots1thru4(sample2match_data, path, dict)
#     dict = {'result_id': 2, 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All norm.sight.obs. on deut.', 'filename': '9929990999-a' }; s2mplots1thru4(sample2match_data, path, dict)
#     dict = {'result_id': 3, 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.gt,0], 'obs_title': 'All coldef.obs. on prot.', 'filename': '9919999999-a' }; s2mplots1thru4(sample2match_data, path, dict)
#     dict = {'result_id': 4, 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.gt,0], 'obs_title': 'All coldef.obs. on deut.', 'filename': '9929999999-a' }; s2mplots1thru4(sample2match_data, path, dict)
#     
#     dict = {'result_id': 9, 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.eq,1], 'obs_title': 'All prot.obs. on prot.', 'filename': '9919991999-a' }; s2mplots1thru4(sample2match_data, path, dict)
#     dict = {'result_id': 10, 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.eq,1], 'obs_title': 'All prot.obs. on deut.', 'filename': '9929991999-a' }; s2mplots1thru4(sample2match_data, path, dict)
#     dict = {'result_id': 11, 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All deut.obs. on prot.', 'filename': '9919992999-a' }; s2mplots1thru4(sample2match_data, path, dict)
#     dict = {'result_id': 12, 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All deut.obs. on deut.', 'filename': '9929992999-a' }; s2mplots1thru4(sample2match_data, path, dict)
#     """
#     dict = {'result_id': 29, 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All img.s for all obs.s sim.prot.', 'filename': 'C-9919999999' }; s2mplots29and30(sample2match_data, path, dict)
#     dict = {'result_id': 30, 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All img.s for all obs.s sim.deut.', 'filename': 'C-9929999999' }; s2mplots29and30(sample2match_data, path, dict)
#     
  

# def extractVisualSearchData(dataSheet,dataArray=pandas.DataFrame(),testArray=pandas.DataFrame()):
#     """
#     This function takes the url an xlsx file containing the result data from a visual search experiment, extracts the data and returns the contained information as numpy array.
#     Input:
#     * dataSheet:     URL to XLSX data sheet containing the information.
#     * dataArray:     Pandas data array containing information from previous experiments, to which the new data is concatenated. Empty array by default.
#     Output:
#     * dataArray:     Pandas data array containing the new relevant information (plus the optional previously existed data).
#     * testArray:     Pandas data array containing the new colordeficiency_tools information (plus the optional previously existed colordeficiency_tools data)
#     * extraDataDict: Dictionary containing the additional data from the experiments
#     """
#     
#     #Get all the sheet names from the excel file
#     sheet_names = pandas.ExcelFile(dataSheet).sheet_names
#     # Create empty array for colordeficiency_tools and relevant datam create empty dictionary for the extra data
#     if dataArray.empty: dataArray = pandas.DataFrame(); 
#     if testArray.empty: testArray = pandas.DataFrame(); 
#     extraDataDict= {}
#     # Read file sheet by sheet
#     for sheet_name in sheet_names:    
#         if not "~" in sheet_name:
#             if ('practSets' in sheet_name) or ('sets' in sheet_name): pass # Ignore data about the order of sets and samples for now
#             else:
#                 # Read excel sheet and extract data
#                 excel_sheet = pandas.read_excel(dataSheet, sheet_name)
#                 array_tmp, extraDataDict_tmp = extractDataFromPsychoPyXLSX(excel_sheet)
#                 # Update dictionary containing extra data
#                 extraDataDict.update(extraDataDict_tmp)
#                 # Append colordeficiency_tools or relevant data to respective data array
#                 if 'practTrials' in sheet_name: testArray = pandas.concat([array_tmp,testArray]) if not testArray.empty else array_tmp
#                 else: dataArray = pandas.concat([array_tmp,dataArray]) if not dataArray.empty else array_tmp
#     
#     testArray = testArray.reset_index() 
#     dataArray = dataArray.reset_index()
#     
#     return dataArray, testArray, extraDataDict

  
 

# def plotVisualSearchData(path,visualSearchDataPath):
#     
#     """
#     Input:  * path                 - Path where the plots should be saved
#             * visualSearchDataPath - Path to the CSV-file containing Pandas array with the visual search data.
#     """
#     
#     visual_search_data = pandas.read_csv(visualSearchDataPath,index_col=False,header=0,sep=';')
#     #print visual_search_data
#     
#     """
#     # Compute img#1-4 from https://docs.google.com/document/d/1w305_EUYiReLrQ34H-teJwsNICXvRK-cv7E6TgVfukM/edit#heading=h.khuve23asvax
#     dict = {'result_id': 1, 'obs_coldef_type': str(0), 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All norm.sight.obs.' }; vsplots1thru4(visual_search_data, path, dict)
#     dict = {'result_id': 2, 'obs_coldef_type': str(9), 'obs_operator': ['observer_coldef_type',operator.gt,0], 'obs_title': 'All col.def.obs.' }; vsplots1thru4(visual_search_data, path, dict) 
#     dict = {'result_id': 3, 'obs_coldef_type': str(1), 'obs_operator': ['observer_coldef_type',operator.eq,1], 'obs_title': 'All prot.obs.' }; vsplots1thru4(visual_search_data, path, dict)
#     dict = {'result_id': 4, 'obs_coldef_type': str(2), 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All deut.obs.' }; vsplots1thru4(visual_search_data, path, dict)
#     
#     # Compute img#5-8 from https://docs.google.com/document/d/1w305_EUYiReLrQ34H-teJwsNICXvRK-cv7E6TgVfukM/edit#heading=h.khuve23asvax
#     dict = {'result_id': 5, 'obs_coldef_type': str(0), 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All norm.sight.obs.' }; vsplots5thru8(visual_search_data, path, dict)
#     dict = {'result_id': 6, 'obs_coldef_type': str(9), 'obs_operator': ['observer_coldef_type',operator.gt,0], 'obs_title': 'All col.def.obs.' }; vsplots5thru8(visual_search_data, path, dict) 
#     dict = {'result_id': 7, 'obs_coldef_type': str(1), 'obs_operator': ['observer_coldef_type',operator.eq,1], 'obs_title': 'All prot.obs.' }; vsplots5thru8(visual_search_data, path, dict)
#     dict = {'result_id': 8, 'obs_coldef_type': str(2), 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All deut.obs.' }; vsplots5thru8(visual_search_data, path, dict)
#     """
#     
#     # Compute img#67-70 from https://docs.google.com/document/d/1w305_EUYiReLrQ34H-teJwsNICXvRK-cv7E6TgVfukM/edit#heading=h.khuve23asvax
#     dict = {'result_id': 67, 'obs_coldef_type': str(0), 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All sets for all norm.sight.obs.', 'filename': 'V-00099909999999' }; vsplots67thru70(visual_search_data, path, dict)
#     dict = {'result_id': 68, 'obs_coldef_type': str(9), 'obs_operator': ['observer_coldef_type',operator.gt,0], 'obs_title': 'All sets for all col.def.obs.', 'filename': 'V-00099999999999' }; vsplots67thru70(visual_search_data, path, dict)
#     dict = {'result_id': 69, 'obs_coldef_type': str(1), 'obs_operator': ['observer_coldef_type',operator.eq,1], 'obs_title': 'All sets for all prot.obs.', 'filename': 'V-00099919999999' }; vsplots67thru70(visual_search_data, path, dict)
#     dict = {'result_id': 70, 'obs_coldef_type': str(2), 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All sets for all deut.obs.', 'filename': 'V-00099929999999' }; vsplots67thru70(visual_search_data, path, dict)
#     

    
# def plotEI2015_SaMSEM_ViSDEM():
#     path = "/Users/thomas/Dropbox/01_NZT/01_PhD/02_Conferences/EI-2015_Color-Imaging-XX/01_Artikler/02_Evaluation-methods/images/"
#     
#     # Plots for ViSDEM
#     visualSearchDataPath = '../colordeficiency-data/visual-search-data.csv'
#     visual_search_data = pandas.read_csv(visualSearchDataPath,index_col=False,header=0,sep=';')
#     dict = {'result_id': '', 'obs_coldef_type': str(0), 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All sets for all norm.sight.obs.', 'filename': 'visdem-normal-observers' }; vsplots67thru70(visual_search_data, path, dict) #Res#67
#     dict = {'result_id': '', 'obs_coldef_type': str(2), 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All sets for all deut.obs.', 'filename': 'visdem-deutan-observers' }; vsplots67thru70(visual_search_data, path, dict) #Res#70
#     
#     # Plots for SaMSEM
#     sample2MatchDataPath = '../colordeficiency-data/sample-2-match-data.csv'
#     sample2match_data = pandas.read_csv(sample2MatchDataPath,index_col=False,sep=';')
#     dict = {'result_id': '', 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All img.s for all obs.s sim.prot.', 'filename': 'samsem-protanopia' }; s2mplots29and30(sample2match_data, path, dict) #Res#29
#     dict = {'result_id': '', 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All img.s for all obs.s sim.deut.', 'filename': 'samsem-deuteranopia' }; s2mplots29and30(sample2match_data, path, dict) #Res#30
#     
    
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
