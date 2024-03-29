'''
Created on 26. aug. 2014

@author: joschua
'''

import copy, json, matplotlib.pyplot as plt, numpy, operator, os, pandas, sys

from scipy import stats

from .colordeficiency import settings
from .colordeficiency_tools import getAllXXXinPath, getStatsFromFilename, getSetFromScene, keys2values, getColDefTypeForObserver
from .analysis_tools import writeMetaDataOfExperiments, extractExperimentData, writePandastoLatex, makePearsonChi2Contingency, makePearsonChi2Contingency2x2Test, plotQQPlot, plotResidualPlots, plotHistogram, plotAccuracyGraphs, plotCIAverageGraphs, plotRTGraphs, getAccuracy, getCIAverage, organizeArray, extractDataFromPsychoPyXLSX
from .analysis_tools import preparePandas4AccuracyPlots, preparePandas4Chi2, preparePandas4RTPlots
from .analysis_tools import makeMedianTest, makePearsonChi2Contingency, makePearsonChi2Contingency2x2Test, makePairwiseStudentTTest
from .analysis_tools import makePairedRTDataArray

from PyPDF2 import PdfFileReader, PdfFileMerger

def analyzeSaMSEMData(dict):
    """
    This functions analyzes the data from the sample-2-match experiment as can be found in the path.
    Before: analyzeSample2MatchData
    """
    
    if 'path_in' in dict:
        path_in = dict['path_in']
    else:
        print("Caution: No path for input folder containing the data has been defined. Please define path to folder by dict['path_in']=path_in")    
        return
    
    path_out_default = '../colordeficiency-data/'    
    if 'path_out' in dict:
        path_out = dict['path_out']
    else:
        print("Caution: No path for output folder where the data should be stored has been defined. Using default output path instead: "+str(path_out_default))
        path_out = path_out_default
    
    
    path = os.path.join(os.path.dirname(os.path.abspath(os.path.join(__file__,os.pardir))),'colordeficiency-data')
    
    # 0. Step: Get all the relevant information, i.e. obs_col_defs etc.
    observer_ids = os.path.join(path,"observer_ids.csv")
    obs_ids_sheet = pandas.read_csv(observer_ids,sep=";")
    
    # 1. Step: Read all the XLSX data in the path
    ext = 'xlsx'; xlsx_files = getAllXXXinPath(path_in,ext)
    dataArray = pandas.DataFrame()
    i=1
    for xlsx_file in xlsx_files:
        if not '~' in xlsx_file:
            sys.stdout.write(xlsx_file)
            dataArray_tmp, testArray, extraDataDict = extractExperimentData(os.path.join(path_in,xlsx_file))
            
            experiment_type = extraDataDict['expName'] if 'expName' in extraDataDict else 'none'
            if experiment_type == "sample-2-match":
                newDataArray = dataArray_tmp[['sim_id','coldef_type','resp.corr_raw','resp.rt_raw','origFile']]
                
            if '0. Participant ID' in extraDataDict:
                obsID = int(extraDataDict['0. Participant ID'])
                newDataArray['observer_id'] = obsID
                obs_coldef_type = obs_ids_sheet.loc[obs_ids_sheet['observer_id']==obsID,['observer_coldef_type']]
                newDataArray['observer_coldef_type'] = int(obs_coldef_type['observer_coldef_type'])
            
            if "2. Session" in extraDataDict:
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
            
        dataArray.is_correct = dataArray.is_correct.astype(bool)
        dataArray.image_id = dataArray.image_id.astype(int)
        dataArray = dataArray[['image_id','sim_id','coldef_type','is_correct','resp_time','observer_id','observer_coldef_type','session_id','filepath']]
        
    elif experiment_type == "visual-search":
        pass
       
    
    # 3. Saving data to file
    try:
        sys.stdout.write("Starting to save ... ")
        if experiment_type == "sample-2-match":
            dataArray.to_csv(os.path.join(path_out,'samsem-data.csv'),sep=";")
            sys.stdout.write("Success: Sample-to-match data successfully saved in '"+str(path_out)+"'.\n")
        elif experiment_type == "visual-search":
            dataArray.to_csv(os.path.join(path_out,'visdem-data.csv'),sep=";")
            sys.stdout.write("Visual-search data successfully saved.")
        else:
            sys.stdout.write("Caution: No data saved.")
    except Exception as e:
        print(e) 


def analyzeViSDEMData(dict):
    """
    This function analyzes the data from the visual searach experiment as can be found in the path.
    Before: analyseVisualSearchData
    
    """
    
    if 'path_in' in dict:
        path_in = dict['path_in']
    else:
        print("Caution: No path for input folder containing the data has been defined. Please define path to folder by dict['path_in']=path_in")    
        return
    
    path_out_default = '../colordeficiency-data/'    
    if 'path_out' in dict:
        path_out = dict['path_out']
    else:
        print("Caution: No path for output folder where the data should be stored has been defined. Using default output path instead: "+str(path_out_default))
        path_out = path_out_default
    
    if 'round' in dict:
        round = dict['round']
    else:
        print("Error: You have to chose a round first.")
    
    path = os.path.join(os.path.dirname(os.path.abspath(os.path.join(__file__,os.pardir))),'colordeficiency-data')
        
    # 0. Step: Get all the relevant information, i.e. motive_ids, obs_col_defs etc.
    if round == 1:
        visualsearch_ids = os.path.join(path,"visualsearch_ids.csv")
    elif round == 2:
        visualsearch_ids = os.path.join(path,"visualsearch_ids_2.csv")
        
    vs_ids_sheet = pandas.read_csv(visualsearch_ids,sep=";")
    
    # Get all the relevant information about the observers, i.e. obs_col_defs etc.
    observer_ids = os.path.join(path,"observer_ids.csv")
    obs_ids_sheet = pandas.read_csv(observer_ids,sep=";")
    
    # 1. Step: Read all the XLSX data in the path
    ext = 'xlsx'; xlsx_files = getAllXXXinPath(path_in,ext)
    dataArray = pandas.DataFrame()
    i=1
    for xlsx_file in xlsx_files:
        sys.stdout.write(xlsx_file)
        dataArray_tmp, testArray, extraDataDict = extractExperimentData(os.path.join(path_in,xlsx_file))
        
        newDataArray = dataArray_tmp[['dalt_id','coldef_type','resp.corr_raw','resp.rt_raw','stimFile']]
        
        if "2. Session" in extraDataDict:
            sessionID = int(extraDataDict['2. Session'])
        newDataArray['session_id'] = sessionID
        
        if 'group' in extraDataDict:
            obsGroup = str(extraDataDict['group'])
        newDataArray['obsGroup'] = obsGroup
            
        if '0. Participant ID' in extraDataDict:
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
    
    dataArray.loc[dataArray['dalt_id'] == 'none', ['dalt_id']] = 0
    
        
    dataArray = dataArray.rename(columns={'dalt_id': 'dalt_id',
                              'coldef_type': 'coldef_type',
                              'resp.corr_raw': 'is_correct',
                              'resp.rt_raw': 'resp_time',
                              'stimFile': 'filepath'})
    dataArray = dataArray.reset_index()
    
    # Adding set_id, motive_id and variant_id to each file
    for index, row in dataArray.iterrows():
        path_tmp = row['filepath']
        filename = os.path.basename(path_tmp).split('.')[0]
        dict_tmp = getStatsFromFilename(filename)
        imgID_tmp = int(dict_tmp['img_id'])
        
        tempVSDataArray = vs_ids_sheet.loc[vs_ids_sheet['image_id']==imgID_tmp,['set_id','motive_id','variant_id']]
        
        dataArray.at[index,'image_id'] = imgID_tmp
        dataArray.ix[index,'set_id'] = int(tempVSDataArray['set_id'])
        dataArray.ix[index,'motive_id'] = int(tempVSDataArray['motive_id'])
        dataArray.ix[index,'variant_id'] = int(tempVSDataArray['variant_id'])

    dataArray.image_id = dataArray.image_id.astype(int)
    dataArray.set_id = dataArray.set_id.astype(int)
    dataArray.motive_id = dataArray.motive_id.astype(int)
    dataArray.variant_id = dataArray.variant_id.astype(int)
    dataArray.is_correct = dataArray.is_correct.astype(bool)
    
    dataArray = dataArray[['image_id','set_id','motive_id','variant_id','dalt_id','coldef_type','is_correct','resp_time','observer_id','observer_coldef_type','session_id','filepath','obsGroup']]
    
    # 3. Saving data to file
    try:
        dataArray.to_csv(os.path.join(path_out, 'visdem-data.csv'),sep=";")
        sys.stdout.write("Success: ViSDEM data successfully saved in '"+str(path_out)+"'.\n")
    except Exception as e:
        print(e)  



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
    Comparing RT and ACC data of different simulation methods of one specific observer groups and their corresponding color deficiency simulation type. Color deficiency type of the observer group matches the color deficiency type of the simulation method.
    All images simulating a specific color deficiency are collapsed.
    On the x-axis, we have the the individual simulation methods.
    Input: * samsem_data:    SaMSEM data that is to be analyzed.
           * path:           Path to the folder, where the result images should be stored. 
           * dict:           A dictionary containing different options.
                             ** coldef_type:      The color deficiency type of the simulation method being used.
                             ** filename:         Name of the file as which the result image should be stored.
                             ** method_ids:       IDs of methods that are included in the analyis. By default: All available simulation methods included in samsem_data. 
                             ** subfolder:        Subfolder in path in which the images should be stored.
    Plots the SaMSEM Res#29+30.
    Before: s2mplots29and30
    """
    
    telleoevelse = dict['telleoevelse'] if 'telleoevelse' in dict else 0
    
    # Defining the subfolder for the results
    path_res = path
    if 'subfolder' in dict:
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    # Retrieving data from data set
    coldef_type = dict['coldef_type'] # Color deficiency type of the simulation method
    dict.update({'investigated-item': 'simulation method'})
    filename_orig = dict['filename']
    
    if 'method_ids' in dict:
        method_ids = dict['method_ids']
    else:
        method_ids = sorted(set(samsem_data['sim_id'].values.astype(int)))
    method_ids = sorted(method_ids) # IDs of the methods that need to be plotted
    
    print("Starting SAMSEM_RES#01+04: Analyzing data for " + str(settings.id2ColDefLong[dict['coldef_type']]) + " simulation methods: "+str(keys2values(method_ids,settings.id2Sim))+".")
    
    pandas_dict = {}; i = 0; order_dict = {}; sim_names = []
    # Retrieving data for each of the algorithms algorithm
    for sim_id in method_ids:
        
        sim_method = settings.id2Sim[sim_id]
        if (sim_id != 3) and (sim_id != 99):
            whatArr_tmp = [dict['obs_operator'],['sim_id',operator.eq,sim_id],['coldef_type',operator.eq,coldef_type]]
        else:
            whatArr_tmp = [dict['obs_operator'],['sim_id',operator.eq,sim_id]] # For the kotera and the dummy method, both protanopia and deuteranopia variants are identical. Thus, no distinction of coldef_type is necessary.
        relevant_data_tmp = organizeArray(samsem_data,whatArr_tmp)   
               
        pandas_dict.update({sim_method: relevant_data_tmp})
        order_dict.update({i: sim_method}); i += 1
        sim_names.append(sim_method)
    
    # Plot response time data as boxplots
    if telleoevelse: print("Observations RT plots")
    boxes, labels = preparePandas4RTPlots(pandas_dict, order_dict)
    plotRTGraphs(boxes,labels,path_res,dict,order_dict)
        
    # Plot accuracy data with confidence intervals
    c = 1.96; type = 'wilson-score';
    
    if telleoevelse: print("Observations ACC plots")
    accuracies = preparePandas4AccuracyPlots(pandas_dict,order_dict,c,type)
    plotAccuracyGraphs(accuracies,path_res,dict,order_dict)
        
    # Make median test as csv file
    dict.update({'filename': filename_orig+"-RT"})
    makeMedianTest(numpy.array(boxes), path_res, sim_names,dict)
    
    # Make Chi2 contingency test as text file
    obs_array, obs_pandas = preparePandas4Chi2(pandas_dict, order_dict)
    
    start = 0; end = 4 # For the global Chi2 test, we are excluding the dummy method.
    res = [start,end]
    dict.update({'filename': filename_orig+'-ACC'})
    makePearsonChi2Contingency(obs_array, obs_pandas, labels, path_res, dict, res)
    
    # Make pairwise Chi2 contingency test matrix as csv file
    makePearsonChi2Contingency2x2Test(obs_array, path_res, labels, dict)
    
    # Make normality Q-Q and log-Q-Q plots
    for i in range(numpy.shape(boxes)[0]):
        distribution_tmp = boxes[i]
        label_tmp = labels[i]
        dict.update({'filename': filename_orig+'-RT-'+label_tmp})
        plotQQPlot(distribution_tmp, path_res, dict)
        
        distribution_log_tmp = numpy.log(distribution_tmp)
        distribution_log_tmp = distribution_log_tmp[~numpy.isnan(distribution_log_tmp)]
        dict.update({'filename': filename_orig+'-RT-'+label_tmp+'-log'})
        plotQQPlot(distribution_log_tmp, path_res, dict)

def samsemPlots1thru4Paired(samsem_data,path,dict):
    
    compute_paired_data = 1
    telleoevelse = dict['telleoevelse'] if 'telleoevelse' in dict else 0
    
    # Defining the subfolder for the results
    path_res = path
    if 'subfolder' in dict:
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    # Retrieving data from data set
    coldef_type = dict['coldef_type'] # Color deficiency type of the simulation method
    dict.update({'investigated-item': 'simulation method'})
    filename_orig = dict['filename']
    
    if 'method_ids' in dict:
        method_ids = dict['method_ids']
    else:
        method_ids = sorted(set(samsem_data['sim_id'].values.astype(int)))
    method_ids = sorted(method_ids) # IDs of the methods that need to be plotted
    
    print("Starting SAMSEM_RES#01+04-paired: "+filename_orig+": Individual methods - ACC paired student-t test, RT paired median test.")
    
    if compute_paired_data:
        makePairedDataForSaMSEM(samsem_data, dict['path_data'], dict)
    
    samsem_data_RT_paired_path = os.path.join(dict['path_data'],filename_orig+'_samsem-data-RT-paired.csv')
    samsem_data_RT_paired = pandas.read_csv(samsem_data_RT_paired_path,index_col=False,sep=';')  
    
    f = open(os.path.join(dict['path_data'],filename_orig+'_samsem-data-paired_meta-data.txt'), 'r')
    b = json.load(f)
    f.close()
    
    samsem_data_RT_paired_array, labels = makePairedRTDataArray(samsem_data_RT_paired, b, 'sim_id')
    labels =  keys2values(labels, settings.id2Sim)
    
    # Make median test
    dict.update({'filename': filename_orig+'-RT', 'RT_difference':1})
    makeMedianTest(samsem_data_RT_paired_array, path_res, labels, dict)
    
    samsem_data_ACC_paired_path = os.path.join(dict['path_data'],filename_orig+'_samsem-data-ACC-paired.csv')
    samsem_data_ACC_paired = pandas.read_csv(samsem_data_ACC_paired_path,index_col=False,sep=';')  
    
    f = open(os.path.join(dict['path_data'],filename_orig+'_samsem-data-paired_meta-data.txt'), 'r')
    b = json.load(f)
    f.close()
    
    boxes = []
    labels = []
    #import math
    for i in b:
        #res_str = "orig-to-"+settings.id2Dalt[i]
        curr_col_name = "sim_id_"+str(i).zfill(2)
        curr_dalt_name = settings.id2Sim[i]
        curr_col = samsem_data_ACC_paired[curr_col_name].values
        #curr_comp = numpy.greater(orig_col,curr_col)*(-1.0) + numpy.less(orig_col,curr_col)*(1.0)
            
        boxes.append(curr_col)
        labels.append(curr_dalt_name)
    
    dict.update({'filename': filename_orig+"-ACC", 'ACC_difference': 1})
    makePairwiseStudentTTest(boxes, path_res, labels, dict)
    
    res_str = str(labels)+"\n"
    res_str += str(numpy.mean(boxes,axis=1))
    text_file_path = os.path.join(path_res,filename_orig+"-ACC-means.txt")
    text_file = open(text_file_path, "w+")
    text_file.write(res_str)
    text_file.close()  
    
def samsemPlots9thru12(samsem_data,path,dict):
    """
    Literally identical to samsemPlots1thru4. Check documentation there.
    """
    
    samsemPlots1thru4(samsem_data,path,dict)


def samsemPlots17thru20(samsem_data, path, dict):
    """
    Plots the SaMSEM Res#17-30
    Meaning that for each algorithm individually simulating one specific color deficiency results for all images are plotted.
    On the x-axis, we have the image ids. The images are stored in (subfolder) defined by the path input.
    """
    
    path_res = path
    
    if 'subfolder' in dict:
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    coldef_type = dict['coldef_type']
    print("Starting SAMSEM_RES#17+20: Analyzing data of all images for " + str(settings.id2ColDefLong[dict['coldef_type']]) + " simulation methods.")
    
    if 'plot_types' in dict:
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
    Comparing RT and ACC data of different observer groups of all simulation methods for a chosen color deficiency type.
    All images simulating a specific color deficiency of all simulation methods are collapsed.
    On the x-axis, we have the the observer groups.
    Input: * samsem_data:    SaMSEM data that is to be analyzed.
           * path:           Path to the folder, where the result images should be stored. 
           * dict:           A dictionary containing different options.
                             ** coldef_type:      The color deficiency type of the simulation method being used.
                             ** filename:         Name of the file as which the result image should be stored.
                             ** method_ids:       IDs of methods that are included in the analyis. By default: All available simulation methods included in samsem_data. 
                             ** observer_groups:  ID of the observer groups that should be included in the analyis.
                             ** subfolder:        Subfolder in path in which the images should be stored.
    Plots the SaMSEM Res#29+30.
    Before: s2mplots29and30
    """
    
    telleoevelse = dict['telleoevelse'] if 'telleoevelse' in dict else 0
        
    # Retrieving input data for the analysis
    path_res = path    
    if 'subfolder' in dict:
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
        
    filename_orig = dict['filename']
    coldef_type = dict['coldef_type']
    observer_groups = dict['observer_groups']
    dict.update({'investigated-item': 'observer groups'})
    
    method_ids = dict['method_ids'] if 'method_ids' in dict else sorted(set(samsem_data['sim_id'].values.astype(int)))
    
    print("Starting SAMSEM_RES#29+30: Analyzing data of observer groups "+ str(keys2values(observer_groups,settings.id2ColDefShort))+" for " + str(settings.id2ColDefLong[dict['coldef_type']]) + " simulation methods: "+str(keys2values(method_ids,settings.id2Sim))+".")
    
    # Restricting input data to only include the simulation methods that have been chosen for the analysis
    rel_data = pandas.DataFrame()
    for method_id in method_ids:
        if (method_id != 3) and (method_id != 99): 
            whatArr_tmp = [['sim_id',operator.eq,method_id],['coldef_type',operator.eq,coldef_type]]
        else:
            whatArr_tmp = [['sim_id',operator.eq,method_id]] # For the kotera and the dummy method, both protanopia and deuteranopia variants are identical. Thus, no distinction of coldef_type is necessary.
        rel_data_tmp = organizeArray(samsem_data,whatArr_tmp)
        rel_data = pandas.concat([rel_data_tmp, rel_data])
    samsem_data_adj = rel_data.reset_index()
    
    # Retrieving data for each of the observation groups
    i = 0; pandas_dict = {}; order_dict = {}
    for observer_group in observer_groups:
        observer_coldef_type = observer_group
        observer_coldef_type_short = settings.id2ColDefShort[observer_coldef_type]
        
        whatArr_tmp = [['observer_coldef_type',operator.eq,observer_coldef_type]]
        obs_group_data_tmp = organizeArray(samsem_data_adj,whatArr_tmp)
        
        pandas_dict.update({observer_coldef_type_short:obs_group_data_tmp})
        order_dict.update({i:observer_coldef_type_short}) ; i += 1
    
    # Plot response time data as boxplots
    if telleoevelse: print("Observations RT plots")
    boxes, labels = preparePandas4RTPlots(pandas_dict, order_dict)
    plotRTGraphs(boxes,labels,path_res, dict)
    
    # Plot accuracy with confidence intervals
    c = 1.96; type = 'wilson-score'
    if telleoevelse: print("Observations ACC plots")
    accuracies = preparePandas4AccuracyPlots(pandas_dict,order_dict,c,type)
    plotAccuracyGraphs(accuracies,path_res,dict,order_dict)
        
    # Make median test as csv file
    dict.update({'filename': filename_orig+"-RT"})
    makeMedianTest(boxes, path_res, labels, dict)
    
    # Make Chi2 contingency test as txt file
    obs_array, obs_pandas = preparePandas4Chi2(pandas_dict, order_dict)
    dict.update({'filename': filename_orig+'-ACC'})
    makePearsonChi2Contingency(obs_array, obs_pandas, labels, path_res, dict)
    
    # Make Chi2 contingency test matrix as csv file
    makePearsonChi2Contingency2x2Test(obs_array, path_res, labels, dict)
    
    # Make normality plots as Q-Q and log-Q-Q plots
    for i in range(numpy.shape(boxes)[0]):
        distribution_tmp = boxes[i]
        label_tmp = labels[i]
        dict.update({'filename': filename_orig+'-RT-'+label_tmp})
        plotQQPlot(distribution_tmp, path_res, dict)
        
        distribution_log_tmp = numpy.log(distribution_tmp)
        distribution_log_tmp = distribution_log_tmp[~numpy.isnan(distribution_log_tmp)]
        dict.update({'filename': filename_orig+'-RT-'+label_tmp+'-log'})
        plotQQPlot(distribution_log_tmp, path_res, dict)
        
    
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
    if 'result_id' in dict:
        result_id = dict['result_id']   
        if not os.path.exists(os.path.join(path,str(result_id))): os.makedirs(os.path.join(path,str(result_id)))
    
    if 'subfolder' in dict:
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    #intro_string = result_id if result_id else dict['filename']  # filename does not make any sense at this point
    #sys.stdout.write("Starting Res#"+str(intro_string)+'.') 
    coldef_type = dict['coldef_type']
    if 'plot_types' in dict:
        plot_types = dict['plot_types']
    else:
        plot_types = ['histogramm', 'residual', 'q-q']
    print("Starting SaMSEM_RES#35+36: Computing normality plot ("+str(plot_types)+") for "+str(settings.id2ColDefLong[coldef_type])+" simulation methods.")
    
    boxes = []; labels = []
    
    sim_ids = set(samsem_data['sim_id'].values.astype(int))
    for sim_id in sim_ids:
        
        filename = str(settings.id2ColDefLong[coldef_type])+"-method-RT-"+str(sim_id)+"-"+str(settings.id2Sim[sim_id])
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
            
            alg_RT_log_values = numpy.log(alg_RT_values)
            alg_RT_log_values = alg_RT_log_values[~numpy.isnan(alg_RT_log_values)]
            filename_log = str(settings.id2ColDefLong[coldef_type])+"-method-RT-"+str(sim_id)+"-"+str(settings.id2Sim[sim_id])+"-log"
            dict.update({'filename': filename_log})
            plotQQPlot(alg_RT_log_values, path_res, dict)
    
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
    if 'result_id' in dict:
        result_id = dict['result_id']   
        if not os.path.exists(os.path.join(path,str(result_id))): os.makedirs(os.path.join(path,str(result_id)))
    
    if 'subfolder' in dict:
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    
    #intro_string = result_id if result_id else dict['filename']  # filename does not make any sense at this point
    #sys.stdout.write("Starting Res#"+str(intro_string)+'.') 
    coldef_type = dict['coldef_type']
    if 'plot_types' in dict:
        plot_types = dict['plot_types']
    else:
        plot_types = ['histogramm', 'q-q']
    print("Starting SaMSEM_RES#37+38: Computing normality plot ("+str(plot_types)+") of observer groups for "+str(settings.id2ColDefLong[coldef_type])+" simulation methods.")
    
    if 'filename' in dict:
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
        filename_tmp = filename + "RT-normal-RT"
        plotHistogram(normal_response_values, path_res, dict.update({'filename': filename_tmp}))
        filename_tmp = filename + "RT-protan-RT"
        plotHistogram(protan_response_values, path_res, dict.update({'filename': filename_tmp}))
        filename_tmp = filename + "RT-deutan-RT"
        plotHistogram(deutan_response_values, path_res, dict.update({'filename': filename_tmp}))
    
    if "q-q" in plot_types:
        filename_tmp = filename + "-RT-normal"
        dict.update({'filename': filename_tmp})
        plotQQPlot(normal_response_values, path_res, dict)
        
        filename_tmp = filename + "-RT-normal-log"
        normal_response_log_values = numpy.log(normal_response_values)
        normal_response_log_values = normal_response_log_values[~numpy.isnan(normal_response_log_values)]
        dict.update({'filename': filename_tmp})
        plotQQPlot(normal_response_log_values, path_res, dict)

        filename_tmp = filename + "-RT-protan"
        dict.update({'filename': filename_tmp})
        plotQQPlot(protan_response_values, path_res, dict)
        
        filename_tmp = filename + "-RT-protan-log"
        protan_response_log_values = numpy.log(protan_response_values)
        protan_response_log_values = protan_response_log_values[~numpy.isnan(protan_response_log_values)]
        dict.update({'filename': filename_tmp})
        plotQQPlot(protan_response_log_values, path_res, dict)
        
        filename_tmp = filename + "-RT-deutan"
        dict.update({'filename': filename_tmp})
        plotQQPlot(deutan_response_values, path_res, dict)
        
        filename_tmp = filename + "-RT-deutan-log"
        deutan_response_log_values = numpy.log(deutan_response_values)
        deutan_response_log_values = deutan_response_log_values[~numpy.isnan(deutan_response_log_values)]
        dict.update({'filename': filename_tmp})
        plotQQPlot(deutan_response_log_values, path_res, dict)


def samsemPlots41and42(samsem_data,path,dict):
    """
    Making pearson chi2-contingency test for all algorithms samsem data.
    """
    coldef_type = dict['coldef_type']
    print("Starting SAMSEM_RES#41+42: Computing of Chi2 for simulation methods of " + str(settings.id2ColDefLong[coldef_type])+".")
    
    if 'subfolder' in dict:
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
    print("Starting SAMSEM_RES#43+44: Computing of Chi2 for observer groups for " + str(settings.id2ColDefLong[coldef_type]) + " simulation methods.")
    
    
    if 'subfolder' in dict:
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
            
        whatArr_tmp = [['variant_id',operator.eq,0],dict['obs_operator'],['set_id',operator.eq,set_id],['is_correct',operator.eq,True]];howArr_tmp = []
        DNData =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
            
        whatArr_tmp = [['variant_id',operator.eq,2],['dalt_id',operator.eq,0],dict['obs_operator'],['set_id',operator.eq,set_id],['is_correct',operator.eq,True]];howArr_tmp = []
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
    
    motive_ids = set(visual_search_data['motive_id'].values.astype(int))        
    for motive_id in motive_ids:
        sys.stdout.write("Scene#"+str(motive_id)+'.')
        # extract only DN and DE images for norm.sigh.obs.
        filename_tmp = '000999'+dict['obs_coldef_type']+str(getSetFromScene(motive_id)).zfill(3)+str(motive_id).zfill(3)+'9-v'
            
        whatArr_tmp = [['variant_id',operator.eq,0],dict['obs_operator'],['motive_id',operator.eq,motive_id],['is_correct',operator.eq,True]];howArr_tmp = []
        DNData =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
            
        whatArr_tmp = [['variant_id',operator.eq,2],['dalt_id',operator.eq,0],dict['obs_operator'],['motive_id',operator.eq,motive_id],['is_correct',operator.eq,True]];howArr_tmp = []
        DEData = organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
            
        boxes_tmp = [];labels_tmp = []
        DND_values = DNData['resp_time'].values; labels_tmp.append('DN') if DND_values.size else labels_tmp.append('DN - No data'); boxes_tmp.append(DND_values)
        DED_values = DEData['resp_time'].values; labels_tmp.append('DE') if DED_values.size else labels_tmp.append('DE - No data'); boxes_tmp.append(DED_values)
            
        plt.figure(); plt.boxplot(boxes_tmp, notch=1)
        plt.xticks([1,2],labels_tmp); plt.title('Scene '+str(motive_id)+': '+dict['obs_title']); plt.ylabel('response time'); plt.ylim([0,5])
        plt.savefig(os.path.join(path,str(dict['result_id']),filename_tmp+".png")); plt.close()
    
    sys.stdout.write('\n')


def visdemPlots53thru60(visdem_data,path,dict):
    
    print("   Starting ViSDEM_RES#53-60 - "+dict['filename']+": Individual methods - ACC plots, ACC Chi2 tests, RT boxplots, RT median test, and Q-Q/Q-Q-log normality check.")
    
    dict.update({'RT_difference':0})
    
    path_res = path
    if 'subfolder' in dict:
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    filename_orig = dict['filename']
    
    dict.update({'investigated-item': 'daltonization method'})
    set_ids = dict['sets'] if 'sets' in dict else sorted(set(visdem_data['set_id'].values.astype(int))) 
        
    #Retrieving data    
    visdem_data_adj = pandas.DataFrame()
    for set_id in set_ids:
        # Getting data from all variants for all normal sighted observers
        whatArr_tmp = [['set_id',operator.eq,set_id]]
        rel_data_tmp =  organizeArray(visdem_data,whatArr_tmp)
        visdem_data_adj = pandas.concat([rel_data_tmp, visdem_data_adj])
    visdem_data_adj.reset_index()
    
    dalt_method_ids = sorted(set(visdem_data['dalt_id'].values.astype(int)))
    
    pandas_dict = {}; i = 0; order_dict = {}; dalt_names = []; 
    for dalt_method_id in dalt_method_ids:
        dalt_method = settings.id2Dalt[dalt_method_id]
        obs_groups = dict['obs_groups']
        relevant_data_tmp = pandas.DataFrame()
        for obs_group, coldef_type in obs_groups.items():
            
            whatArray = [['dalt_id', operator.eq, dalt_method_id],['variant_id',operator.eq,1],['observer_coldef_type',operator.eq,obs_group]]
            if (dalt_method_id != 0)  and (dalt_method_id != 99):
                whatArray.append(['coldef_type',operator.eq,coldef_type])
            relevant_data_group_tmp = organizeArray(visdem_data_adj, whatArray)
            relevant_data_tmp = pandas.concat([relevant_data_group_tmp,relevant_data_tmp])
        relevant_data_tmp.reset_index()
        
        pandas_dict.update({dalt_method: relevant_data_tmp})
        order_dict.update({i: dalt_method}); i += 1
        dalt_names.append(dalt_method)
    
    # Make response time plots
    boxes, labels = preparePandas4RTPlots(pandas_dict, order_dict)
    plotRTGraphs(boxes, labels, path_res, dict)
    
    # Make Accuracy plots
    c = 1.96; type = 'wilson-score';
    accuracies = preparePandas4AccuracyPlots(pandas_dict,order_dict,c,type,dict)
    plotAccuracyGraphs(accuracies, path_res, dict, order_dict)
    
    # Make median test
    dict.update({'filename': filename_orig+'-RT'})
    makeMedianTest(boxes, path_res, labels, dict)
    
    # Make Chi2 contingency test
    obs_array, obs_pandas = preparePandas4Chi2(pandas_dict, order_dict)
    
    start = 1; end = 5; res = [start,end]
    dict.update({'filename': filename_orig+'-ACC'})
    #print obs_array
    makePearsonChi2Contingency(obs_array, obs_pandas, labels, path_res, dict, res)
    
    # Make Chi2 contingency test matrix
    makePearsonChi2Contingency2x2Test(obs_array, path_res, labels, dict)
    
    # 7. Make normality plots
    qq_plots = dict['qq_plots'] if 'qq_plots' in dict else 0
    if qq_plots:
        for i in range(numpy.shape(boxes)[0]):
            distribution_tmp = boxes[i]
            if distribution_tmp.size:
                label_tmp = labels[i]
                dict.update({'filename': filename_orig+'-RT-'+label_tmp})
                plotQQPlot(distribution_tmp, path_res, dict)
                
                distribution_log_tmp = numpy.log(distribution_tmp)
                distribution_log_tmp = distribution_log_tmp[~numpy.isnan(distribution_log_tmp)]
                dict.update({'filename': filename_orig+'-RT-'+label_tmp+'-log'})
                plotQQPlot(distribution_log_tmp, path_res, dict)


def visdemPlots53thru60Paired(visdem_data,path,dict):
    
    print("   Starting ViSDEM_RES#53-60-paired - "+dict['filename']+": Individual methods - ACC paired student-t test, ACC Chi2 tests, RT paired median test.")
    
    compute_paired_data = 1
    
    path_res = path
    if 'subfolder' in dict:
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    filename_orig = dict['filename']

    set_ids = dict['sets'] if 'sets' in dict else sorted(set(visdem_data['set_id'].values.astype(int)))

    visdem_data_adj = pandas.DataFrame()
    for set_id in set_ids:
        # Getting data from all variants for all normal sighted observers
        whatArr_tmp = [['set_id',operator.eq,set_id]]
        rel_data_tmp =  organizeArray(visdem_data,whatArr_tmp)
        visdem_data_adj = pandas.concat([rel_data_tmp, visdem_data_adj])
    visdem_data_adj.reset_index()
    
    if compute_paired_data: makePairedDataForViSDEM(visdem_data_adj, dict['path_data'], dict)
    
    visdem_data_RT_paired_path = os.path.join(dict['path_data'],filename_orig+'_visdem-data-RT-paired.csv')
    visdem_data_RT_paired = pandas.read_csv(visdem_data_RT_paired_path,index_col=False,sep=';')  
    
    f = open(os.path.join(dict['path_data'],filename_orig+'_visdem-data-paired_meta-data.txt'), 'r'); b = json.load(f); f.close()
    
    visdem_data_RT_paired_array, labels = makePairedRTDataArray(visdem_data_RT_paired, b, 'dalt_id')
    labels =  keys2values(labels, settings.id2Dalt)
    
    # Make median test
    dict.update({'filename': filename_orig+'-RT', 'RT_difference':1})
    makeMedianTest(visdem_data_RT_paired_array, path_res, labels, dict)
    
    visdem_data_ACC_paired_path = os.path.join(dict['path_data'],filename_orig+'_visdem-data-ACC-paired.csv')
    visdem_data_ACC_paired = pandas.read_csv(visdem_data_ACC_paired_path,index_col=False,sep=';')  
    
    f = open(os.path.join(dict['path_data'],filename_orig+'_visdem-data-paired_meta-data.txt'), 'r'); b = json.load(f); f.close()
    
    orig_col = visdem_data_ACC_paired['dalt_id_00'].values; comp_dict = {}; boxes = []; labels = []
    for i in b:
        #res_str = "orig-to-"+settings.id2Dalt[i]
        curr_col_name = "dalt_id_"+str(i).zfill(2)
        curr_dalt_name = settings.id2Dalt[i]
        curr_col = visdem_data_ACC_paired[curr_col_name].values
        #curr_comp = numpy.greater(orig_col,curr_col)*(-1.0) + numpy.less(orig_col,curr_col)*(1.0)
            
        boxes.append(curr_col);labels.append(curr_dalt_name)
    
    dict.update({'filename': filename_orig+"-ACC", 'ACC_difference': 1})
    makePairwiseStudentTTest(boxes, path_res, labels, dict)
    
    res_str = str(labels)+"\n"
    res_str += str(numpy.mean(boxes,axis=1))
    text_file_path = os.path.join(path_res,filename_orig+"-ACC-means.txt")
    text_file = open(text_file_path, "w+")
    text_file.write(res_str)
    text_file.close()
     
    
def visdemPlots67thru70(visdem_data,path,dict):
    
    print("   Starting ViSDEM_RES#67-70: Plotting of ACC and RT data, and computying of Chi2 of ACC and median test of RT for observer group " + dict['filename']+".")
    
    telleoevelse = dict['telleoevelse'] if 'telleoevelse' in dict else 0
    
    path_res = path
    if 'subfolder' in dict:
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
        if not os.path.exists(path_res): os.makedirs(path_res)
    filename_orig = dict['filename']
    dict.update({'investigated-item': 'daltonization method'})
    
    #DCData = pandas.DataFrame(); CCData = pandas.DataFrame(); SCData = pandas.DataFrame()
    if 'sets' in dict:
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
    
    # 1. Retrieving data from the image variants
    whatArr_tmp = [['variant_id',operator.eq,0],['dalt_id',operator.eq,0],dict['obs_operator']]
    DCData =  organizeArray(visdem_data_adj,whatArr_tmp)
            
    whatArr_tmp = [['variant_id',operator.eq,1],['dalt_id',operator.eq,0],dict['obs_operator']]
    CCData = organizeArray(visdem_data_adj,whatArr_tmp)
        
    whatArr_tmp = [['variant_id',operator.eq,2],['dalt_id',operator.eq,0],dict['obs_operator']]
    SCData = organizeArray(visdem_data_adj,whatArr_tmp)
    
    pandas_dict = {'Diff.Col.': DCData, 'Conf.Col.': CCData, 'SameCol.': SCData}
    order_dict = {0: 'Diff.Col.', 1: 'Conf.Col.', 2: 'SameCol.'}
    
    # 2. Make response time plots
    boxes, labels = preparePandas4RTPlots(pandas_dict, order_dict,dict)
    plotRTGraphs(boxes, labels, path_res, dict)
    #labels = ['DC', 'CC', 'SC']
    
    # 3. Make Accuracy plots
    c = 1.96; type = 'wilson-score';
    accuracies = preparePandas4AccuracyPlots(pandas_dict,order_dict,c,type,dict)
    plotAccuracyGraphs(accuracies, path_res, dict, order_dict)
    
    # 4. Make median test 
    dict.update({'filename': filename_orig+'-RT'})
    makeMedianTest(boxes, path_res, labels, dict)
              
    # 5. Make Chi2 contingency test
    obs_array, obs_pandas = preparePandas4Chi2(pandas_dict, order_dict)
        
    dict.update({'filename': filename_orig+'-ACC'})
    makePearsonChi2Contingency(obs_array, obs_pandas, labels, path_res, dict)

    # 6. Make Chi2 contingency test matrix
    makePearsonChi2Contingency2x2Test(obs_array, path_res, labels, dict)
    
    # 7. Make normality plots
    for i in range(numpy.shape(boxes)[0]):
        distribution_tmp = boxes[i]
        if distribution_tmp.size:
            label_tmp = labels[i]
            dict.update({'filename': filename_orig+'-RT-'+label_tmp})
            plotQQPlot(distribution_tmp, path_res, dict)
        
            distribution_log_tmp = numpy.log(distribution_tmp)
            distribution_log_tmp = distribution_log_tmp[~numpy.isnan(distribution_log_tmp)]
            dict.update({'filename': filename_orig+'-RT-'+label_tmp+'-log'})
            plotQQPlot(distribution_log_tmp, path_res, dict)
        
     
def vsplots71thru74(visual_search_data,path,dict):
    
    result_id = dict['result_id']
    intro_string = result_id if result_id else dict['filename'] 
    sys.stdout.write("Starting Res#"+str(intro_string)+' -> ')    
    if not os.path.exists(os.path.join(path,str(dict['result_id']))): os.makedirs(os.path.join(path,str(dict['result_id'])))
    
    if 'sets' in dict:
        set_ids = dict['sets']
    else:
        set_ids = []
    
    beforeData = pandas.DataFrame(); afterData = pandas.DataFrame();
    
    for set_id in set_ids:
        # get all the data before daltonization
        whatArr_tmp = [['dalt_id',operator.eq,0],['variant_id',operator.eq,1],dict['obs_operator'],];howArr_tmp = []
        beforeData_tmp =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        beforeData = pandas.concat([beforeData_tmp,beforeData])
        
        # get all the data after daltonization
        whatArr_tmp = [['dalt_id',operator.eq,2],['variant_id',operator.eq,1],dict['obs_operator'],];howArr_tmp = []
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
    
    if 'filename' in dict:
        filename = dict['filename']
    else:    
        filename = "visdem-normal-deutan-observers"
    
    if 'obs_title' in dict:
        obs_title = dict['obs_title']
    else:
        obs_title = ''
    
    if 'fontsize' in dict:
        fontsize = dict['fontsize']
    else:
        fontsize = 18
        
    if 'y_lim_ACC' in dict:
        y_lim_ACC = dict['y_lim_ACC']
    else:
        y_lim_ACC = [.0,1.] 
        
    if 'y_lim_RT' in dict:
        y_lim_RT = dict['y_lim_RT']
    else:
        y_lim_RT = [0,3000]
    
    if 'result_id' in dict:
        result_id = dict['result_id']
    else:
        result_id = ''
    intro_string = result_id if result_id else filename
         
    sys.stdout.write("Starting Res#"+str(intro_string))    
    if not os.path.exists(os.path.join(path,str(result_id))): os.makedirs(os.path.join(path,str(result_id)))
    
    Data_norm = pandas.DataFrame()
    Data_deut = pandas.DataFrame()
    if 'sets' in dict:
        set_ids = dict['sets']
    else:
        set_ids = []
    
    # 1. Retrieving data from the image variants
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
    print()
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
    if 'fontsize' in dict:
        fontsize = dict['fontsize']
    else:
        fontsize = 18
    
    result_id = dict['result_id']
    intro_string = result_id if result_id else dict['filename'] 
    sys.stdout.write("Starting Res#"+str(intro_string))    
    if not os.path.exists(os.path.join(path,str(dict['result_id']))): os.makedirs(os.path.join(path,str(dict['result_id'])))
    
    DNData_norm = pandas.DataFrame(); DCData_norm = pandas.DataFrame()
    DNData_deut = pandas.DataFrame(); DCData_deut = pandas.DataFrame()
    if 'sets' in dict:
        set_ids = dict['sets']
    else:
        set_ids = []
    
    # 1. Retrieving data from the image variants
    for set_id in set_ids:
        whatArr_tmp = [['variant_id',operator.eq,0],['dalt_id',operator.eq,0],['observer_coldef_type',operator.eq,0],['set_id',operator.eq,set_id]];howArr_tmp = []
        DNData_norm_tmp =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        DNData_norm = pandas.concat([DNData_norm_tmp,DNData_norm])
        
        whatArr_tmp = [['variant_id',operator.eq,1],['dalt_id',operator.eq,0],['observer_coldef_type',operator.eq,0],['set_id',operator.eq,set_id]];howArr_tmp = []
        DCData_norm_tmp = organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        DCData_norm = pandas.concat([DCData_norm_tmp,DCData_norm])
        
        whatArr_tmp = [['variant_id',operator.eq,0],['dalt_id',operator.eq,0],['observer_coldef_type',operator.eq,2],['set_id',operator.eq,set_id]];howArr_tmp = []
        DNData_deut_tmp =  organizeArray(visual_search_data,whatArr_tmp,howArr_tmp)
        DNData_deut = pandas.concat([DNData_deut_tmp,DNData_deut])
        
        whatArr_tmp = [['variant_id',operator.eq,1],['dalt_id',operator.eq,0],['observer_coldef_type',operator.eq,2],['set_id',operator.eq,set_id]];howArr_tmp = []
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

def visdemPlots77thru80(visdem_data,path,dict):
    
    print("   Starting ViSDEM_RES#77-80 - "+dict['filename']+": Individual "+dict['investigated-item']+"s - ACC plots and ACC Chi2 tests.")
    
    path_res_orig = path
    if 'subfolder' in dict:
        subfolder = dict['subfolder']
        path_res_orig = os.path.join(path,subfolder)
        if not os.path.exists(path_res_orig): os.makedirs(path_res_orig)
    filename_orig = dict['filename']
    item = dict['investigated-item']
    
    coldef_type = dict['coldef_type']
    
    item_ids = dict[item] if item in dict else sorted(set(visdem_data[item+'_id'].values.astype(int)))
    dalt_ids = dict['dalt_ids'] if 'dalt_ids' in dict else sorted(set(visdem_data['dalt_id'].values.astype(int)))
    
    visdem_data_adj = pandas.DataFrame()
    for item_id in item_ids:
        # Getting data from all variants for all normal sighted observers
        whatArr_tmp = [['set_id',operator.eq,item_id]]
        rel_data_tmp =  organizeArray(visdem_data,whatArr_tmp)
        visdem_data_adj = pandas.concat([rel_data_tmp, visdem_data_adj])
    visdem_data_adj.reset_index()
    
    # 1. Retrieving data from the image variants
    path_res = os.path.join(path_res_orig,'after_dalt_id')
    if not os.path.exists(path_res): os.makedirs(path_res)
    pdf_file_names = []
    for dalt_id in dalt_ids:
        dalt_name = settings.id2Dalt[dalt_id]
        filename_tmp = filename_orig+'-'+dalt_name
        pdf_file_names.append(os.path.join(path_res,filename_tmp+'-ACC.pdf'))
        title = ""#settings.id2ColDefShort[dict['obs_operator'][2]] + " "+ dalt_name
        dict.update({'filename':filename_tmp, 'obs_title': title})        

        # Retrieving data for each of the observation groups
        i = 0; pandas_dict = {}; order_dict = {}
        for item_id in item_ids:
            
            whatArr_tmp = [dict['obs_operator'],['variant_id',operator.eq,1],['dalt_id',operator.eq,dalt_id],[item+'_id',operator.eq,item_id]]
            if (dalt_id != 0)  and (dalt_id != 99):
                whatArr_tmp.append(['coldef_type',operator.eq,coldef_type])
            item_data_tmp = organizeArray(visdem_data_adj,whatArr_tmp)
            
            item_label = str(item_id)
            if item == 'motive':
                set_id =  item_data_tmp['set_id'].values.astype(int)[0]
                item_label = str(set_id)+'-'+str(item_id)
                
                   
            pandas_dict.update({item_label:item_data_tmp})
            order_dict.update({i:item_label}) ; i += 1
        
        # Plot response time data as boxplots
        #if telleoevelse: print "Observations RT plots"
        boxes, labels = preparePandas4RTPlots(pandas_dict, order_dict,dict)
        #plotRTGraphs(boxes,labels,path_res, dict)
        
        # Plot accuracy with confidence intervals
        c = 1.96; type = 'wilson-score'
        #if telleoevelse: print "Observations ACC plots"
        accuracies = preparePandas4AccuracyPlots(pandas_dict,order_dict,c,type,dict)
        plotAccuracyGraphs(accuracies,path_res,dict,order_dict)
            
        # Make median test as csv file
        dict.update({'filename': filename_tmp+"-RT"})
        #makeMedianTest(boxes, path_res, labels, dict)
        
        # Make Chi2 contingency test as txt file
        obs_array, obs_pandas = preparePandas4Chi2(pandas_dict, order_dict)
        dict.update({'filename': filename_tmp+'-ACC'})
        makePearsonChi2Contingency(obs_array, obs_pandas, labels, path_res, dict)
        
        # Make Chi2 contingency test matrix as csv file
        makePearsonChi2Contingency2x2Test(obs_array, path_res, labels, dict)
        
        # Make normality plots as Q-Q and log-Q-Q plots
        for i in range(numpy.shape(boxes)[0]):
            distribution_tmp = boxes[i]
            label_tmp = labels[i]
            dict.update({'filename': filename_orig+'-RT-'+label_tmp})
            #plotQQPlot(distribution_tmp, path_res, dict)
            
            distribution_log_tmp = numpy.log(distribution_tmp)
            distribution_log_tmp = distribution_log_tmp[~numpy.isnan(distribution_log_tmp)]
            dict.update({'filename': filename_orig+'-RT-'+label_tmp+'-log'})
            #plotQQPlot(distribution_log_tmp, path_res, dict)
    
    merger = PdfFileMerger()
    for filename in sorted(pdf_file_names):
        merger.append(PdfFileReader(filename, "rb"))
    merger.write(os.path.join(path_res_orig, item+"s-"+settings.id2ColDefShort[dict['obs_operator'][2]]+"-after_dalt_id.pdf"))
    merger.close()
    
    # After_item_id        
    path_res = os.path.join(path_res_orig,'after_'+item+'_id')
    if not os.path.exists(path_res): os.makedirs(path_res)
    pdf_file_names = []
    for item_id in item_ids:
    #for dalt_id in dalt_ids:
        filename_tmp = filename_orig+'-'+str(item_id).zfill(2)
        pdf_file_names.append(os.path.join(path_res,filename_tmp+'-ACC.pdf'))
        title = ""#settings.id2ColDefShort[dict['obs_operator'][2]] + " "+ item +"#"+str(item_id)
        dict.update({'filename':filename_tmp, 'obs_title': title})

        # Retrieving data for each of the observation groups
        i = 0; pandas_dict = {}; order_dict = {}
        #for item_id in item_ids:
        for dalt_id in dalt_ids:
            dalt_name = settings.id2Dalt[dalt_id]
            
            whatArr_tmp = [dict['obs_operator'],['variant_id',operator.eq,1],['dalt_id',operator.eq,dalt_id],[item+'_id',operator.eq,item_id]]
            if (dalt_id != 0)  and (dalt_id != 99):
                whatArr_tmp.append(['coldef_type',operator.eq,coldef_type])
            item_data_tmp = organizeArray(visdem_data_adj,whatArr_tmp)
                   
            pandas_dict.update({dalt_name:item_data_tmp})
            order_dict.update({i:dalt_name}) ; i += 1
        
        # Plot response time data as boxplots
        #if telleoevelse: print "Observations RT plots"
        boxes, labels = preparePandas4RTPlots(pandas_dict, order_dict)
        #plotRTGraphs(boxes,labels,path_res, dict)
        
        # Plot accuracy with confidence intervals
        c = 1.96; type = 'wilson-score'
        #if telleoevelse: print "Observations ACC plots"
        accuracies = preparePandas4AccuracyPlots(pandas_dict,order_dict,c,type)
        plotAccuracyGraphs(accuracies,path_res,dict,order_dict)
            
        # Make median test as csv file
        dict.update({'filename': filename_tmp+"-RT"})
        #makeMedianTest(boxes, path_res, labels, dict)
        
        # Make Chi2 contingency test as txt file
        obs_array, obs_pandas = preparePandas4Chi2(pandas_dict, order_dict)
        dict.update({'filename': filename_tmp+'-ACC'})
        makePearsonChi2Contingency(obs_array, obs_pandas, labels, path_res, dict)
        
        # Make Chi2 contingency test matrix as csv file
        makePearsonChi2Contingency2x2Test(obs_array, path_res, labels, dict)
        
        # Make normality plots as Q-Q and log-Q-Q plots
        for i in range(numpy.shape(boxes)[0]):
            distribution_tmp = boxes[i]
            label_tmp = labels[i]
            dict.update({'filename': filename_orig+'-RT-'+label_tmp})
            #plotQQPlot(distribution_tmp, path_res, dict)
            
            distribution_log_tmp = numpy.log(distribution_tmp)
            distribution_log_tmp = distribution_log_tmp[~numpy.isnan(distribution_log_tmp)]
            dict.update({'filename': filename_orig+'-RT-'+label_tmp+'-log'})
            #plotQQPlot(distribution_log_tmp, path_res, dict)
    
    merger = PdfFileMerger()
    for filename in sorted(pdf_file_names):
        merger.append(PdfFileReader(filename, "rb"))
    merger.write(os.path.join(path_res_orig, item+"s-"+settings.id2ColDefShort[dict['obs_operator'][2]]+"-after_"+item+"_id.pdf"))
    merger.close()

def visdemPlots81thru82(visdem_data,path,dict):
    
    print("   Starting ViSDEM_RES#81-82 - "+dict['filename']+": Natural vs. Ishihara sets - ACC plots and ACC Chi2 tests.")
    
    path_res = path
    if 'subfolder' in dict:
        subfolder = dict['subfolder']
        path_res = os.path.join(path,subfolder)
    if not os.path.exists(path_res): os.makedirs(path_res)
    filename_orig = dict['filename']
    #item = dict['investigated-item']
    
    coldef_type = dict['coldef_type']
    
    #item_ids = dict[item] if dict.has_key(item) else sorted(set(visdem_data[item+'_id'].values.astype(int)))
    dalt_ids = dict['dalt_ids'] if 'dalt_ids' in dict else sorted(set(visdem_data['dalt_id'].values.astype(int)))
    
    natural_sets = [1,2,3,4,5,6,7,8,9,10]
    natural_visdem_data = pandas.DataFrame()
    for set_id in natural_sets:
        # Getting data from all variants for all normal sighted observers
        whatArr_tmp = [['set_id',operator.eq,set_id]]
        rel_data_tmp =  organizeArray(visdem_data,whatArr_tmp)
        natural_visdem_data = pandas.concat([rel_data_tmp, natural_visdem_data])
    
    ishihara_sets = [11,12,13]
    ishihara_visdem_data = pandas.DataFrame() 
    for set_id in ishihara_sets:
        # Getting data from all variants for all normal sighted observers
        whatArr_tmp = [['set_id',operator.eq,set_id]]
        rel_data_tmp =  organizeArray(visdem_data,whatArr_tmp)
        ishihara_visdem_data = pandas.concat([rel_data_tmp, ishihara_visdem_data])
    ishihara_visdem_data.reset_index()
    
    # 1. Retrieving data from the image variants
    #pdf_file_names = []
    for dalt_id in dalt_ids:
        dalt_name = settings.id2Dalt[dalt_id]
        filename_tmp = filename_orig+'-'+dalt_name
        #pdf_file_names.append(os.path.join(path_res,filename_tmp+'-ACC.pdf'))
        title = ""#settings.id2ColDefShort[dict['obs_operator'][2]] + " "+ dalt_name
        dict.update({'filename':filename_tmp, 'obs_title': title})        

        # Retrieving data for each of the observation groups
        i = 0; pandas_dict = {}; order_dict = {}
        
        whatArr_tmp = [dict['obs_operator'],['variant_id',operator.eq,1],['dalt_id',operator.eq,dalt_id]]
        if (dalt_id != 0)  and (dalt_id != 99):
            whatArr_tmp.append(['coldef_type',operator.eq,coldef_type])
        natural_data_tmp = organizeArray(natural_visdem_data,whatArr_tmp)
        ishihara_data_tmp = organizeArray(ishihara_visdem_data,whatArr_tmp)
            
                       
        pandas_dict.update({'Natural':natural_data_tmp})
        order_dict.update({0:'Natural'})
        pandas_dict.update({'Ishihara':ishihara_data_tmp})
        order_dict.update({1:'Ishihara'})
        
        # Plot response time data as boxplots
        #if telleoevelse: print "Observations RT plots"
        boxes, labels = preparePandas4RTPlots(pandas_dict, order_dict,dict)
        #plotRTGraphs(boxes,labels,path_res, dict)
        
        # Plot accuracy with confidence intervals
        c = 1.96; type = 'wilson-score'
        #if telleoevelse: print "Observations ACC plots"
        accuracies = preparePandas4AccuracyPlots(pandas_dict,order_dict,c,type,dict)
        plotAccuracyGraphs(accuracies,path_res,dict,order_dict)
            
        # Make median test as csv file
        dict.update({'filename': filename_tmp+"-RT"})
        #makeMedianTest(boxes, path_res, labels, dict)
        
        # Make Chi2 contingency test as txt file
        obs_array, obs_pandas = preparePandas4Chi2(pandas_dict, order_dict)
        dict.update({'filename': filename_tmp+'-ACC'})
        makePearsonChi2Contingency(obs_array, obs_pandas, labels, path_res, dict)
        
        # Make Chi2 contingency test matrix as csv file
        makePearsonChi2Contingency2x2Test(obs_array, path_res, labels, dict)
        
        # Make normality plots as Q-Q and log-Q-Q plots
        for i in range(numpy.shape(boxes)[0]):
            distribution_tmp = boxes[i]
            label_tmp = labels[i]
            dict.update({'filename': filename_orig+'-RT-'+label_tmp})
            #plotQQPlot(distribution_tmp, path_res, dict)
            
            distribution_log_tmp = numpy.log(distribution_tmp)
            distribution_log_tmp = distribution_log_tmp[~numpy.isnan(distribution_log_tmp)]
            dict.update({'filename': filename_orig+'-RT-'+label_tmp+'-log'})
            #plotQQPlot(distribution_log_tmp, path_res, dict)

    

def makePairedDataForSaMSEM(samsem_data,path,dict):
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
    samsem_data_RT_paired = samsem_data_template.copy(); samsem_data_ACC_paired = samsem_data_template.copy()
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
                    # For the Kotera and the dummy method, both protan and deutan variant are the same
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
    samsem_data_RT_paired.to_csv(os.path.join(path,dict['filename']+'_samsem-data-RT-paired.csv'),sep=";")
    
    samsem_data_ACC_paired =  samsem_data_ACC_paired[columns]
    samsem_data_ACC_paired.observer_id = samsem_data_ACC_paired.observer_id.astype(int)
    samsem_data_ACC_paired.observer_coldef_type = samsem_data_ACC_paired.observer_coldef_type.astype(int)
    samsem_data_ACC_paired.image_id = samsem_data_ACC_paired.image_id.astype(int)
    samsem_data_ACC_paired.coldef_type = samsem_data_ACC_paired.coldef_type.astype(int)
    samsem_data_ACC_paired.to_csv(os.path.join(path,dict['filename']+'_samsem-data-ACC-paired.csv'),sep=";")
    
    f = open(os.path.join(path,dict['filename']+'_samsem-data-paired_meta-data.txt'), 'w')
    json.dump(sim_ids, f); f.close()


def makePairedDataForViSDEM(visdem_data,path,dict):
    """
    Make paired data for all observers and all images collapsed of all algorithms individually.
    The result images are stored in (subfolders) defined by the path input.
    Input: * samsem_data:    SaMSEM data that is to be analyzed
           * path:           Path to the folder, where the result images should be stored.
    """
    
    #print dict
    #print visdem_data
    # 1. Restrict data to only one type of observer color deficiency
    obs_groups = dict['obs_groups']
    #print
    #coldef_type = dict['coldef_type']
    #print coldef_type
    #observer_coldef_type = dict['observer_coldef_type']
    
    
    #print obs_ids_sheet
    
    
    visdem_data_restr = pandas.DataFrame()
    for group_coldef_type, value in obs_groups.items():
        whatArr_tmp = [['observer_coldef_type',operator.eq,group_coldef_type],['variant_id',operator.eq,1]]
        visdem_data_restr_tmp = organizeArray(visdem_data,whatArr_tmp)
        visdem_data_restr = pandas.concat([visdem_data_restr_tmp,visdem_data_restr])
    visdem_data_restr.reset_index()

    dalt_ids = sorted(set(visdem_data_restr['dalt_id'].values.astype(int)))
    observer_ids = sorted(set(visdem_data_restr['observer_id'].values.astype(int)))
    
    columns=['observer_id', 'observer_coldef_type', 'coldef_type', 'set_id', 'motive_id', 'image_id', 'variant_id']
    for dalt_id in dalt_ids:
        col_tmp = "dalt_id_"+str(dalt_id).zfill(2)
        columns.append(col_tmp)
    
    #observer_coldef_types = set(visdem_data_restr['observer_coldef_type'].values)
    #print observer_coldef_types
        
    visdem_data_template = pandas.DataFrame(columns=columns)
    visdem_data_RT_paired = visdem_data_template.copy()
    visdem_data_ACC_paired = visdem_data_template.copy()
    index = 0

    for observer_id in observer_ids:
        whatArray_tmp = [['observer_id',operator.eq,observer_id]]
        visdem_data_restr_obs = organizeArray(visdem_data_restr, whatArray_tmp)
        
        obs_coldef_type = getColDefTypeForObserver(observer_id)
        ass_coldef_type = obs_groups[obs_coldef_type] # assigned coldef tyoe for particular observer group
        #print observer_id, obs_coldef_type, ass_coldef_type
        
        set_ids = sorted(set(visdem_data_restr_obs['set_id'].values.astype(int)))
        
        for set_id in set_ids:
            whatArray_tmp = [['set_id', operator.eq, set_id]]
            visdem_data_restr_set = organizeArray(visdem_data_restr_obs, whatArray_tmp)
            motive_ids = sorted(set(visdem_data_restr_set['motive_id'].values.astype(int)))

            
            for motive_id in motive_ids:
                whatArray_tmp = [['motive_id',operator.eq,motive_id],['variant_id', operator.eq,1]]
                visdem_data_restr_motive =organizeArray(visdem_data_restr_set, whatArray_tmp)
                
                pandas_tmp = pandas.DataFrame({'observer_id': observer_id,
                                                  'observer_coldef_type': obs_coldef_type,
                                                  'coldef_type': ass_coldef_type,
                                                  'set_id': set_id,
                                                  'motive_id': motive_id,
                                                  'image_id': 'nix',
                                                  'variant_id': 'nix',
                                                  'dalt_id': 'nix'
                                                  },[index])
                pandas_RT_tmp = pandas_tmp.copy(); pandas_ACC_tmp = pandas_tmp.copy()
                
                tmp_cdt = 'a'; tmp_var = 'v'
                for dalt_id in dalt_ids:
                    if dalt_id not in [0,99]: whatArray_tmp = [['dalt_id',operator.eq,dalt_id],['coldef_type',operator.eq,ass_coldef_type]]
                    else: whatArray_tmp = [['dalt_id',operator.eq,dalt_id]]
                    
                    field = organizeArray(visdem_data_restr_motive, whatArray_tmp).reset_index().loc[0]
                    
                    if not field.empty:
                        RT_tmp = field['resp_time']*1000 if bool(field['is_correct']) else float('NaN')
                        ACC_tmp = field['is_correct']
                        
                        image_id = field['image_id']; pandas_RT_tmp['image_id'] = image_id; pandas_ACC_tmp['image_id'] = image_id
                        tmp_var += str(field['variant_id'])
                        tmp_cdt += str(field['coldef_type'])
                    else: RT_tmp = float('NaN'); ACC_tmp = float('NaN')
                    
                    pandas_tmp['coldef_type'] = str(tmp_cdt); pandas_tmp['variant_id'] = str(tmp_var)
                        
                    pandas_RT_tmp["dalt_id_"+str(dalt_id).zfill(2)]= float(RT_tmp)
                    pandas_ACC_tmp["dalt_id_"+str(dalt_id).zfill(2)]= ACC_tmp
                    
                    
                visdem_data_RT_paired = visdem_data_RT_paired.append(pandas_RT_tmp)
                visdem_data_ACC_paired = visdem_data_ACC_paired.append(pandas_ACC_tmp)
                index += 1
                
    # Layout RT for storage in path
    visdem_data_RT_paired = visdem_data_RT_paired[columns]
    visdem_data_RT_paired.observer_id = visdem_data_RT_paired.observer_id.astype(int)
    visdem_data_RT_paired.observer_coldef_type = visdem_data_RT_paired.observer_coldef_type.astype(int)
    visdem_data_RT_paired.set_id = visdem_data_RT_paired.set_id.astype(int)
    visdem_data_RT_paired.motive_id = visdem_data_RT_paired.motive_id.astype(int)
    visdem_data_RT_paired.image_id = visdem_data_RT_paired.image_id.astype(int)
    visdem_data_RT_paired.coldef_type = visdem_data_RT_paired.coldef_type.astype(str)
    visdem_data_RT_paired.to_csv(os.path.join(path,dict['filename']+'_visdem-data-RT-paired.csv'),sep=";")
    
    # Layout ACC for storage in path
    visdem_data_ACC_paired = visdem_data_ACC_paired[columns]
    visdem_data_ACC_paired.observer_id = visdem_data_ACC_paired.observer_id.astype(int)
    visdem_data_ACC_paired.observer_coldef_type = visdem_data_ACC_paired.observer_coldef_type.astype(int)
    visdem_data_ACC_paired.set_id = visdem_data_ACC_paired.set_id.astype(int)
    visdem_data_ACC_paired.motive_id = visdem_data_ACC_paired.motive_id.astype(int)
    visdem_data_ACC_paired.image_id = visdem_data_ACC_paired.image_id.astype(int)
    visdem_data_ACC_paired.coldef_type = visdem_data_ACC_paired.coldef_type.astype(str)
    visdem_data_ACC_paired.to_csv(os.path.join(path,dict['filename']+'_visdem-data-ACC-paired.csv'),sep=";")
    
    f = open(os.path.join(path,dict['filename']+'_visdem-data-paired_meta-data.txt'), 'w')
    json.dump(dalt_ids, f); f.close()
