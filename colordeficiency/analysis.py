'''
Created on 26. aug. 2014

@author: joschua
'''

import numpy
import settings
import matplotlib.pyplot as plt
import pandas
import os
import sys
import operator
from test import getAllXXXinPath, getStatsFromFilename

def getSetFromScene(sce_id):
    visualsearch_ids = "../data/visualsearch_ids.xlsx"
    vs_ids_sheet = pandas.read_excel(visualsearch_ids)
    set_id = int(vs_ids_sheet[vs_ids_sheet.scene_id==sce_id].set_id.values[0])  

def organizeArray(dataArray_in,logArray,sortArray):
    """
    Returns an adjusted dataArray according to the logical operations mentioned in the logical dictionary
    Input: 
    * logDict:      Contains the name of the columns, the value of interest and the logial operation, i.e. ["num", 3, operator.qt] means operator.qt(dataArray["num"],3)
    * dataArray:    Pandas data array with the original values.
    * sortArray:    Titles of the columns that should be extracted to show.
    Output:
    *dataArray_out: Pandas data array with the adjusted values. 
    """
    
    dataArray_out = dataArray_in.copy(); i = bool(1);
    
    for entry in logArray:
        column_tmp = entry[0]
        eval_funct = entry[1]
        data_value = entry[2]
        
        # Check if column exist
        if column_tmp in dataArray_in.columns:
            i = i & (eval_funct(dataArray_in[column_tmp],data_value))
    
    # Check if column of sorting interest exists
    sortArray_new = []
    for sort in sortArray:
        if sort in dataArray_in.columns:
            sortArray_new.append(sort)
    
    # Show only columns of interests or everything if array is empty
    dataArray_out = dataArray_out[i][sortArray_new] if sortArray_new else dataArray_out[i]
    
    return dataArray_out

def extractDataFromPsychoPyXLSX(pandasDataSheet):
    """
    This function opens an XLSX file from a file and extracts relevant data and the additional data at the bottom.
    """
    
    # Find line, which only has NaN data because this is the last line with relevant data
    first_column_ID = pandasDataSheet.columns[0] # Get the name of the first column
    end_index = pandasDataSheet[pandasDataSheet[first_column_ID].isnull()].index[0] # Get all the rows that have null respectively NaN values and take the first, which separates the primary data from the additional data.
    relevant_data = pandasDataSheet[0:end_index]
    # Extra extra data and convert into dictionary for easier processing
    extra_data = dict(pandasDataSheet.iloc[end_index+2:-1,0:2].values)
    
    return relevant_data, extra_data

def extractExperimentData(dataSheet,dataArray=pandas.DataFrame(),testArray=pandas.DataFrame()):
    """
    This function takes the url an xlsx file containing the result data from a sample-to-match experiment, extracts the data and returns the contained information as numpy array.
    """
    
    # Get all the sheet names for the excel file
    
    sheet_names = pandas.ExcelFile(dataSheet).sheet_names
    #print sheet_names
    # Create empty array for test and relevatn data. Create empty dictionary for the extra data
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
                # Append test or relevant data to respective data array
                if ("testTrials" in sheet_name) or ("practTrials" in sheet_name): testArray = pandas.concat([array_tmp, testArray]) if not testArray.empty else array_tmp
                else: dataArray = pandas.concat([array_tmp,dataArray]) if not dataArray.empty else array_tmp
    
    dataArray = dataArray.reset_index()
    testArray = testArray.reset_index()
            
    return dataArray, testArray, extraDataDict
    
# def extractVisualSearchData(dataSheet,dataArray=pandas.DataFrame(),testArray=pandas.DataFrame()):
#     """
#     This function takes the url an xlsx file containing the result data from a visual search experiment, extracts the data and returns the contained information as numpy array.
#     Input:
#     * dataSheet:     URL to XLSX data sheet containing the information.
#     * dataArray:     Pandas data array containing information from previous experiments, to which the new data is concatenated. Empty array by default.
#     Output:
#     * dataArray:     Pandas data array containing the new relevant information (plus the optional previously existed data).
#     * testArray:     Pandas data array containing the new test information (plus the optional previously existed test data)
#     * extraDataDict: Dictionary containing the additional data from the experiments
#     """
#     
#     #Get all the sheet names from the excel file
#     sheet_names = pandas.ExcelFile(dataSheet).sheet_names
#     # Create empty array for test and relevant datam create empty dictionary for the extra data
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
#                 # Append test or relevant data to respective data array
#                 if 'practTrials' in sheet_name: testArray = pandas.concat([array_tmp,testArray]) if not testArray.empty else array_tmp
#                 else: dataArray = pandas.concat([array_tmp,dataArray]) if not dataArray.empty else array_tmp
#     
#     testArray = testArray.reset_index() 
#     dataArray = dataArray.reset_index()
#     
#     return dataArray, testArray, extraDataDict

def analyzeVisualSearchData(path):
    """
    This function analyzes the data from the visual searach experiment as can be found in the path.
    """
    
    # 0. Step: Get all the relevant information, i.e. scene_ids, obs_col_defs etc.
    visualsearch_ids = "../data/visualsearch_ids.csv"
    vs_ids_sheet = pandas.read_csv(visualsearch_ids,sep=';')    
    
    observer_ids = "../data/observer_ids.csv"
    obs_ids_sheet = pandas.read_csv(observer_ids,sep=';')
    
    
    # 1. Step: Read all the XLSX data in the path
    ext = 'xlsx'; xlsx_files = getAllXXXinPath(path,ext)
    dataArray = pandas.DataFrame()
    for xlsx_file in xlsx_files:
        sys.stdout.write(xlsx_file)
        dataArray_tmp, testArray, extraDataDict = extractExperimentData(os.path.join(path,xlsx_file))
        
        newDataArray = dataArray_tmp[['dalt_id','coldef_type','resp.corr_raw','resp.rt_raw','stimFile']]
        
        if extraDataDict.has_key("2. Session"):
            sessionID = int(extraDataDict['2. Session'])
            newDataArray['session_id'] = sessionID
            
        if extraDataDict.has_key('0. Participant ID'):
            obsID = int(extraDataDict['0. Participant ID'])
        newDataArray['observer_id'] = obsID
        obs_coldef_type = obs_ids_sheet.loc[obs_ids_sheet['observer_id']==obsID,['observer_coldef_type']]
        newDataArray['observer_coldef_type'] = int(obs_coldef_type['observer_coldef_type'])
        
        dataArray = pandas.concat([dataArray, newDataArray])
        sys.stdout.write(' . ')
    sys.stdout.write('\n')
    #print testArray    
    
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
        #print filename
        dict_tmp = getStatsFromFilename(filename)
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
    
    dataArray = dataArray[['image_id','set_id','scene_id','version_id','dalt_id','coldef_type','is_correct','resp_time','observer_id','observer_coldef_type','session_id','filepath']]
    
    # 3. Saving data to file
    try:
        dataArray.to_csv('../data/visual-search-data.csv',sep=";")
        print "Success: Visual search data saved"
    except Exception as e:
        print e    
 

def plotVisualSearchData(path,visualSearchDataPath):
    
    """
    Input:  * path                 - Path where the plots should be saved
            * visualSearchDataPath - Path to the CSV-file containing Pandas array with the visual search data.
    """
    
    visual_search_data = pandas.read_csv(visualSearchDataPath,index_col=False,header=0,sep=';')
    #print visual_search_data
    
    """
    # Compute img#1-4 from https://docs.google.com/document/d/1w305_EUYiReLrQ34H-teJwsNICXvRK-cv7E6TgVfukM/edit#heading=h.khuve23asvax
    dict = {'result_id': 1, 'obs_coldef_type': str(0), 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All norm.sight.obs.' }; vsplots1thru4(visual_search_data, path, dict)
    dict = {'result_id': 2, 'obs_coldef_type': str(9), 'obs_operator': ['observer_coldef_type',operator.gt,0], 'obs_title': 'All col.def.obs.' }; vsplots1thru4(visual_search_data, path, dict) 
    dict = {'result_id': 3, 'obs_coldef_type': str(1), 'obs_operator': ['observer_coldef_type',operator.eq,1], 'obs_title': 'All prot.obs.' }; vsplots1thru4(visual_search_data, path, dict)
    dict = {'result_id': 4, 'obs_coldef_type': str(2), 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All deut.obs.' }; vsplots1thru4(visual_search_data, path, dict)
    
    # Compute img#5-8 from https://docs.google.com/document/d/1w305_EUYiReLrQ34H-teJwsNICXvRK-cv7E6TgVfukM/edit#heading=h.khuve23asvax
    dict = {'result_id': 5, 'obs_coldef_type': str(0), 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All norm.sight.obs.' }; vsplots5thru8(visual_search_data, path, dict)
    dict = {'result_id': 6, 'obs_coldef_type': str(9), 'obs_operator': ['observer_coldef_type',operator.gt,0], 'obs_title': 'All col.def.obs.' }; vsplots5thru8(visual_search_data, path, dict) 
    dict = {'result_id': 7, 'obs_coldef_type': str(1), 'obs_operator': ['observer_coldef_type',operator.eq,1], 'obs_title': 'All prot.obs.' }; vsplots5thru8(visual_search_data, path, dict)
    dict = {'result_id': 8, 'obs_coldef_type': str(2), 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All deut.obs.' }; vsplots5thru8(visual_search_data, path, dict)
    """
    
    # Compute img#67-70 from https://docs.google.com/document/d/1w305_EUYiReLrQ34H-teJwsNICXvRK-cv7E6TgVfukM/edit#heading=h.khuve23asvax
    dict = {'result_id': 67, 'obs_coldef_type': str(0), 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All sets for all norm.sight.obs.', 'filename': 'V-00099909999999' }; vsplots67thru70(visual_search_data, path, dict)
    dict = {'result_id': 68, 'obs_coldef_type': str(9), 'obs_operator': ['observer_coldef_type',operator.gt,0], 'obs_title': 'All sets for all col.def.obs.', 'filename': 'V-00099999999999' }; vsplots67thru70(visual_search_data, path, dict)
    dict = {'result_id': 69, 'obs_coldef_type': str(1), 'obs_operator': ['observer_coldef_type',operator.eq,1], 'obs_title': 'All sets for all prot.obs.', 'filename': 'V-00099919999999' }; vsplots67thru70(visual_search_data, path, dict)
    dict = {'result_id': 70, 'obs_coldef_type': str(2), 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All sets for all deut.obs.', 'filename': 'V-00099929999999' }; vsplots67thru70(visual_search_data, path, dict)
    

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

from scipy.stats import ttest_ind,f_oneway
#import rpy2.robjects as R
import math

def getAccuracy(data):
    
    #print data
    num_total =  data.values.size
    num_correct = data[data['is_correct']==True].values.size; #print num_correct
    num_incorrect = data[data['is_correct']==False].values.size; #print num_incorrect
    #print num_correct
    #print num_incorrect
    if data.values.size:
        acc = float(num_correct)/float(num_total)
        se = math.sqrt((acc)*(1-acc)/float(num_total))
        lb = acc-1.96*se # lower bound
        ub = acc+1.96*se # upper bound
        return [acc,lb,ub]
    else:
        return 0
    
def vsplots67thru70(visual_search_data,path,dict):
    sys.stdout.write("Starting Res#"+str(dict['result_id'])+' -> ')    
    if not os.path.exists(os.path.join(path,str(dict['result_id']))): os.makedirs(os.path.join(path,str(dict['result_id'])))
    
    DNData = pandas.DataFrame(); DEData = pandas.DataFrame(); DCData = pandas.DataFrame()
    set_ids = [1,2,3,5,6,8,9,10]
    #set_ids = [1]
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
    
    
    #DND_acc = getAccuracy(DNData)
    #print DCData[DCData['is_correct']==True]
    # Reponse time plots
    boxes_tmp = [];labels_tmp = []
    DND_values = DNData[DNData['is_correct']==True]['resp_time'].values; labels_tmp.append('DN') if DND_values.size else labels_tmp.append('DN - No data'); boxes_tmp.append(DND_values)
    DED_values = DEData[DEData['is_correct']==True]['resp_time'].values; labels_tmp.append('DE') if DED_values.size else labels_tmp.append('DE - No data'); boxes_tmp.append(DED_values)
    DCD_values = DCData[DCData['is_correct']==True]['resp_time'].values; labels_tmp.append('DC') if DCD_values.size else labels_tmp.append('DC - No data'); boxes_tmp.append(DCD_values)
    
    #res = R.r['t.test'](DND_values,DED_values)
    
    #p_value = ttest_ind(DND_values,DCD_values)      
    #sys.stdout.write('p-value that DND and DCD are identical: '+str(p_value))  
            
    plt.figure(); plt.boxplot(boxes_tmp, notch=1)
    plt.xticks([1,2,3],labels_tmp); plt.title(dict['obs_title']); plt.ylabel('response time'); plt.ylim([0,3])
    plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-R.png")); plt.close()
    
    # Accuracy plots
    accuracies = {'DN': getAccuracy(DNData),
                  'DC': getAccuracy(DCData),
                  'DE': getAccuracy(DEData)}
    #print accuracies
    acc_plots = []
    #print float(accuracies['DN'])==float('NaN')
    if accuracies['DN']:acc_plots.append(accuracies['DN'][0])
    if accuracies['DE']:acc_plots.append(accuracies['DE'][0])
    if accuracies['DC']:acc_plots.append(accuracies['DC'][0])
    sys.stdout.write(str(accuracies))
    
    if acc_plots:
        plt.figure(); plt.plot([1,2,3],acc_plots,'or'); 
        plt.xticks([1,2,3],labels_tmp); plt.title(dict['obs_title']); plt.ylabel('accuracy'); 
        plt.plot([1,1],[accuracies['DN'][1],accuracies['DN'][2]],color="red");
        plt.plot([2,2],[accuracies['DE'][1],accuracies['DE'][2]],color="red");
        plt.plot([3,3],[accuracies['DC'][1],accuracies['DC'][2]],color="red");
        plt.ylim([0,1]);plt.xlim([0,4])
        plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-A.png")); plt.close()
        
    sys.stdout.write("\n")
    
def analyzeSample2MatchData(path):
    """
    This functions analyzes the data from the sample-2-match experiment as can be found in the path
    """
    
    # 0. Step: Get all the relevant information, i.e. obs_col_defs etc.
    observer_ids = "../data/observer_ids.csv"
    obs_ids_sheet = pandas.read_csv(observer_ids,sep=";")
    
    # 1. Step: Read all the XLSX data in the path
    ext = 'xlsx'; xlsx_files = getAllXXXinPath(path,ext)
    dataArray = pandas.DataFrame()
    for xlsx_file in xlsx_files:
        if not '~' in xlsx_file:
            sys.stdout.write(xlsx_file)
            dataArray_tmp, testArray, extraDataDict = extractExperimentData(os.path.join(path,xlsx_file))
            
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
            sys.stdout.write('.')
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
            dataArray.to_csv('../data/sample-2-match-data.csv',sep=";")
            sys.stdout.write("Sample-to-match data successfully saved.")
        elif experiment_type == "visual-search":
            dataArray.to_csv('../data/visual-data.csv',sep=";")
            sys.stdout.write("Visual-search data successfully saved.")
    except Exception as e:
        print e 

def plotSample2MatchData(path,sample2MatchDataPath):
    
    """
    Input:  * path                 - Path where the plots should be saved
            * sample2MatchDataPath - Path to the CSV-file containing Pandas array with the sample-2-match data.
    """
    
    sample2match_data = pandas.read_csv(sample2MatchDataPath,sep=';')
    """
    dict = {'result_id': 1, 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All norm.sight.obs. on prot.', 'filename': '9919990999-a' }; s2mplots1thru4(sample2match_data, path, dict)
    dict = {'result_id': 2, 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All norm.sight.obs. on deut.', 'filename': '9929990999-a' }; s2mplots1thru4(sample2match_data, path, dict)
    dict = {'result_id': 3, 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.gt,0], 'obs_title': 'All coldef.obs. on prot.', 'filename': '9919999999-a' }; s2mplots1thru4(sample2match_data, path, dict)
    dict = {'result_id': 4, 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.gt,0], 'obs_title': 'All coldef.obs. on deut.', 'filename': '9929999999-a' }; s2mplots1thru4(sample2match_data, path, dict)
    
    dict = {'result_id': 9, 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.eq,1], 'obs_title': 'All prot.obs. on prot.', 'filename': '9919991999-a' }; s2mplots1thru4(sample2match_data, path, dict)
    dict = {'result_id': 10, 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.eq,1], 'obs_title': 'All prot.obs. on deut.', 'filename': '9929991999-a' }; s2mplots1thru4(sample2match_data, path, dict)
    dict = {'result_id': 11, 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All deut.obs. on prot.', 'filename': '9919992999-a' }; s2mplots1thru4(sample2match_data, path, dict)
    dict = {'result_id': 12, 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All deut.obs. on deut.', 'filename': '9929992999-a' }; s2mplots1thru4(sample2match_data, path, dict)
    """
    dict = {'result_id': 29, 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All img.s for all obs.s sim.prot.', 'filename': 'C-9919999999' }; s2mplots29and30(sample2match_data, path, dict)
    dict = {'result_id': 30, 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All img.s for all obs.s sim.deut.', 'filename': 'C-9929999999' }; s2mplots29and30(sample2match_data, path, dict)
    
    
    
def s2mplots1thru4(sample2match_data,path,dict):
    sys.stdout.write("Starting Res#"+str(dict['result_id'])+'.')    
    if not os.path.exists(os.path.join(path,str(dict['result_id']))): os.makedirs(os.path.join(path,str(dict['result_id'])))
    
    coldef_type = dict['coldef_type']
    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],dict['obs_operator']];howArr_tmp = [];
    sample2match_data_reduced =  organizeArray(sample2match_data, whatArr_tmp,howArr_tmp)
    
    boxes_tmp = []; labels_tmp = []; counter = []; i=1
    algorithm_ids = set(sample2match_data['sim_id'].values.astype(int))
    for algorithm_id in algorithm_ids:
        whatArr_tmp = [['sim_id',operator.eq,algorithm_id]];howArr_tmp=[];
        relevant_data_tmp = organizeArray(sample2match_data_reduced,whatArr_tmp,howArr_tmp)   
        
        alg_values = relevant_data_tmp['resp_time'].values; labels_tmp.append(settings.id2Sim[algorithm_id]) if alg_values.size else labels_tmp.append('No data for: '+str(algorithm_id)); boxes_tmp.append(alg_values)
        counter.append(i); i +=1
        
    plt.figure(); plt.boxplot(boxes_tmp, notch=1)
    plt.xticks(counter,labels_tmp); plt.title(dict['obs_title']); plt.ylabel('response time'); plt.ylim([0,5])
    plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+".png")); plt.close()

def plotAccuracyGraphs(accData,path,dict,order=[]):
    
    plt.figure(); plt.ylim([0,1]);plt.xlim([0,len(accData)+1]); 
    
    acc_plots = [];labels_tmp=[];howMany=[];counter=1
    
    if not order:
        for key,value in accData.iteritems():
            acc_plots.append(value[0])
            labels_tmp.append(value[3])
            
            plt.plot(counter,value[0],'or')
            plt.plot([counter,counter],[value[1],value[2]],color="blue")
            howMany.append(counter);counter+=1
    else:
        for key in order:
            value = accData[key]
            acc_plots.append(value[0])
            labels_tmp.append(value[3])
            
            plt.plot(counter,value[0],'or')
            plt.plot([counter,counter],[value[1],value[2]],color="blue")
            howMany.append(counter);counter+=1
        
    plt.xticks(howMany,labels_tmp); 
    plt.title(dict['obs_title']+' - Accuracy'); plt.ylabel('accuracy');
     
    plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-A.png")); plt.close()

def s2mplots29and30(sample2match_data,path,dict):
    sys.stdout.write("Starting Res#"+str(dict['result_id'])+'.')    
    if not os.path.exists(os.path.join(path,str(dict['result_id']))): os.makedirs(os.path.join(path,str(dict['result_id'])))
    
    resp_boxes_tmp = []; labels_tmp = []
    coldef_type = dict['coldef_type']
    
    # 1. Retrieving data for the three observer groups
    # Data for normal sighted observers
    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq,0]];howArr_tmp=[]
    normal_data = organizeArray(sample2match_data,whatArr_tmp,howArr_tmp)
    # Data for protan observers
    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq,1]];howArr_tmp=[]
    protan_data = organizeArray(sample2match_data,whatArr_tmp,howArr_tmp)
    # Data for deutan observers
    whatArr_tmp = [['coldef_type',operator.eq,coldef_type],['observer_coldef_type',operator.eq,2]];howArr_tmp=[]
    deutan_data = organizeArray(sample2match_data,whatArr_tmp,howArr_tmp)
    
    # 2. Reponse time plots
    normal_response_values = normal_data[normal_data['is_correct']==True]['resp_time'].values; labels_tmp.append('Norm.obs.') if normal_response_values.size else labels_tmp.append('Norm.obs. - No data'); resp_boxes_tmp.append(normal_response_values)
    protan_response_values = protan_data[protan_data['is_correct']==True]['resp_time'].values; labels_tmp.append('Prot.obs.') if protan_response_values.size else labels_tmp.append('Prot.obs. - No data'); resp_boxes_tmp.append(protan_response_values)
    deutan_response_values = deutan_data[deutan_data['is_correct']==True]['resp_time'].values; labels_tmp.append('Deut.obs.') if deutan_response_values.size else labels_tmp.append('Prot.obs. - No data'); resp_boxes_tmp.append(deutan_response_values)
    
    plt.figure(); plt.boxplot(resp_boxes_tmp, notch=1)
    plt.xticks([1,2,3],labels_tmp); plt.title(dict['obs_title']); plt.ylabel('response time'); plt.ylim([0,3])
    plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-R.png")); plt.close()
    
    # 3. Accuracy plots
    norm_acc = getAccuracy(normal_data)
    norm_acc.append(labels_tmp[0])
    prot_acc = getAccuracy(protan_data)
    prot_acc.append(labels_tmp[1])
    deut_acc = getAccuracy(deutan_data)
    deut_acc.append(labels_tmp[2])
    accuracies = {'norm': norm_acc,
                  'protan': prot_acc,
                  'deutan': deut_acc}
    #print accuracies
    plotAccuracyGraphs(accuracies,path,dict,order=['norm','protan','deutan'])

def plotEI2015_SaMSEM_ViSDEM():
    path = "/Users/thomas/Dropbox/01_NZT/01_PhD/02_Conferences/EI-2015_Color-Imaging-XX/01_Artikler/02_Evaluation-methods/images/"
    
    # Plots for ViSDEM
    visualSearchDataPath = '../data/visual-search-data.csv'
    visual_search_data = pandas.read_csv(visualSearchDataPath,index_col=False,header=0,sep=';')
    dict = {'result_id': '', 'obs_coldef_type': str(0), 'obs_operator': ['observer_coldef_type',operator.eq,0], 'obs_title': 'All sets for all norm.sight.obs.', 'filename': 'visdem-normal-observers' }; vsplots67thru70(visual_search_data, path, dict) #Res#67
    dict = {'result_id': '', 'obs_coldef_type': str(2), 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All sets for all deut.obs.', 'filename': 'visdem-deutan-observers' }; vsplots67thru70(visual_search_data, path, dict) #Res#70
    
    # Plots for SaMSEM
    sample2MatchDataPath = '../data/sample-2-match-data.csv'
    sample2match_data = pandas.read_csv(sample2MatchDataPath,index_col=False,sep=';')
    dict = {'result_id': '', 'coldef_type': 1, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All img.s for all obs.s sim.prot.', 'filename': 'samsem-protanopia' }; s2mplots29and30(sample2match_data, path, dict) #Res#29
    dict = {'result_id': '', 'coldef_type': 2, 'obs_operator': ['observer_coldef_type',operator.eq,2], 'obs_title': 'All img.s for all obs.s sim.deut.', 'filename': 'samsem-deuteranopia' }; s2mplots29and30(sample2match_data, path, dict) #Res#30
    
    
if 1:
    test_path = "/Users/thomas/Desktop/tull/test/visual-search/data/"
    test_file = "/Users/thomas/Desktop/tull/test/visual-search/data/11_2014_mai_16_1356.xlsx"
    #dataArray, testArray, extraDataDict = extractVisualSearchData(test_file)
    #dataArray, testArray, extraDataDict = extractExperimentData(test_file)
    #print dataArray
    #print testArray
    #print extraDataDict
    #analyzeVisualSearchData(test_path)
    #plotVisualSearchData(os.path.join("/Users/thomas/Desktop/tull/test/results",'VS-plots'),'../data/visual-search-data.csv')   
else:
    test_path = "/Users/thomas/Desktop/tull/test/sample-2-match/data/"
    test_file = "/Users/thomas/Desktop/tull/test/sample-2-match/data/11_2014_mai_16_1307.xlsx"
    #dataArray, testArray, extraDataDict = extractExperimentData(test_file)
    #print dataArray
    #print testArray
    #print extraDataDict
    #analyzeSample2MatchData(test_path)
    #plotSample2MatchData(os.path.join("/Users/thomas/Desktop/tull/test/results/",'S2M-plots'), '../data/sample-2-match-data.csv')
plotEI2015_SaMSEM_ViSDEM()
