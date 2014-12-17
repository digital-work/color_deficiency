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
import test_wech
#from test_wech import getAllXXXinPath, getStatsFromFilenamet


def getSetFromScene(sce_id):
    visualsearch_ids = "../colordeficiency-data/visualsearch_ids.xlsx"
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
    # Create empty array for test_wech and relevatn data. Create empty dictionary for the extra data
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
                # Append test_wech or relevant data to respective data array
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
#     * testArray:     Pandas data array containing the new test_wech information (plus the optional previously existed test_wech data)
#     * extraDataDict: Dictionary containing the additional data from the experiments
#     """
#     
#     #Get all the sheet names from the excel file
#     sheet_names = pandas.ExcelFile(dataSheet).sheet_names
#     # Create empty array for test_wech and relevant datam create empty dictionary for the extra data
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
#                 # Append test_wech or relevant data to respective data array
#                 if 'practTrials' in sheet_name: testArray = pandas.concat([array_tmp,testArray]) if not testArray.empty else array_tmp
#                 else: dataArray = pandas.concat([array_tmp,dataArray]) if not dataArray.empty else array_tmp
#     
#     testArray = testArray.reset_index() 
#     dataArray = dataArray.reset_index()
#     
#     return dataArray, testArray, extraDataDict

def analyzeVisualSearchData(dict):
    """
    This function analyzes the data from the visual searach experiment as can be found in the path.
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
    ext = 'xlsx'; xlsx_files = test_wech.getAllXXXinPath(path_in,ext)
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
        dict_tmp = test_wech.getStatsFromFilename(filename)
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
        sys.stdout.write("Success: Visual search data successfully saved in '"+str(path_out)+"'.\n")
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
        #lb = acc-1.96*se # lower bound
        #ub = acc+1.96*se # upper bound
        #return [acc,lb,ub]
        return [acc,se]
    else:
        return [.0,.0]
    
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
    
def analyzeSample2MatchData(dict):
    """
    This functions analyzes the data from the sample-2-match experiment as can be found in the path
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
    ext = 'xlsx'; xlsx_files = test_wech.getAllXXXinPath(path_in,ext)
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
            dict_tmp = test_wech.getStatsFromFilename(filename)
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
    
    if dict.has_key('y_lim'):
        y_lim = dict['y_lim']
    else:
        y_lim = [.5,1.00]
        
    plt.figure(); plt.ylim(y_lim);plt.xlim([0,len(accData)+1]); plt.grid(axis='y');
    
    acc_plots = [];labels_tmp=[];se=[];howMany=[];counter=1
    #print order
    if not order:
        for key,value in accData.iteritems():
            acc_plots.append(value[0])
            se.append(value[1])
            labels_tmp.append(value[2])
            #labels_tmp.append(value[3])
            
            #plt.plot(counter,value[0],'or')
            #plt.plot([counter,counter],[value[1],value[2]],color="blue")
            howMany.append(counter);counter+=1
    else:
        end = len(order);
        while counter <= end:
            key = order[counter]
            value = accData[key]
            acc_plots.append(value[0])
            se.append(value[1])
            labels_tmp.append(value[2])
            
            #plt.plot(counter,value[0],'or')
            #plt.plot([counter,counter],[value[1],value[2]],color="blue")
            howMany.append(counter);counter+=1
    #print se
    se = 1.96*numpy.array(se)
    plt.errorbar(howMany,acc_plots,se,fmt='or')
    plt.xticks(howMany,labels_tmp); 
    if dict['obs_title']:
        plt.title(dict['obs_title']+' - Accuracy');
    else:
        plt.title('')  
    plt.ylabel('Accuracy');
    #plt.grid('y')
    #plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-A.eps")); 
    plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-A.pdf")); 
    plt.close()

def s2mplots29and30(sample2match_data,path,dict):
    
    result_id = dict['result_id']
    intro_string = result_id if result_id else dict['filename'] 
    sys.stdout.write("Starting Res#"+str(intro_string)+'.')    
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
    normal_response_values = normal_data[normal_data['is_correct']==True]['resp_time'].values*1000; labels_tmp.append('Norm.obs.') if normal_response_values.size else labels_tmp.append('Norm.obs. - No data'); resp_boxes_tmp.append(normal_response_values)
    protan_response_values = protan_data[protan_data['is_correct']==True]['resp_time'].values*1000; labels_tmp.append('Prot.obs.') if protan_response_values.size else labels_tmp.append('Prot.obs. - No data'); resp_boxes_tmp.append(protan_response_values)
    deutan_response_values = deutan_data[deutan_data['is_correct']==True]['resp_time'].values*1000; labels_tmp.append('Deut.obs.') if deutan_response_values.size else labels_tmp.append('Prot.obs. - No data'); resp_boxes_tmp.append(deutan_response_values)
    
    plt.figure(); plt.boxplot(resp_boxes_tmp, notch=1);
    plt.xticks([1,2,3],labels_tmp); plt.title(dict['obs_title']); plt.ylabel('Response time (ms)'); 
    plt.ylim([0,3000]); plt.grid(axis='y')
    #plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-R.eps"));
    plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-R.pdf")); 
    plt.close()
    
    # 3. Save data in file
    for_bruno = pandas.DataFrame(resp_boxes_tmp)
    for_bruno = for_bruno.rename({0:labels_tmp[0],1:labels_tmp[1],2:labels_tmp[2]})
    #for_bruno.to_csv(os.path.join(path,str(dict['result_id']),dict['filename']+".csv"),sep=';')
    
    # 4. Accuracy plots
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
    plotAccuracyGraphs(accuracies,path,dict,order={1:'norm', 2:'protan', 3:'deutan'})

def writeMetaData(experimentData,dict):
    
    if dict.has_key('exp_type'):
        exp_type = dict['exp_type']
    else:
        print "Error: No experiment type has been chosen. Choose either ViSDEM or SaMSEM."
        return
    
    #if dict.has_key('path_in'):
    #    path_in = dict['path_in']
    #else:
    #    print "Error: No input path has been chosen containing the experiment data. Choose a path to the .csv file containing the data."
    #    return
    
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
    
if 0:
    test_path = "/Users/thomas/Desktop/tull/test_wech/visual-search/colordeficiency-data/"
    test_file = "/Users/thomas/Desktop/tull/test_wech/visual-search/colordeficiency-data/11_2014_mai_16_1356.xlsx"
    #dataArray, testArray, extraDataDict = extractVisualSearchData(test_file)
    #dataArray, testArray, extraDataDict = extractExperimentData(test_file)
    #print dataArray
    #print testArray
    #print extraDataDict
    analyzeVisualSearchData(test_path)
    #plotVisualSearchData(os.path.join("/Users/thomas/Desktop/tull/test_wech/results",'VS-plots'),'../colordeficiency-data/visual-search-data.csv')   
elif 0:
    test_path = "/Users/thomas/Desktop/tull/test_wech/sample-2-match/colordeficiency-data/"
    test_file = "/Users/thomas/Desktop/tull/test_wech/sample-2-match/colordeficiency-data/11_2014_mai_16_1307.xlsx"
    #dataArray, testArray, extraDataDict = extractExperimentData(test_file)
    #print dataArray
    #print testArray
    #print extraDataDict
    analyzeSample2MatchData(test_path)
    #plotSample2MatchData(os.path.join("/Users/thomas/Desktop/tull/test_wech/results/",'S2M-plots'), '../colordeficiency-data/sample-2-match-data.csv')
#plotEI2015_SaMSEM_ViSDEM()
