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
    relevant_data = pandasDataSheet[0:end_index-1]
    # Extra extra data and convert into dictionary for easier processing
    extra_data = dict(pandasDataSheet.iloc[end_index+2:-1,0:2].values)
    
    return relevant_data, extra_data
    
def extractVisualSearchData(dataSheet,dataArray=pandas.DataFrame(),testArray=pandas.DataFrame()):
    """
    This function takes the url an xlsx file containing the result data from a visual search experiment, extracts the data and returns the contained information as numpy array.
    Input:
    * dataSheet:     URL to XLSX data sheet containing the information.
    * dataArray:     Pandas data array containing information from previous experiments, to which the new data is concatenated. Empty array by default.
    Output:
    * dataArray:     Pandas data array containing the new relevant information (plus the optional previously existed data).
    * testArray:     Pandas data array containing the new test information (plus the optional previously existed test data)
    * extraDataDict: Dictionary containing the additional data from the experiments
    """
    
    #Get all the sheet names from the excel file
    sheet_names = pandas.ExcelFile(dataSheet).sheet_names
    # Create empty array for test and relevant datam create empty dictionary for the extra data
    if dataArray.empty: dataArray = pandas.DataFrame(); 
    if testArray.empty: testArray = pandas.DataFrame(); 
    extraDataDict= {}
    # Read file sheet by sheet
    for sheet_name in sheet_names:    
        if not "~" in sheet_name:
            if ('practSets' in sheet_name) or ('sets' in sheet_name): pass # Ignore data about the order of sets and samples for now
            else:
                # Read excel sheet and extract data
                excel_sheet = pandas.read_excel(dataSheet, sheet_name)
                array_tmp, extraDataDict_tmp = extractDataFromPsychoPyXLSX(excel_sheet)
                # Update dictionary containing extra data
                extraDataDict.update(extraDataDict_tmp)
                # Append test or relevant data to respective data array
                if 'practTrials' in sheet_name: testArray = pandas.concat([array_tmp,dataArray]) if not testArray.empty else array_tmp
                else: dataArray = pandas.concat([array_tmp,dataArray]) if not dataArray.empty else array_tmp
            
    return dataArray, testArray, extraDataDict

def analyzeVisualSearchData(path):
    """
    This function analyzes the data from the visual searach experiment as can be found in the path.
    """
    
    # 0. Step: Get all the relevant information, i.e. scene_ids, obs_col_defs etc.
    visualsearch_ids = "../data/visualsearch_ids.xlsx"
    vs_ids_sheet = pandas.read_excel(visualsearch_ids)    
    
    observer_ids = "../data/observer_ids.xlsx"
    obs_ids_sheet = pandas.read_excel(observer_ids)
    
    
    # 1. Step: Read all the XLSX data in the path
    ext = 'xlsx'; xlsx_files = getAllXXXinPath(path,ext)
    dataArray = pandas.DataFrame()
    for xlsx_file in xlsx_files:
        sys.stdout.write(xlsx_file)
        dataArray_tmp, testArray, extraDataDict = extractVisualSearchData(os.path.join(path,xlsx_file))
        
        if extraDataDict.has_key('0. Participant ID'):
            obsID = int(extraDataDict['0. Participant ID'])
        
        newDataArray = dataArray_tmp[['dalt_id','coldef_type','resp.corr_raw','resp.rt_raw','stimFile']]
        
        newDataArray['observer_id'] = obsID
        obs_coldef_type = obs_ids_sheet.loc[obs_ids_sheet['observer_id']==obsID,['observer_coldef_type']]
        newDataArray['observer_coldef_type'] = int(obs_coldef_type['observer_coldef_type'])
        
        dataArray = pandas.concat([dataArray, newDataArray])
        sys.stdout.write(' . ')
    sys.stdout.write('\n')
    #print testArray    
    
    # 2. Step: Adapt values to programstandards
    
    for item in settings.dalt2ID:
        dataArray.loc[dataArray['dalt_id'] == item, ['dalt_id']] = settings.dalt2ID[item]
    
    for item in settings.colDefLong2ID:
        dataArray.loc[dataArray['coldef_type'] == item, ['coldef_type']] = settings.colDefLong2ID[item]
        
    dataArray = dataArray.rename(columns={'dalt_id': 'dalt_id',
                              'coldef_type': 'coldef_type',
                              'resp.corr_raw': 'is_correct',
                              'resp.rt_raw': 'resp_time',
                              'stimFile': 'filepath'})
    
    dataArray = dataArray.reset_index()
    
    for index, row in dataArray.iterrows():
        path_tmp = row['filepath']
        filename = os.path.basename(path_tmp).split('.')[0]
        #print filename
        dict_tmp = getStatsFromFilename(filename)
        imgID_tmp = int(dict_tmp['img_id'])
        #print index
        
        tempVSDataArray = vs_ids_sheet.loc[vs_ids_sheet['image_id']==imgID_tmp,['set_id','scene_id','version_id']]
        
        dataArray.at[index,'img_id'] = imgID_tmp
        dataArray.ix[index,'set_id'] = int(tempVSDataArray['set_id'])
        dataArray.ix[index,'scene_id'] = int(tempVSDataArray['scene_id'])
        dataArray.ix[index,'version_id'] = int(tempVSDataArray['version_id'])
        #dataArray.ix[index,'observer_coldef_type'] = obs_coldef_type_tmp
        
    # 3. Create array with the relevant data
    # Define columns of interest
    whatArray = [['dalt_id',operator.eq,'none'],
                 ['coldef_type',operator.eq,'normal']] # The field over which the data should be selected
    howArray = ['resp_time']#['dalt_id', 'coldef_type', 'pupsidupsi', 'resp.corr_raw', 'resp.rt_raw'] # The columns of interest that should be displayed afterwards
    dataArray_tmp = organizeArray(dataArray,whatArray,howArray)
    
    #print dataArray
    # 4. Plot Data
    #plotVisualSearchData(path,dataArray_tmp)
   
    try:
        # 5. Saving data to file
        dataArray.to_csv('../data/visual-search-data.csv')
        print "Success: Visual search data saved"
    except Exception as e:
        print e    
 

def plotVisualSearchData(path,visualSearchDataPath):
    
    """
    Input:  * path                 - Path where the plots should be saved
            * visualSearchDataPath - Path to the CSV-file containing Pandas array with the visual search data.
    """
    
    visual_search_data = pandas.read_csv(visualSearchDataPath,index_col=False,header=0)
    #print visual_search_data
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
        plt.xticks([1,2],labels_tmp); plt.title('Set '+str(scene_id)+': '+dict['obs_title']); plt.ylabel('response time'); plt.ylim([0,5])
        plt.savefig(os.path.join(path,str(dict['result_id']),filename_tmp+".png")); plt.close()
    
    sys.stdout.write('\n')
  
def extractSample2MatchData(dataSheet,dataArray=pandas.DataFrame(),testArray=pandas.DataFrame()):
    """
    This function takes the url an xlsx file containing the result data from a sample-to-match experiment, extracts the data and returns the contained information as numpy array.
    """
    pass

if 0:
    test_path = "/Users/thomas/Desktop/tull/test/visual-search/data/"
    #extractVisualSearchData(test_path)
    #analyzeVisualSearchData(test_path)
    plotVisualSearchData(os.path.join("/Users/thomas/Desktop/tull/test/visual-search/data/",'00_VS-plots'),'../data/visual-search-data.csv')   
if 1:
    test_path = "/Users/thomas/Desktop/tull/test/sample-2-match/data/"
    extractSample2MatchData(test_path)
    
