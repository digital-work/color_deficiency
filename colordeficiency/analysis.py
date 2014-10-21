'''
Created on 26. aug. 2014

@author: joschua
'''

import numpy
#from openpyxl import load_workbook
import pandas
import os
import sys
import operator
from test import getAllXXXinPath


def extractDataFromXLSX(pandasDataSheet):
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
    This function takes the url an xlsx file containing the reult data from a visual search experiment, extracts the data and returns the contained information as numpy array.
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
                array_tmp, extraDataDict_tmp = extractDataFromXLSX(excel_sheet)
                # Update dictionary containing extra data
                extraDataDict.update(extraDataDict_tmp)
                # Append test or relevant data to respective data array
                if 'practTrials' in sheet_name: testArray = pandas.concat([array_tmp,dataArray]) if not testArray.empty else array_tmp
                else: dataArray = pandas.concat([array_tmp,dataArray]) if not dataArray.empty else array_tmp
            
    return dataArray, testArray, extraDataDict

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

import settings
import matplotlib.pyplot as plt

def analyzeVisualSearchData(path):
    """
    This function analyzes the data from the visual searach experiment as can be found in the path.
    """
    
    # 0. Step: Get all the relevant information, i.e. scene_ids, obs_col_defs etc.
    
    excel_sheet = pandas.read_excel(dataSheet, sheet_name)
    
    
    
    # 1. Step: Read all the XLSX data in the path
    ext = 'xlsx'; xlsx_files = getAllXXXinPath(path,ext)
    dataArray = pandas.DataFrame()
    for xlsx_file in xlsx_files:
        sys.stdout.write(xlsx_file + ' . ')
        dataArray_tmp, testArray, extraDataDict = extractVisualSearchData(os.path.join(path,xlsx_file))
        if extraDataDict.has_key('0. Participant ID'):
            obsID = int(extraDataDict['0. Participant ID'])
        dataArray = pandas.concat([dataArray, dataArray_tmp])
        
    # 2. Create array with the relevant data
    # Define columns of interest
    boxes = []; labels = []; plots = []; i = 1
    whatArray = [['dalt_id',operator.eq,'none'],
                 ['coldef_type',operator.eq,'normal']] # The field over which the data should be selected
    howArray = ['resp.rt_raw']#['dalt_id', 'coldef_type', 'pupsidupsi', 'resp.corr_raw', 'resp.rt_raw'] # The columns of interest that should be displayed afterwards
    dataArray_tmp = organizeArray(dataArray,whatArray,howArray)
    
    boxes.append(dataArray_tmp.values)
    labels.append('none')
    plots.append(i)
    i += 1
    
    #prin dataArray_tmp
    import copy
    for coldef_type in settings.coldef_types_long:
        boxes_tmp = copy.copy(boxes)
        labels_tmp = copy.copy(labels)
        plots_tmp = copy.copy(plots)
        ii = i
        
        for daltonization_type in settings.daltonization_types:
            whatArray = [['dalt_id',operator.eq,daltonization_type],
                          ['coldef_type',operator.eq,coldef_type]] # The field over which the data should be selected
            howArray = ['resp.rt_raw']#['dalt_id', 'coldef_type', 'pupsidupsi', 'resp.corr_raw', 'resp.rt_raw'] # The columns of interest that should be displayed afterwards
            dataArray_tmp = organizeArray(dataArray,whatArray,howArray)
            if not dataArray_tmp.empty:
                # Gjoer noe med denne saken
                # 3. Visualize the data
                boxes_tmp.append(dataArray_tmp.values)
                labels_tmp.append(daltonization_type)
                plots_tmp.append(ii)
                ii += 1
            else: print "Caution: No results available for " + daltonization_type
                
        plt.figure()
        plt.boxplot(boxes_tmp)
        plt.xticks(plots_tmp,labels_tmp)
        if not os.path.exists(os.path.join(path,'resultater')): os.makedirs(os.path.join(path,'resultater'))
        plt.savefig(os.path.join(path,'resultater',coldef_type+".png"))
        plt.close()

#xlsx = "dummy.xlsx"
#extractVisualSearchData(xlsx)
test_path = "/Users/thomas/Desktop/test/visual-search/data/"
analyzeVisualSearchData(test_path)

def extractSample2MatchData(dataSheet,dataArray=numpy.array([])):
    """
    This function takes the url an xlsx file containing the result data from a sample-to-match experiment, extracts the data and returns the contained information as numpy array.
    """
    pass