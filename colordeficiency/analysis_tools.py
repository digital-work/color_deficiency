'''
Created on 13. jan. 2015

@author: joschua
'''

import matplotlib.pyplot as plt
import os
import pandas
import numpy
from scipy import stats
import math

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

def plotCIAverageGraphs(meanData,path,dict,order=[]):
    
    if dict.has_key('y_lim'):
        y_lim = dict['y_lim']
    else:
        y_lim = [1.,5000.]
     
    plt.figure(); plt.ylim(y_lim);plt.xlim([0,len(meanData)+1]); plt.grid(axis='y');
    
    mean_plots = [];labels_tmp=[];se=[];howMany=[];counter=1
    #print order
    if not order:
        for key,value in meanData.iteritems():
            mean_plots.append(value[0])
            se.append(value[1])
            labels_tmp.append(value[2])
            howMany.append(counter);counter+=1
    else:
        end = len(order);
        while counter <= end:
            key = order[counter]
            value = meanData[key]
            mean_plots.append(value[0])
            se.append(value[1])
            labels_tmp.append(value[2])
            howMany.append(counter);counter+=1
            
    se = 1.96*numpy.array(se)
    plt.errorbar(howMany,mean_plots,se,fmt='or')
    plt.xticks(howMany,labels_tmp); 
    if dict['obs_title']:
        plt.title(dict['obs_title']+' - CI mean');
    else:
        plt.title('')  
    plt.ylabel('Response Times (ms)');
    plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-CI.pdf")); 
    plt.close()

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
            howMany.append(counter);counter+=1
    else:
        end = len(order);
        while counter <= end:
            key = order[counter]
            value = accData[key]
            acc_plots.append(value[0])
            se.append(value[1])
            labels_tmp.append(value[2])
            howMany.append(counter);counter+=1
            
    se = 1.96*numpy.array(se)
    plt.errorbar(howMany,acc_plots,se,fmt='or')
    plt.xticks(howMany,labels_tmp); 
    if dict['obs_title']:
        plt.title(dict['obs_title']+' - Accuracy');
    else:
        plt.title('')  
    plt.ylabel('Accuracy');
    plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-A.pdf")); 
    plt.close()

def plotRTGraphs(boxes,labels,path,dict,order=[]):
    
    if dict.has_key('y_lim'):
        y_lim = dict['y_lim']
    else:
        y_lim = [.0,5000]

    counter = numpy.array(range(1,numpy.size(boxes)+1))
    
    plt.figure(); plt.boxplot(boxes, notch=1)
    plt.xticks(counter,labels); plt.title(dict['obs_title']); plt.ylabel('Response Times (ms)'); plt.ylim(y_lim); plt.grid(axis='y');
    plt.savefig(os.path.join(path,str(dict['result_id']),dict['filename']+"-R.pdf")); plt.close()
    
def getSetFromScene(sce_id):
    visualsearch_ids = "../colordeficiency-data/visualsearch_ids.xlsx"
    vs_ids_sheet = pandas.read_excel(visualsearch_ids)
    set_id = int(vs_ids_sheet[vs_ids_sheet.scene_id==sce_id].set_id.values[0]) 
    
    
def getAccuracy(data):
    
    #print data
    num_total =  data.values.size
    #print num_total
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


def getCIAverage(data):
    
    if data.size:
        num_total = data.size
        mean = stats.nanmean(data)
        se = stats.nanstd(data)/math.sqrt(num_total)
        
        return [mean,se]
    else:
        return [.0,.0]