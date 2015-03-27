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
import copy

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

def organizeArray(dataArray_in,logArray,sortArray=[]):
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
    
    if dict.has_key('result_id'):
        path_res = os.path.join(path,str(dict['result_id']))
    else:
        path_res = path        
    
    if dict.has_key('y_lim_RT'):
        y_lim = dict['y_lim_RT']
    else:
        y_lim = [.0,1750.]
    plt.figure(); plt.ylim(y_lim);plt.xlim([0,len(meanData)+1]); plt.grid(axis='y');
    
    mean_plots = [];labels_tmp=[];se=[];howMany=[];counter=1
    
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
    if dict.has_key('obs_title'):
        plt.title(dict['obs_title']+' - CI mean');
    else:
        plt.title('')  
    plt.ylabel('Response Times (ms)');
    plt.savefig(os.path.join(path_res,dict['filename']+"-CI.pdf")) 
    plt.close()

def plotAccuracyGraphs(accData,path,dict,order=[]):
    
    if dict.has_key('y_lim_ACC'):
        y_lim = dict['y_lim_ACC']
    else:
        y_lim = [.5,1.]
    
    if dict.has_key('fontsize'):
        fontsize = dict['fontsize']
    else:
        fontsize = 12
        
    if dict.has_key('fmt'):
        fmt = dict['fmt']
    else:
        fmt = 'or'
        
    if dict.has_key('color'):
        color = dict['color']
    else:
        color = 'red'
        
    if dict.has_key('multiple_graphs'):
        multiple_graphs = dict['multiple_graphs']
    else:
        multiple_graphs = 0
        
    if dict.has_key('result_id'):
        result_id = dict['result_id']
    else:
        result_id = ''
    
    if dict.has_key('figsize'):
        figsize = dict['figsize']
    else:
        figsize = [8., 6.]
    
    if dict.has_key('xlabel'):
        xlabel = dict['xlabel']
    else:
        xlabel = ''
        
    if dict.has_key('ylabel'):
        ylabel = dict['ylabel']
    else:
        ylabel = 'Accuracy'
        
    if not multiple_graphs: 
        plt.figure() 
    fig = plt.gcf()
    plt.ylim(y_lim);plt.xlim([0,len(accData)+1]); plt.grid(axis='y');
    #plt.fontsize(fontsize)
    acc_plots = [];labels_tmp=[];se=[];howMany=[];counter=1
    
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
            
    acc_plots = numpy.array(acc_plots)
    se = numpy.array(se)
    se_int = 1.96*se
    lower_bound = acc_plots-se_int
    upper_bound = acc_plots+se_int
    plt.errorbar(howMany,acc_plots,se_int,fmt=fmt, color=color)
    plt.xticks(howMany,labels_tmp,fontsize=fontsize); 
    title_default = ''
    if dict.has_key('obs_title'):
        if not dict['obs_title']:        
            title = dict['obs_title']+' - Accuracy'
        else:
            title = title_default
    else:
        title = title_default 
    plt.title(title)
    plt.ylabel(ylabel,fontsize=fontsize);
    plt.xlabel(xlabel,fontsize=fontsize)
    fig.set_size_inches(figsize)
    plt.savefig(os.path.join(path,str(result_id),dict['filename']+"-ACC.pdf")); 
    
    if not multiple_graphs:
        plt.close()
        
    data = {'l-bounds': lower_bound,'acc': acc_plots, 'u-bounds':upper_bound}
    a = pandas.DataFrame(data, index=labels_tmp, columns=['l-bounds','acc', 'u-bounds'])
    a.to_csv(os.path.join(path,str(result_id),dict['filename']+"-ACC-bounds.csv"),sep=';')
    writePandastoLatex(a, os.path.join(path,str(result_id),dict['filename']+"-ACC-bounds.tex"))
    

def plotRTGraphs(boxes,labels,path,dict,order=[]):
    
    if dict.has_key('y_lim_RT'):
        y_lim = dict['y_lim_RT']
    else:
        y_lim = [.0,1750]
        
    if dict.has_key('obs_title'):
        obs_title = dict['obs_title']
    else:
        obs_title = ""
    
    if dict.has_key('fontsize'):
        fontsize = dict['fontsize']
    else:
        fontsize = 12
    
    if dict.has_key('result_id'):
        save_to_file = os.path.join(path,str(dict['result_id']),dict['filename']+"-RT.pdf")
    if dict.has_key('filename'):
        save_to_file = os.path.join(path,dict['filename']+"-RT.pdf")
    counter = numpy.array(range(1,numpy.shape(boxes)[0]+1))
    
    plt.figure(); plt.boxplot(boxes, notch=1)
    plt.xticks(counter,labels,fontsize=fontsize); plt.title(obs_title,fontsize=fontsize); plt.tick_params(axis='y', labelsize=fontsize);plt.tick_params(axis='x', labelsize=fontsize); plt.ylabel('Response Times (ms)', fontsize=fontsize); plt.ylim(y_lim); plt.grid(axis='y');
    plt.savefig(save_to_file); plt.close()

def plotHistogram(distribution,path,dict):
    
    if dict.has_key('bins'):
        bins = dict['bins']
    else:
        bins = 20
        
    if dict.has_key('x_lim_hist'):
        x_lim = dict['x_lim_hist']
    else:
        x_lim = [.0,1750.]
        
    if dict.has_key('filename') and dict.has_key('result_id'):
        save_to_file = os.path.join(path,str(dict['result_id']),dict['filename']+"-HIST.pdf") # This is only for samsemPlots35Thru37. Please remove it
    elif dict.has_key('filename'): 
        save_to_file = os.path.join(path,dict['filename']+"-HIST.pdf")
        
    if dict.has_key('obs_title'):
        obs_title = dict['obs_title']
    else:
        obs_title = dict['filename']
    
    plt.figure(); plt.hist(distribution, bins = bins);
    plt.title(obs_title); plt.ylabel('Response Times (ms)'); plt.xlim(x_lim); plt.grid(axis='y');
    plt.savefig(save_to_file); plt.close()

def plotResidualPlots(boxes,labels,path,dict):
    
    if dict.has_key('y_lim_RES'):
        y_lim = dict['y_lim_RES']
    else:
        y_lim = [-1750.,1750.]
    
    if dict.has_key('filename') and dict.has_key('result_id'):
        save_to_file = os.path.join(path,str(dict['result_id']),dict['filename']+"-HIST.pdf") # This is only for samsemPlots35Thru37. Please remove it
    elif dict.has_key('filename'): 
        save_to_file = os.path.join(path,dict['filename']+"-RES.pdf")
    
    
    if dict.has_key('obs_title'):
        obs_title = dict['obs_title']
    else:
        obs_title = dict['filename']
    
    counter = numpy.array(range(1,numpy.shape(boxes)[0]+1))
    res_boxes = []
    i=1
    for box in boxes:
        i +=1
        mean_tmp = stats.nanmean(box)
        residual_values_tmp = box-mean_tmp
        res_boxes.append(residual_values_tmp)
    
    plt.figure(); plt.boxplot(res_boxes, notch=1);
    plt.xticks(counter,labels); plt.title(obs_title); plt.ylabel('Response Times (ms)'); plt.ylim(y_lim); plt.grid(axis='y');
    plt.savefig(save_to_file); plt.close()

def plotQQPlot(distribution,path,dict):
    
    if dict.has_key('filename') and dict.has_key('result_id'):
        save_to_file = os.path.join(path,str(dict['result_id']),dict['filename']+"-QQ.pdf") # This is only for samsemPlots35Thru37. Please remove it
    elif dict.has_key('filename'): 
        save_to_file = os.path.join(path,dict['filename']+"-QQ.pdf")
    
    stats.probplot(distribution, dist="norm", plot=plt)
    plt.savefig(save_to_file); plt.close()
    
def getSetFromScene(sce_id):
    visualsearch_ids = "../colordeficiency-data/visualsearch_ids.xlsx"
    vs_ids_sheet = pandas.read_excel(visualsearch_ids)
    set_id = int(vs_ids_sheet[vs_ids_sheet.scene_id==sce_id].set_id.values[0]) 
    
    
def getAccuracy(data):
    
    num_total =  numpy.shape(data.values)[0]
    num_correct = numpy.shape(data[data['is_correct']==True].values)[0] 
    num_incorrect = numpy.shape(data[data['is_correct']==False].values)[0]
    
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
        num_total = numpy.shape(data)[0]
        mean = stats.nanmean(data)
        se = stats.nanstd(data)/math.sqrt(num_total)
        
        return [mean,se]
    else:
        return [.0,.0]
    
def makePearsonChi2Contingency2x2Test(data,path,methods,dict):
    """
    Veit ikkje om detter her stemmer eller gir mening.
    """
    
    num_methods = numpy.shape(methods)[0]
    num_columns = numpy.shape(data)[1]
    if not num_methods == num_columns:
        print "Error: Number of columns does not match the labels."
        return
    
    if dict.has_key('filename'):
        filename_csv = dict['filename']+"_pearson-chi2-contingency-2x2-test-matrix_p-values.csv"
        filename_tex = dict['filename']+"_pearson-chi2-contingency-2x2-test-matrix_p-values.tex"
    else:
        filename_csv = "pearson-chi2-test_p-values-matrix.csv"
        filename_tex = "pearson-chi2-test_p-values-matrix.tex"
        
    range_methods = range(num_methods)
    
    method_counter = copy.copy(range_methods)
    result_array_template = numpy.chararray(num_methods)
    result_array_template[:] = "x"
    result_array_template = numpy.array(result_array_template)
    
    template = pandas.DataFrame(columns=methods)
    template.loc[0] = result_array_template
    matrix = pandas.DataFrame(columns=methods)
    #print data
    for method in range_methods:
        method_counter.pop(0)
        values = data[:,method]
        curr_row = pandas.DataFrame.copy(template)
        
        if method_counter:
            for to_method in method_counter: # Get current to_values
                to_values = data[:,to_method]
                curr_distr = numpy.array([values,to_values])
                #print curr_distr
                chi2, p, dof, ex = stats.chi2_contingency(curr_distr) # Compare only simulation methods
                curr_row[methods[to_method]] = p
                #print ex
        matrix = matrix.append(curr_row)
    matrix.index = methods
    matrix = matrix.drop(matrix.index[[num_methods-1]])
    matrix = matrix[methods[1:num_methods]]
    matrix.to_csv(os.path.join(path,filename_csv),sep=';')
    #print matrix
    writePandastoLatex(matrix, os.path.join(path,filename_tex))
    
    
def writePandastoLatex(pandasArr,path):
    columns =  pandasArr.columns
    num_columns = numpy.shape(columns)[0]
    range_columns = sorted(range(0,num_columns))
    
    index =  pandasArr.index
    num_index = numpy.shape(index)[0]
    #print num_index
    range_index = sorted(range(0,num_index))
    
    order = "| c ||"
    header = "& "
    for i in range_columns:
        order += " c "
        header += " "+str(columns[i])+" "
        if i != num_columns-1:
            order += "|"
            header += "& "
    order += "|"
    
    res_str = "\\begin{tabular}{"+order+"}\n"
    res_str += "\t\\hline\n"
    res_str += "\t"+header+"\\\\ \\hline \\hline\n"
    #res_str += 
    counter_row = 0
    #print pandasArr.dtypes
    for index, row in  pandasArr.iterrows():
        #print index
        #print type(row[i])
        
        counter_row +=1
        #print counter_row
        res_str += "\t"+index+" & "
        for i in range_columns:
            #print row[i]
            #print type(row[i])
            if type(row[i]) == str:
                res_str += row[i]+" "
            elif (type(row[i]) == float) or (type(row[i])==numpy.float64):
                res_str += "%.4f"%(row[i])+" "
            elif (type(row[i]) == int) or (type(row[i]) == numpy.int64):
                #print 'hiersimmer'
                res_str += str(int(row[i]))+" "
            if i != num_columns-1:
                res_str += "& "
        if counter_row != num_index:   
            res_str += "\\\\ \\hline\n"
        else:
            res_str += "\\\\ \n"
    res_str += "\t\\hline\n"
    res_str += "\\end{tabular}\n"
    
    text_file = open(path, "w+")
    text_file.write(res_str)
    text_file.close()