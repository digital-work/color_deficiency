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
import sys

##########
###
###   General methods
###
##########


def organizeArray(dataArray,whatArray,howArray=[]):
    """
    Returns an adjusted dataArray according to the logical operations mentioned in the logical dictionary
    Input: 
    * logDict:      Contains the name of the columns, the value of interest and the logial operation, i.e. ["num", 3, operator.qt] means operator.qt(dataArray["num"],3)
    * dataArray:    Pandas data array with the original values.
    * sortArray:    Titles of the columns that should be extracted to show.
    Output:
    *dataArray_out: Pandas data array with the adjusted values. 
    """
    
    dataArray_out = dataArray.copy(); i = bool(1);
    
    for entry in whatArray:
        column_tmp = entry[0]
        eval_funct = entry[1]
        data_value = entry[2]
        
        # Check if column exist
        if column_tmp in dataArray.columns:
            i = i & (eval_funct(dataArray[column_tmp],data_value))
    
    # Check if column of sorting interest exists
    sortArray_new = []
    for sort in howArray:
        if sort in dataArray.columns:
            sortArray_new.append(sort)
    
    # Show only columns of interests or everything if array is empty
    dataArray_out = dataArray_out[i][sortArray_new] if sortArray_new else dataArray_out[i]
    
    return dataArray_out


def writePandastoLatex(pandasArr,path,dict={}):
    
    tabularx = dict['tabularx'] if 'tabularx' in dict else 0
    header_table = dict['header_table'] if 'header_table' in dict else 0
    
    columns =  pandasArr.columns
    num_columns = numpy.shape(columns)[0]
    range_columns = sorted(range(0,num_columns))
    
    index =  pandasArr.index
    num_index = numpy.shape(index)[0]
    range_index = sorted(range(0,num_index))
    
    if header_table:
        order = header_table
        if pandasArr.index.name:
            header = pandasArr.index.name+" & "
        else:
            header = "& "
        for i in range_columns:
            header += " "+str(columns[i])+" "
            if i != num_columns-1:
                header += "& "
    else:
        order = "| c ||"
        if pandasArr.index.name:
            header = pandasArr.index.name+" & "
        else:
            header = "& "
        for i in range_columns:
            order += " c "
            header += " "+str(columns[i])+" "
            if i != num_columns-1:
                order += "|"
                header += "& "
        order += "|"
    
    res_str = "\\sisetup{\n"
    res_str += "detect-weight = true,\n"
    res_str += "detect-inline-weight = math\n"
    res_str += "}%\n"
    if tabularx:
        res_str += "\\begin{tabularx}{\\textwidth}{"+order+"}\n"
    else:
        res_str += "\\begin{tabular}{"+order+"}\n"
    res_str += "\t\\hline\n"
    res_str += "\t"+header+"\\\\ \\hline \\hline\n"
    counter_row = 0
    
    # Loop thru rows
    for index, row in  pandasArr.iterrows():
        counter_row +=1
        res_str += "\t"+str(index)+" & "
        
        # Loop thru cells
        for i in range_columns:
            cell = ''
            if (type(row[i]) == str) or (type(row[i]) == str):
                cell = row[i]
                
            elif (type(row[i]) == float) or (type(row[i])==numpy.float64):
                scientific_notation = True
                
                if 'scientific_notation' in dict:
                    scientific_notation = dict['scientific_notation']
                    
                if scientific_notation:
                    numerals = 2
                    value = float(row[i])
                    
                    if 'numerals' in dict:
                        numerals = int(dict['numerals']) 
                    
                    if value < 0.01:
                        cell = "\\num[round-precision="+str(numerals)+", round-mode=figures, scientific-notation=true]{"+str(float(row[i]))+"}"
                    else:
                        cell = ("%."+str(numerals)+"f")%(value)
                        
                    p_values = False
                    if 'p_values' in dict:
                        p_values = dict['p_values']
                    if p_values:
                        if value <= 0.05:
                            bkgcolor = "red"
                            if 'bkgcolor' in dict:
                                bkgcolor = dict['bkgcolor']
                            textcolor = "white"
                            if 'textcolor' in dict:
                                textcolor = dict['textcolor']
                            cell = "\\cellcolor{"+bkgcolor+"}\\textbf{\\textcolor{"+textcolor+"}{"+cell+"}}"
                    
                else:
                    cell = str(float(row[i]))
                
            elif (type(row[i]) == int) or (type(row[i]) == numpy.int64):
                cell = str(int(row[i]))
            
            res_str += cell + " "
            
            if i != num_columns-1:
                res_str += "& "
                
        if counter_row != num_index:   
            res_str += "\\\\ \\hline\n"
            
        else:
            res_str += "\\\\ \n"
            
    res_str += "\t\\hline\n"
    if tabularx:
        res_str += "\\end{tabularx}\n"
    else:
        res_str += "\\end{tabular}\n"
    
    text_file = open(path, "w+")
    text_file.write(res_str)
    text_file.close()


##########
###
###   Reading data from PsychoPy experiment
###
##########


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
            if ("roundTrials" in sheet_name) or ('practSets' in sheet_name) or ('sets' in sheet_name): 
                pass # Ignore data about the order of set and samples for now
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


def writeMetaDataOfExperiments(experimentData,path,dict):
    """
    This function returns the meta data of a SaMSEM or ViSDEM experiment like for example number of sessions, sets, simulations or daltonizations used etc.
    The information is stored inside a text file that is saved at the path defined by path_out.
    Input: experimentData:    * A pandas array with either the SaMSEM or ViSDEM data.
           dict:              * A dictionary containing all options for the computation of the meta data. 
                              Requiree are: ** exp_type: Whether the experiment was SaMSEM or ViSDEM.
                              Optional are: ** path_out: Path to the folder, where the text file containing the meta data should be stored.
    """
    
    if 'exp_type' in dict:
        exp_type = dict['exp_type']
    else:
        print("Error: No experiment type has been chosen. Choose either ViSDEM or SaMSEM.")
        return
  
    if 'filename' in dict:
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
        motives = sorted(set(experimentData['motive_id']))
        f.write('... # of motives: '+str(len(motives))+' ; '+str(motives)+'\n')
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
        print("Error: No valid experiment format has been chosen. Choose either visdem or samsem.")
        return


##########
###
###   Plotting ACC data
###
##########


def preparePandas4AccuracyPlots(pandas_dict,order_dict,c=1.96,type="normal", dict={}):
    
    #print("Type of computation for confidence intervals: "+ str(type))
    #print("Length of confidence interval c: " + str(c))
    
    telleoevelse = dict['telleoevelse'] if 'telleoevelse' in dict else 0
    accuracies = {}
    observations = []
    
    for i in range(len(order_dict)):
        key = order_dict[i]
        
        ACC_tmp, obs_tmp = getAccuracy(pandas_dict[key],c,type)
        #ACC_tmp.append(key)
        accuracies.update({key: ACC_tmp})
        observations.append(numpy.array(obs_tmp).astype(int))
    
    observations = numpy.transpose(observations)
    if telleoevelse: print("ACC plots "+dict['filename']+":\n" + str(observations))
        
    return accuracies


def getAccuracy(data,c,type):
    
    num_total =  float(numpy.shape(data.values)[0])
    num_correct = float(numpy.shape(data[data['is_correct']==True].values)[0]) 
    num_incorrect = float(numpy.shape(data[data['is_correct']==False].values)[0])
    
    #if telleoevelse: print "ACC plots "+dict['filename']+": incorr "+str(int(num_incorrect))+", corr "+str(int(num_correct))
    
    if data.values.size:
        if type == 'normal':
            p_hat = num_correct/num_total
            acc = p_hat
            se = c*math.sqrt((p_hat)*(1.0-p_hat)/num_total)
            
            lower_bound = se
            upper_bound = se
            
        elif type == 'wilson-score':
            p_hat = num_correct/num_total
            p_hat_adj = (p_hat + (c**2.)/(2.*num_total)) / (1.+(c**2.)/num_total)
            acc = p_hat
            se = c * math.sqrt(p_hat*(1.-p_hat)/num_total+(c**2.)/(4.*num_total**2.)) / (1.+(c**2.)/num_total)
            
            lower_bound = p_hat - (p_hat_adj - se)
            upper_bound = (p_hat_adj + se) - p_hat
            
        else:
            acc = float('NaN')
            lower_bound = float('NaN')
            upper_bound = float('NaN')
            
        return [acc,lower_bound,upper_bound], [num_correct, num_incorrect]
    
    else:
        return [float('NaN'),float('NaN'),float('NaN')], [num_correct, num_incorrect]


def plotAccuracyGraphs(accData,path,dict,order=[]):
    
    result_id = dict['result_id'] if 'result_id' in dict else ''
        
    multiple_graphs = dict['multiple_graphs'] if 'multiple_graphs' in dict else 0
    figsize = dict['figsize'] if 'figsize' in dict else [8., 6.]
    
    xlabel = dict['xlabel'] if 'xlabel' in dict else ''
    ylabel = dict['ylabel'] if 'ylabel' in dict else 'Accuracy'
    y_lim = dict['y_lim_ACC'] if 'y_lim_ACC' in dict else [.5,1.]
    fmt = dict['fmt'] if 'fmt' in dict else 'or'
    color = dict['color'] if 'color' in dict else 'red'
    fontsize = dict['fontsize'] if 'fontsize' in dict else 12
      
    if not multiple_graphs: 
        plt.figure() 
        
    fig = plt.gcf()
    plt.ylim(y_lim)
    plt.xlim([0,len(accData)+1]) 
    plt.grid(axis='y')
     
    acc_plots = []; lower_bound = []; upper_bound = []; labels_tmp=[]; howMany=[]; counter=1
    if not order:
        for key,value in accData.items():
            acc_plots.append(value[0])
            lower_bound.append(value[1])
            upper_bound.append(value[2])
            labels_tmp.append(value[3])
            howMany.append(counter);counter+=1
    else:
        end = len(order);
        while counter <= end:
            key = order[counter-1]
            value = accData[key]
            acc_plots.append(value[0])
            lower_bound.append(value[1])
            upper_bound.append(value[2])
            labels_tmp.append(key)
            howMany.append(counter);counter+=1
            
    acc_plots = numpy.array(acc_plots)
    bounds = numpy.array([lower_bound,upper_bound])
    
    plt.errorbar(howMany,acc_plots,bounds,fmt=fmt, color=color)
    plt.xticks(howMany,labels_tmp,fontsize=fontsize); 
    title_default = ''
    if 'obs_title' in dict:
        if dict['obs_title']:        
            title = dict['obs_title']
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
        
    data = {'l-bounds': acc_plots-lower_bound,'acc': acc_plots, 'u-bounds': acc_plots+upper_bound}
    a = pandas.DataFrame(data, index=labels_tmp, columns=['l-bounds','acc', 'u-bounds'])
    a.to_csv(os.path.join(path,str(result_id),dict['filename']+"-ACC-bounds.csv"),sep=';')
    
    writePandastoLatex(a, os.path.join(path,str(result_id),dict['filename']+"-ACC-bounds.tex"))

def plotZScoreGraphs(zs_data,path,dict,order=[]):
        
    multiple_graphs = dict['multiple_graphs'] if 'multiple_graphs' in dict else 0
    figsize = dict['figsize'] if 'figsize' in dict else [8., 6.]
    
    xlabel = dict['xlabel'] if 'xlabel' in dict else ''
    ylabel = dict['ylabel'] if 'ylabel' in dict else 'ZScores'
    y_lim = dict['y_lim_ZS'] if 'y_lim_ZS' in dict else [-1.,1.]
    fmt = dict['fmt'] if 'fmt' in dict else 'or'
    color = dict['color'] if 'color' in dict else 'red'
    fontsize = dict['fontsize'] if 'fontsize' in dict else 12
      
    if not multiple_graphs: plt.figure() 
        
    fig = plt.gcf()
    plt.ylim(y_lim)
    plt.xlim([0,len(zs_data)+1]) 
    plt.grid(axis='y')
     
    zs_plots = []; lower_bound = []; upper_bound = []; labels_tmp=[]; howMany=[]; counter=1
    if not order:
        for key,value in zs_data.items():
            zs_plots.append(value[0])
            lower_bound.append(value[1])
            upper_bound.append(value[2])
            labels_tmp.append(value[3])
            howMany.append(counter);counter+=1
    else:
        end = len(order);
        while counter <= end:
            key = order[counter-1]
            value = zs_data[key]
            zs_plots.append(value[0])
            lower_bound.append(value[1])
            upper_bound.append(value[2])
            labels_tmp.append(key)
            howMany.append(counter);counter+=1
            
    zs_plots = numpy.array(zs_plots)
    bounds = numpy.array([lower_bound,upper_bound])
    
    plt.errorbar(howMany,zs_plots,bounds,fmt=fmt, color=color)
    plt.xticks(howMany,labels_tmp,fontsize=fontsize); 
    title_default = ''
    if 'obs_title' in dict:
        if dict['obs_title']: title = dict['obs_title']
        else: title = title_default
    else: title = title_default 
    plt.title(title)
    plt.ylabel(ylabel,fontsize=fontsize);
    plt.xlabel(xlabel,fontsize=fontsize)
    fig.set_size_inches(figsize)
    plt.savefig(os.path.join(path,dict['filename']+"-ZS.pdf")); 
    
    if not multiple_graphs:
        plt.close()
    print(zs_plots)    
    data = {'l-bounds': zs_plots-lower_bound,'zs': zs_plots, 'u-bounds': zs_plots+upper_bound}
    a = pandas.DataFrame(data, index=labels_tmp, columns=['l-bounds','zs', 'u-bounds'])
    a.to_csv(os.path.join(path,dict['filename']+"-ZS-bounds.csv"),sep=';')
    writePandastoLatex(a, os.path.join(path,dict['filename']+"-ACC-bounds.tex"))


##########
###
###   Making Chi2 test
###
##########


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
    obs_pandas = pandas.DataFrame(data=data, index=['correct','incorrect'])[labels_array]
    
    return obs_array, obs_pandas
    

def makePearsonChi2Contingency(obs_array, obs_pandas, labels, path_res, dict, res=[] ):
    
    obs_adj = obs_array if not res else obs_array[:,res[0]:res[1]]
    
    try: 
        chi2, p, dof, ex = stats.chi2_contingency(obs_adj, correction=False, lambda_ = 'pearson') # Compare only simulation methods
    except:
        chi2, p, dof, ex = float('NaN'), float('NaN'), 0, float('NaN')
    
    res_str = ""
    res_str = res_str + dict['investigated-item'].capitalize()+" and observations:\n" + str(labels)
    res_str = res_str + "\n" +str(obs_array)
    if res:
        res_str = res_str + "\n\n"+dict['investigated-item'].capitalize()+" included in test:\n" + str(labels[res[0]:res[1]])
    res_str = res_str + "\n\nChi2: %f, p-value: %E, dof: %i, expect: " % (chi2, p, dof) + "\n"+str(ex)
    text_file = open(os.path.join(path_res,dict['filename']+"-chi2-contigency.txt"), "w+")
    text_file.write(res_str)
    text_file.close()
    
    writePandastoLatex(obs_pandas, os.path.join(path_res,dict['filename']+"_observations.tex"))    
 
    
def makePearsonChi2Contingency2x2Test(data,path,methods,dict):
    """
    Veit ikkje om detter her stemmer eller gir mening.
    """
    
    telleoevelse = dict['telleoevelse'] if 'telleoevelse' in dict else 0
    
    num_methods = numpy.shape(methods)[0]
    range_methods = list(range(num_methods))
    num_columns = numpy.shape(data)[1]
    if telleoevelse: "ACC Chi2 "+dict['filename']+":\n"+str(data)
    
    if not num_methods == num_columns:
        print("Error: Number of columns does not match the labels for Pearson Chi2 test.")
        return
    
    if 'filename' in dict:
        filename_csv = dict['filename']+"-chi2.csv"
        filename_tex = dict['filename']+"-chi2.tex"
    else:
        filename_csv = "-chi2.csv"
        filename_tex = "-chi2.tex"
    
    numerals = int(dict['numerals']) if 'numerals' in dict else 2
    textcolor = dict['textcolor'] if 'textcolor' in dict else 'white'
    bkgcolor = dict['bkgcolor'] if 'bkgcolor' in dict else 'red'
        
    method_counter = copy.copy(range_methods)
    result_array_template = numpy.chararray(num_methods)
    result_array_template[:] = "x"
    result_array_template = numpy.array(result_array_template)
    
    template = pandas.DataFrame(columns=methods)
    template.loc[0] = result_array_template
    matrix = pandas.DataFrame(columns=methods)
    for method in range_methods:
        method_counter.pop(0)
        values = data[:,method]
        curr_row = pandas.DataFrame.copy(template)
        
        if method_counter:
            for to_method in method_counter: # Get current to_values
                to_values = data[:,to_method]
                curr_distr = numpy.array([values,to_values])
                #print curr_distr
                try:
                    chi2, p, dof, ex = stats.chi2_contingency(curr_distr,correction=False,lambda_="pearson") # Compare only two methods
                except Exception as e:
                    p = float('NaN')
                curr_row[methods[to_method]] = p
                
        matrix = matrix.append(curr_row)
        
    matrix.index = methods
    matrix = matrix.drop(matrix.index[[num_methods-1]])
    matrix = matrix[methods[1:num_methods]]
    matrix.to_csv(os.path.join(path,filename_csv),sep=';')
    
    writePandastoLatex(matrix, os.path.join(path,filename_tex), {'scientific_notation': True, 'textcolor': textcolor, 'bkgcolor': bkgcolor, 'p_values': True, 'numerals': numerals})
    

##########
###
###   Plotting RT data
###
##########


def preparePandas4RTPlots(pandas_dict,order_dict,dict={}):
    
    telleoevelse = dict['telleoevelse'] if 'telleoevelse' in dict else 0
    
    boxes = []; labels = []
    for i in range(len(order_dict)):
        key = order_dict[i]
        
        values_tmp = pandas_dict[key][pandas_dict[key]['is_correct']==True]['resp_time'].values*1000; 
        labels.append(key) if values_tmp.size else labels.append(key + ' - No data'); 
        boxes.append(values_tmp)
        
    if telleoevelse: print("RT boxplots "+dict['filename']+": "+str([numpy.shape(i) for i in boxes]))    
    return boxes, labels

def plotRTGraphs(boxes,labels,path,dict,order=[]):
    
    telleoevelse = dict['telleoevelse'] if 'telleoevelse' in dict else 0
    
    obs_title = dict['obs_title'] if 'obs_title' in dict else ""
    y_lim = dict['y_lim_RT'] if 'y_lim_RT' in dict else [.0,1750]
    fontsize = dict['fontsize'] if 'fontsize' in dict else 12
        
    if 'result_id' in dict: save_to_file = os.path.join(path,str(dict['result_id']),dict['filename']+"-RT.pdf")
    if 'filename' in dict: save_to_file = os.path.join(path,dict['filename']+"-RT.pdf")
    
    counter = numpy.array(list(range(1,numpy.shape(boxes)[0]+1)))
    
    if telleoevelse: print("RT boxplots "+dict['filename']+": " + str([numpy.shape(i) for i in boxes]))
    
    plt.figure();
    
    plt.boxplot(boxes, notch=1)
    plt.xticks(counter,labels,fontsize=fontsize); 
    plt.title(obs_title,fontsize=fontsize); 
    plt.tick_params(axis='y', labelsize=fontsize);
    plt.tick_params(axis='x', labelsize=fontsize); 
    plt.ylabel('Response Times (ms)', fontsize=fontsize); 
    plt.ylim(y_lim); 
    plt.grid(axis='y');
    plt.savefig(save_to_file); 
    
    plt.close()

def makePairedRTDataArray(data_RT_paired, methods,id_label):
    """
    Input: * RT Pandas data frame paired
           * IDs of methods to compare
           * Label of columns being compared
    Output: * All possible combinations
            * Labels for the combinations
    """
    
    RT_paired = []; labels = [];
    
    for method in methods:
        col_tmp = id_label+"_"+str(method).zfill(2)
        values_RT_tmp = data_RT_paired[col_tmp].values
        
        RT_paired.append(values_RT_tmp)
        labels.append(method)
        """
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
        """
    return RT_paired, labels


##########
###
###   Making normality plots of RT data
###
##########


def plotQQPlot(distribution,path,dict):
    
    if 'filename' in dict and 'result_id' in dict:
        save_to_file = os.path.join(path,str(dict['result_id']),dict['filename']+"-QQ.pdf") # This is only for samsemPlots35Thru37. Please remove it
    elif 'filename' in dict: 
        save_to_file = os.path.join(path,dict['filename']+"-QQ.pdf")
    
    stats.probplot(distribution, dist="norm", plot=plt)
    plt.savefig(save_to_file); plt.close()
        

##########
###
###   General statistics methods
###
##########

    
def makeSignTest(data,path,methods):
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
    
    RT_difference = dict['RT_difference'] if 'RT_difference' in dict else 0
    telleoevelse = dict['telleoevelse'] if 'telleoevelse' in dict else 0
    
    if telleoevelse:
        print("RT median test "+dict['filename']+": "+str([numpy.size(i) for i in data]))  
    
    num_methods = numpy.shape(methods)[0]
    range_methods = list(range(num_methods))
    num_columns = numpy.shape(data)[0]
    if not num_methods == num_columns:
        print("Error: Number of columns does not match the labels for the median test. " + "Expected columns: %i, actual columns %i" % (num_methods, num_columns))
        return
    
    if 'filename' in dict:
        filename_csv = dict['filename']+"-median.csv"
        filename_latex = dict['filename']+"-median.tex"
    else:
        filename_csv = "-median.csv"
    
    numerals = int(dict['numerals']) if 'numerals' in dict else 2
    textcolor = dict['textcolor'] if 'textcolor' in dict else 'white'
    bkgcolor = dict['bkgcolor'] if 'bkgcolor' in dict else 'red'

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
                
                if RT_difference:
                    #print 1
                    #print values
                    curr_comp = values-to_values
                    if len(curr_comp) != 0:
                        curr_comp_wonan = curr_comp[~numpy.isnan(curr_comp)]
                        #print curr_comp_wonan
                        if len(curr_comp_wonan):
                            stat, p_value, m, table = stats.median_test(curr_comp_wonan,[.0],correction=False)
                        stat, p_value = stats.ttest_1samp(curr_comp_wonan,.0)
                    else:
                        p_value = "Nix"
                
                else:
                    if (len(values) !=0) and (len(to_values) != 0):
                        stat, p_value, m, table =  stats.median_test(values,to_values,correction=False)
                    else:
                        p_value = "Nix"
                        
                curr_row[methods[to_method]] = p_value
                
        matrix = matrix.append(curr_row)
        
    matrix.index = methods
    matrix = matrix.drop(matrix.index[[num_methods-1]])
    matrix = matrix[methods[1:num_methods]]
    #print matrix
    matrix.to_csv(os.path.join(path,filename_csv),sep=';')

    writePandastoLatex(matrix,os.path.join(path,filename_latex), {'scientific_notation': True, 'textcolor': textcolor, 'bkgcolor': bkgcolor, 'p_values': True, 'numerals': numerals})

def makePairwiseStudentTTest(data,path,methods,dict):
    """
    Input: * data:    Numpy data array with relevant data that should be compared
           * path:    Path, to which the results should be save to as matrix
           * labels:  Name of the columns in the pandas data set, which should be analyzed
    Output
    """
    
    ACC_difference = dict['ACC_difference'] if 'ACC_difference' in dict else 0
    telleoevelse = dict['telleoevelse'] if 'telleoevelse' in dict else 0
    
    if telleoevelse:
        print("ACC student t "+dict['filename']+": "+str([numpy.size(i) for i in data]))  
    
    num_methods = numpy.shape(methods)[0]
    range_methods = list(range(num_methods))
    num_columns = numpy.shape(data)[0]
    if not num_methods == num_columns:
        print("Error: Number of columns does not match the labels for the student t test. " + "Expected columns: %i, actual columns %i" % (num_methods, num_columns))
        return
    
    if 'filename' in dict:
        filename_csv = dict['filename']+"-student-t.csv"
        filename_latex = dict['filename']+"-student-t.tex"
    else:
        filename_csv = "-student-t.csv"
        filename_latex = "-student-t.tex"
    
    numerals = int(dict['numerals']) if 'numerals' in dict else 2
    textcolor = dict['textcolor'] if 'textcolor' in dict else 'white'
    bkgcolor = dict['bkgcolor'] if 'bkgcolor' in dict else 'red'

    method_counter = copy.copy(range_methods)
    result_array_template = numpy.chararray(num_methods)
    result_array_template[:] = "x"
    result_array_template = numpy.array(result_array_template)
    
    template = pandas.DataFrame(columns=methods)
    template.loc[0] = result_array_template
    matrix = pandas.DataFrame(columns=methods)
    
    for method in range_methods:
        values = data[method]
        method_counter.pop(0)
        
        curr_row = pandas.DataFrame.copy(template)
        if method_counter:
            for to_method in method_counter: # Get current to_values
                to_values = data[to_method]
                
                if ACC_difference:
                    curr_comp = numpy.greater(values,to_values)*(-1.0) + numpy.less(values,to_values)*(1.0)
                    
                    if len(curr_comp) != 0:
                        t, p_value = stats.ttest_1samp(curr_comp,.0)
                    else:
                        p_value = "Nix"
                else:
                    if (len(values) !=0) and (len(to_values) != 0):
                        t, p_value = stats.ttest_ind(values,to_values)
                        
                    else:
                        p_value = "Nix"
                    
                curr_row[methods[to_method]] = p_value
                
        matrix = matrix.append(curr_row)
        
    matrix.index = methods
    matrix = matrix.drop(matrix.index[[num_methods-1]])
    matrix = matrix[methods[1:num_methods]]
    matrix.to_csv(os.path.join(path,filename_csv),sep=';')
    
    #print numpy.mean(data,1)
    writePandastoLatex(matrix,os.path.join(path,filename_latex), {'scientific_notation': True, 'textcolor': textcolor, 'bkgcolor': bkgcolor, 'p_values': True, 'numerals': numerals})



##########
###
###   Obsolete functions
###
##########


def getCIAverage(data):
    
    if data.size:
        num_total = numpy.shape(data)[0]
        mean = stats.nanmean(data)
        se = stats.nanstd(data)/math.sqrt(num_total)
        
        return [mean,se]
    else:
        return [.0,.0]


def plotCIAverageGraphs(meanData,path,dict,order=[]):
    
    path_res = os.path.join(path,str(dict['result_id'])) if 'result_id' in dict else path      
    
    y_lim = dict['y_lim_RT'] if 'y_lim_RT' in dict else [.0,1750.]
    
    
    mean_plots = [];labels_tmp=[];se=[];howMany=[];counter=0
    if not order:
        for key,value in meanData.items():
            mean_plots.append(value[0])
            se.append(value[1])
            labels_tmp.append(value[2])
            howMany.append(counter);counter+=1
    else:
        end = len(order);
        while counter < end:
            key = order[counter]
            value = meanData[key]
            mean_plots.append(value[0])
            se.append(value[1])
            labels_tmp.append(value[2])
            howMany.append(counter);counter+=1
    
    plt.figure(); 
    plt.ylim(y_lim);
    plt.xlim([0,len(meanData)+1]); 
    plt.grid(axis='y');
    se = 1.96*numpy.array(se)
    plt.errorbar(howMany,mean_plots,se,fmt='or')
    plt.xticks(howMany,labels_tmp); 
    if 'obs_title' in dict:
        plt.title(dict['obs_title']+' - CI mean');
    else:
        plt.title('')  
    plt.ylabel('Response Times (ms)');
    plt.savefig(os.path.join(path_res,dict['filename']+"-CI.pdf")) 
    plt.close()
    

def plotHistogram(distribution,path,dict):
    
    bins = dict['bins'] if 'bins' in dict else 20
    x_lim = dict['x_lim_hist'] if 'x_lim_hist' in dict else [.0,1750.]
    obs_title = dict['obs_title'] if 'obs_title' in dict else dict['filename']
        
    if 'filename' in dict and 'result_id' in dict:
        save_to_file = os.path.join(path,str(dict['result_id']),dict['filename']+"-HIST.pdf") # This is only for samsemPlots35Thru37. Please remove it
    elif 'filename' in dict: 
        save_to_file = os.path.join(path,dict['filename']+"-HIST.pdf")
    
    plt.figure(); plt.hist(distribution, bins = bins);
    plt.title(obs_title); plt.ylabel('Response Times (ms)'); plt.xlim(x_lim); plt.grid(axis='y');
    plt.savefig(save_to_file); plt.close()


def plotResidualPlots(boxes,labels,path,dict):
    
    if 'y_lim_RES' in dict:
        y_lim = dict['y_lim_RES']
    else:
        y_lim = [-1750.,1750.]
    
    if 'filename' in dict and 'result_id' in dict:
        save_to_file = os.path.join(path,str(dict['result_id']),dict['filename']+"-HIST.pdf") # This is only for samsemPlots35Thru37. Please remove it
    elif 'filename' in dict: 
        save_to_file = os.path.join(path,dict['filename']+"-RES.pdf")
    
    
    if 'obs_title' in dict:
        obs_title = dict['obs_title']
    else:
        obs_title = dict['filename']
    
    counter = numpy.array(list(range(1,numpy.shape(boxes)[0]+1)))
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
    
    
def checkNormality(distributions,labels,path,options,project_str="",plot_types={'q-q','hist','boxplot','residuals'},isLog = False):
    """
    Similar to samsemPlots35thru37
    """
    if project_str:
        folder_name = project_str.lower().replace(" ","-")+"_normality-check"
        project = project_str
    else:
        folder_name = "new-normality-check"
        project = "New normality check"
    print("* Start checking normality of \'"+str(project)+"\'")
    
    # 1. Check that dimensions of distribtions and labels are the same
    distr_size = numpy.shape(distributions)
    labels_size = numpy.shape(labels)
    try:
        pass
    except:
        print("Error: Dimensions of labels and distributions have to match.")
        return
    
    # 2. Check that all plot types are accepted
    try:
        plot_types_acc = plot_types
    except:
        print("Caution: Plot type has to be one of the following: \'q-q\', \'hist\', \'boxplot\' or \'residuals\'")
    
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
    
    print("* Stop checking normality")