import matplotlib
import math


colorBlue_3pc = (41/255.0, 171/255.0, 226/255.0, 0.03)
colordarkRed  =(139/255.0, 99/255.0,  105/255.0 , 1)
colorTurquoise = (86/255.0, 200/255.0,  215/255.0 , 1)
colorYellow = (252/255.0, 202/255.0,  63/255.0 , 1)
colorGreen = (125/255.0, 210/255.0,  138/255.0 , 1)
colorGrey = (174/255.0, 141/255.0,  146/255.0 , 1)
colorBlue = (41/255.0, 171/255.0, 226/255.0, 1)
colorOrange = (255/255.0, 188/255.0, 103/255.0, 1)
colorRed = (218/255.0, 114/255.0, 126/255.0, 1)

colorVec = [colorBlue,colorOrange,colorRed ,colorGrey, colordarkRed]
colorMain = [colorBlue,colorOrange,colorRed ]

markerVec = ["^", "s", "X", "o", "D"]

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_dataframe(dataframe, varx, vary, cond_col = 'None', min_col = 0):
    """return the data from the dataframe in dictionary form 

    Args:
        dataframe ([panda dataframe]): panda dataframe
        varx ([str]): name of the column x
        vary ([str]): name of the column y
        cond_col (str, optional): Condition used in the dataframe for filtering using dataframe[cond_col].to_list(). Defaults to 'None'.
        min_col (int, optional): [description]. Defaults to 0.
    
    Returns:
        [dict_y, dict_count, list_value_x, list_value_y]: dictionary of y value, dictionary of count for each varx, list of all x, list of all y
    """
    
    dict_count = dict()
    dict_y = dict()
    
    list_value_x = dataframe[varx].to_list()
    list_value_y = dataframe[vary].to_list()
        
    list_value_x_list = list()
    list_value_y_list = list()
    
    if cond_col!='None':
        list_value_cond = dataframe[cond_col].to_list()

        for i in range(len(list_value_x)):
            if list_value_cond[i]/list_value_x[i] > min_col:
                list_value_x_list.append(list_value_x[i])
                list_value_y_list.append(list_value_y[i])
    
        list_value_x=list_value_x_list
        list_value_y=list_value_y_list
    
    
    for i in range(len(list_value_x)):
    
        xrow = list_value_x[i]
        yrow = list_value_y[i]

        if xrow in dict_y:
            dict_y[xrow].append(yrow)
            dict_count[xrow]+=1
        else:
            dict_y[xrow] = [yrow]
            dict_count[xrow] = 1
    
    return dict_y, dict_count, list_value_x, list_value_y
    
def get_average_x_y(dataframe, varx, vary, min_count = 0, SEM = False , minx = float('-inf'), cond_col = 'None', min_col = 0):
    """return the average of vary in the dataframe for all unique varx in the dataframe

    Args:
        dataframe ([panda dataframe]): panda dataframe
        varx ([str]): name of the column x
        vary ([str]): name of the column y
        min_count (int, optional): minimum count for a specifiv varx to have its vary averaged Defaults to 0.
        SEM (bool, optional): if true std is replace with SEM. Defaults to False.
        minx ([double], optional): minimum x considered. Defaults to float('-inf').
        cond_col (str, optional): Condition used in the dataframe for filtering using dataframe[cond_col].to_list(). Defaults to 'None'.
        min_col (int, optional): [description]. Defaults to 0.

    Returns:
        [x, y, std]: lists of x,y and standard variation
    """
    dict_std, dict_avg = dict(), dict()
    x,y,std = list(), list(), list()

    dict_y, dict_count, list_value_x, list_value_y = get_dataframe(dataframe, varx, vary, cond_col, min_col)

    for key, list_y in dict_y.items():
        if dict_count[key]>min_count and key>=minx:
            x.append(key)
            cmp_avg = sum(list_y)/len(list_y)
            y.append(cmp_avg)
            dict_avg[key]=cmp_avg

    for i in range(len(list_value_x)):
        xrow = list_value_x[i]
        yrow = list_value_y[i]
        
        if dict_count[xrow]>min_count and xrow>=minx:
            avg = dict_avg[xrow]
            n = dict_count[xrow]
            stdadd = ((yrow-avg)**2)/n
            if xrow in dict_std:
                dict_std[xrow]+=stdadd
            else:
                dict_std[xrow] = stdadd

    for xkey in x:
        if dict_count[xkey]>min_count and xkey>=minx:
            if SEM:
                std.append(math.sqrt(dict_std[xkey] )/math.sqrt(dict_count[xkey]) )
            else:
                std.append(math.sqrt(dict_std[xkey] ))
    
    return x, y, std

def get_quartile_x_y(dataframe, varx, vary, quartile = 95 , min_count = 0, cond_col = 'None', min_col = 0):
    """allow to output the value at quartile% when sorted for all possible varx

    Args:
        dataframe ([panda dataframe]): panda dataframe
        varx ([str]): name of the column x
        vary ([str]): name of the column y
        quartile ([double]): percentage to consider. If 95 then the closest to 95% value when sorted is given
        min_count (int, optional): minimum count for a specifiv varx to have its vary averaged Defaults to 0.
        cond_col (str, optional): Condition used in the dataframe for filtering using dataframe[cond_col].to_list(). Defaults to 'None'.
        min_col (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """

    x,y = list(), list()
    dict_y, dict_count, xrow_list, yrow_list = get_dataframe(dataframe, varx, vary, cond_col, min_col)
    
    for key, value in dict_y.items():
        if dict_count[key]>min_count:
            lst = dict_y[key]
            lst = sorted(lst)
            
            n = int(quartile/100*len(lst))
            
            if n> len(lst):
                n = len(lst)
            x.append(key)
            y.append(lst[n])

    return x, y

def get_density_x_y(dataframe, varx, cond_col = 'None', min_col = 0, check_cond = True, toPrint = False):
    """get the density of varx in the sample

    Args:
        dataframe ([panda dataframe]): panda dataframe
        varx ([str]): name of the column x
        cond_col (str, optional): Condition used in the dataframe for filtering using dataframe[cond_col].to_list(). Defaults to 'None'.
        min_col (int, optional): [description]. Defaults to 0.
        check_cond (bool, optional): [description]. If true cond_col is checked, having cond_col ='None' is equivalent to check_cond=False
        toPrint (bool, optional): [description]. If True print each row

    Returns:
        x, y, dict_count, total
    """

    dict_count = dict()
    total = 0

    x,y = list(), list()
    
    list_xrow = dataframe[varx].to_list()
    if cond_col!='None':
        list_cond = dataframe[cond_col].to_list()
    
    if check_cond :
        list_density = dataframe['density'].to_list()
    
    for i in range(len(list_xrow)):
        
        xrow = list_xrow[i]
        if toPrint:
            print(xrow)
        if cond_col!='None' and check_cond:
            if list_density[i] !=0 :
                if list_cond[i]/list_density[i]>min_col:
                    total+=1
                    if xrow in dict_count:
                        dict_count[xrow]+=1
                    else:
                        dict_count[xrow] = 1
        else:    
            total+=1
            if xrow in dict_count:
                dict_count[xrow]+=1
            else:
                dict_count[xrow] = 1

    for key, value in dict_count.items():
        x.append(key)
        y.append(value/total)
    
    return x, y, dict_count, total

def get_all(dataframe, varx, vary):
    
    x, y = list(), list()
    
    for index, row in dataframe.iterrows():

        x.append(row[varx])
        y.append(row[vary])
        
    return x, y
