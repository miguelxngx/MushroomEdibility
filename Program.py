import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import random

from absl import app
from collections import OrderedDict
from random import seed
from random import randrange
from csv import reader
from math import exp
from keras.utils import to_categorical
from joblib import Parallel, delayed
__test_errors__ = []
__train_errors__ = []

#Load CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader=reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return np.array(dataset)

#print array to file
def printToFile(a, file):
    f= open("./"+file+".txt","w+")
    m = np.array(a)
    for row in a:
        for element in row:
            f.write("%d " % element)
        f.write("\n")
    f.close()

#Get y to 1 or 0
def encodeYs(samples):
    y = list()
    for sample in samples:
        if sample == 'e':
            y.append(1)
        else:
            y.append(0)
    return np.array(y)

def transform_char_column_to_int(column):
    newColumn = list()
    for value in column:
        newColumn.append(ord(value))
    return newColumn

def encode_label(column):
    labelValue = {}
    new_column = []
    x = 0
    sortedColumn = list(column)
    sortedColumn.sort()
    for value in sortedColumn:
        if labelValue.get(value) == None:
            labelValue[value] = x
            x = x + 1
    for value in column:
        new_column.append(labelValue[value])
    return np.array(new_column)

def get_error(samples, params,y, __h__):
    error_acum =0
    error = 0
    for i in range(len(samples)):
        hyp = __h__[i]
        
        if(y[i] == 1): # avoid the log(0) error
            if(hyp == 0):
                hyp = 0.0001
            error = (-1) * np.log(hyp)
        if(y[i] == 0):
            if(hyp == 1):
                hyp = 0.9999
            error = (-1) * np.log(1 - hyp)
        #print( "error %f  hyp  %f  y %f " % (error, hyp,  y[i])) 
        error_acum = error_acum + error
    mean_error_param = error_acum / len(samples)
    return mean_error_param
    
def h(sample, params):
    acum = 0
    for i in range(len(params)):
        acum = acum + params[i]*sample[i]
    acum = acum*(-1)
    acum = 1/(1+ exp(acum))
    return acum

def calculate_h(train_samples, test_samples, params):
    #num_cores=mp.cpu_count()
    #__h__ = Parallel(n_jobs=num_cores, require='sharedmem')(delayed(h)(sample, params) for sample in samples)
    # i=0

    global __test_h__
    global __train_h__
    i=0
    for sample in train_samples:
        __train_h__[i] = h(sample, params)
        i = i + 1
    
    i=0
    for sample in test_samples:
        __test_h__[i] = h(sample, params)
        i = i + 1

    # num_cores=mp.cpu_count()
    # Parallel(n_jobs=num_cores, require='sharedmem')(delayed(parallel_calculate_h)(samples,params, i, num_cores) for i in range(num_cores))

# def parallel_calculate_h(samples, params, core, num_cores):
#     for i in range(int(len(samples)/num_cores)+1):
#         index = (i*num_cores) + core
#         if index >= len(samples):
#             break
#         __h__[index] = h(samples[index], params)

def GD(samples, params, y, alpha):
    temp_params=list(params)
    global __train_h__
    for i in range(len(params)):
        acum=0
        for j in range(len(samples)):
            error = __train_h__[j] - y[j]
            acum = acum + error*samples[j][i]
        temp_params[i] = params[i] - alpha*(1/len(samples))*acum
    return temp_params
    # num_cores=mp.cpu_count()
    # Parallel(n_jobs=num_cores, require='sharedmem')(delayed(parallel_GD)(samples, params, y, alpha, i, num_cores) for i in range(num_cores))

# def parallel_GD(sample, params, y, alpha, core, num_cores):
#     global __temp_params__
#     for i in range(len(__temp_params__)):
#         index = (i*num_cores) + core
#         if index >= len(__temp_params__):
#             break
#         acum = 0
#         for j in range(len(samples)):
#             error = __h__[j] - y[j]
#             acum = acum + error*samples[j][index]
#         __temp_params__[index] = params[index] - alpha*(1/len(samples))*acum

def append_column(col_num):
    global dataset
    global samples
    column = dataset[:,col_num]
    column = encode_label(column)
    encoded_column = to_categorical(column)
    samples = np.c_[samples, encoded_column]

if __name__ == "__main__":
    filename = "agaricus-lepiota.data"
    dataset = load_csv(filename)
    #np.random.shuffle(dataset)

    y = encodeYs(dataset[:,0])
    samples = np.ones(len(dataset))

    column_numbers = [5, 20, 22, 13, 3, 21]
    #column_numbers = [3, 5, 9, 13, 14, 18, 20, 21, 22]
    #column_numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    for number in column_numbers:
        append_column(number)

    params = [0.]*len(samples[0])
    print("Number of parameters: ", len(params))
    # params = [-0.36359968992289865, -0.7804092374985632, 0.307043025342549, 0.28388063997872254, 0.23336002951038579, 0.21849925797443406, -1.0065101137838095, 0.17618698972902336, 0.19331396828058994, -0.23203029481001997, 0.24306604535378656, 2.530194965134385, -2.605919656634758, -2.400858906985501, 2.5363560089870165, -0.3439108509804396, 4.167703513235124, -2.937401215145801, -0.652300554499876, -0.6574629930330406, -1.6465772343986027, 0.34774674919018156, -0.1927481665966499, -0.13132271549453267, 0.20318819192306548, 0.6637503848623367, 0.21777464311953224, 0.5083262274282084, -0.6845041769516962, 0.3218743500932876, 0.1483801070390594, -0.11948805013708752, 1.0444592049180068, -0.7850008029510199, 0.3020393036061937, -0.9250973954960775, -0.4767692127473334, -0.3439108509804396, 0.3071927430790346, 0.7316819293396559, -0.3888287155286478, 0.6360950894059747, -0.0996837418676045, -0.2616082652752854, -0.4677686653482565, -0.3439108509804396, -0.26455801652213246, 0.24486917757967286, 0.1734514292217399, -1.4627893668378933, 1.3968487154821607, 1.5726135794066722, 0.1894318323188479, -2.499299470515836, 0.9914400735353294, -0.9106864728147662, 0.1853899902808415, 0.37580042705531314, -0.46155641090635996, 1.2255760178720376, -0.266837193825725, -1.687705690950852, 0.45112316083268983, -0.10765222497338381, 0.12443041544607816, -0.16186209594640788, -0.08993966466605591, 0.10197050538337875, -0.9102478994053762, 0.6797012742388672]
    alpha = 2.2
    train_samples = samples[0:7800,:]
    train_y = y[0:7800]
    test_samples = samples[7800:,:]
    test_y = y[7800:]
    samples = samples.tolist()
    epoch = 1
    __train_h__ = [0]*len(train_samples)
    __test_h__ = [0]*len(test_samples)
    __temp_params__ = [0.]*len(samples[0])
    calculate_h(train_samples,test_samples,params)
    start_time = time.time()
    try:
        while True:
            if epoch % 128 == 0 and alpha > 0.015:
                alpha = alpha * .5
            old_params = list(params)
            # print(params)
            params = GD(train_samples,params,train_y,alpha)
            # GD(samples,params,y,alpha)
            # params = list(__temp_params__)
            calculate_h(train_samples,test_samples,params)
            train_error = get_error(train_samples,params,train_y,__train_h__)
            __train_errors__.append(train_error)
            test_error = get_error(test_samples, params, test_y,__test_h__)
            __test_errors__.append(test_error)
            print("Epoch: ", epoch, " train error: ", train_error, " test error: ", test_error)
            epoch = epoch + 1
    except KeyboardInterrupt:
        pass
    printToFile(samples, "Samples")
    print ("final params:")
    print (params)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    plt.plot(__test_errors__)
    plt.plot(__train_errors__)
    plt.show()
