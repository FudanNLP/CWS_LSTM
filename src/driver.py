import numpy as np
import sys
import time
from CWS import CWS
from Data import Data
from MLP import MLP_LayerSetting
from LSTM import LSTM_LayerSetting
from Activation import *
if __name__ == '__main__':
    flag_toy_data = False
    squared_filter_length_limit = False
    random_seed = 1234
    alpha = 0.2
    batch_size = 20
    n_epochs = 60
    
    L2_reg = 0.0001
    HINGE_reg = 0.2
    wordVecLen = 100
    preWindowSize = 1
    surWindowSize = 2
    flag_dropout = [True]
    flag_dropout_scaleWeight = False
    use_bigram_feature = True
    windowSize = preWindowSize + surWindowSize
    if(use_bigram_feature == True):
        windowSize += windowSize - 1
    layer_types = ['LSTM']
    layer_sizes = [wordVecLen*windowSize,150]
    dropout_rates = [ 0.2]
    use_bias = True
    
    mlp_layerSetting = MLP_LayerSetting(activation = Tanh())
    lstm_layerSetting = LSTM_LayerSetting(gate_activation = Sigmoid(),cell_activation = Tanh())
    #layer_setting = [lstm_layerSetting,mlp_layerSetting]
    layer_setting = [lstm_layerSetting]
    #layer_setting[l1]
    dataSet = 'msr'
    flag_use_idioms = False
    if(flag_use_idioms == False):
        path_lookup_table = '../../PreTrainedWordEmbedding/charactor_OOVthr_50_%dv.txt' % wordVecLen
        path_train_data = '../../sighan2005/processed_wo_idioms/pro/%s_train.utf8' % dataSet
        path_test_data = '../../sighan2005/processed_wo_idioms/pro/%s_test.utf8' % dataSet
        path_dev_data = None
        if(dataSet == 'ctb6'):path_dev_data = '../../sighan2005/processed_wo_idioms/pro/%s_dev.utf8' % dataSet
    else:
        path_lookup_table = '../../PreTrainedWordEmbedding/charactor_OOVthr_50_%dv.txt' % wordVecLen
        path_train_data = '../../sighan2005/%s_train_processed.utf8' % dataSet
        path_test_data = '../../sighan2005/%s_test_processed.utf8' % dataSet
        path_dev_data = None
        if(dataSet == 'ctb6'):path_dev_data = '../../sighan2005/%s_dev_processed.utf8' % dataSet
    flag_random_lookup_table = False
    dic_label = {'B':1,'E':2,'M':3,'S':0}
    data = Data(path_lookup_table=path_lookup_table,
                wordVecLen=wordVecLen,path_train_data=path_train_data,
                path_test_data=path_test_data,path_dev_data = path_dev_data,
                flag_random_lookup_table=flag_random_lookup_table,
                dic_label=dic_label,
                use_bigram_feature=use_bigram_feature,
                random_seed=random_seed,flag_toy_data = flag_toy_data)
    seg_result_file = 'seg_result/seg_result_%s' % dataSet
    cws = CWS(alpha = alpha,
             squared_filter_length_limit=squared_filter_length_limit,
             batch_size=batch_size,
             n_epochs=n_epochs,
             seg_result_file=seg_result_file,
             L2_reg = L2_reg,
             HINGE_reg = HINGE_reg,
             wordVecLen = wordVecLen,
             preWindowSize = preWindowSize,
             surWindowSize = surWindowSize,
             flag_dropout = flag_dropout,
             flag_dropout_scaleWeight = flag_dropout_scaleWeight,
             layer_sizes=layer_sizes,
             dropout_rates=dropout_rates,
             layer_types = layer_types,
             layer_setting = layer_setting,
             data=data,
             use_bias=use_bias,
             use_bigram_feature=use_bigram_feature,
             random_seed=random_seed)
    cws.fit()
    
