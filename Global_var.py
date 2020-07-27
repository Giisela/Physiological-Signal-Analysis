global ECG
ECG = {}

global Trigger
Trigger = {}

'''global ECG_100
ECG_100 = {}'''
global EMGZ
EMGZ = {}
global EMGMF
EMGMF = {}
global EDA
EDA = {}

global original_ECG
original_ECG = {}
global original_EMGZ
original_EMGZ = {}
global original_EMGMF
original_EMGMF = {}
global original_EDA
original_EDA = {}

global feature_ECG
feature_ECG = {}
global feature_EDA
feature_EDA = {}
global feature_EMGZ
feature_EMGZ = {}
global feature_EMGMF
feature_EMGMF = {}


global ECG_J
ECG_J = {}
global EMGZ_J
EMGZ_J = {}
global EMGMF_J
EMGMF_J = {}
global EDA_J
EDA_J = {}


global ECG_EMGZ
ECG_EMGZ = {}
global ECG_EMGMF
ECG_EMGMF = {}
global ECG_EDA
ECG_EDA = {}

global EMGZ_ECG
EMGZ_ECG = {}
global EMGZ_EMGMF
EMGZ_EMGMF = {}
global EMGZ_EDA
EMGZ_EDA = {}

global EMGMF_ECG
EMGMF_ECG = {}
global EMGMF_EMGZ
EMGMF_EMGZ = {}
global EMGMF_EDA
EMGMF_EDA = {}

global EDA_ECG
EDA_ECG = {}
global EDA_EMGZ
EDA_EMGZ = {}
global EDA_EMGMF
EDA_EMGMF = {}

global keysRead
keysRead = []
global keysRead_J
keysRead_J = []

global acc_app_val
acc_app_val = []
global precision_app_val
precision_app_val = []
global recall_app_val
recall_app_val = []
global f1_score_app_val
f1_score_app_val = []
global mean_absolut_val
mean_absolut_val = []
global specificity_app_N_val
specificity_app_N_val = []
global specificity_app_F_val
specificity_app_F_val = []
global specificity_app_H_val
specificity_app_H_val = []
global sensitivity_app_N_val
sensitivity_app_N_val = []
global sensitivity_app_F_val
sensitivity_app_F_val = []
global sensitivity_app_H_val
sensitivity_app_H_val = []

global acc_app_test
acc_app_test = []
global precision_app_test
precision_app_test = []
global recall_app_test
recall_app_test = []
global f1_score_app_test
f1_score_app_test = []
global mean_absolut_test
mean_absolut_test = []
global specificity_app_N_test
specificity_app_N_test = []
global specificity_app_F_test
specificity_app_F_test = []
global specificity_app_H_test
specificity_app_H_test = []
global sensitivity_app_N_test
sensitivity_app_N_test = []
global sensitivity_app_F_test
sensitivity_app_F_test = []
global sensitivity_app_H_test
sensitivity_app_H_test = []


global acc_app_val_nn
acc_app_val_nn = []
global precision_app_val_nn
precision_app_val_nn = []
global recall_app_val_nn
recall_app_val_nn = []
global f1_score_app_val_nn
f1_score_app_val_nn = []
global mean_absolut_val_nn
mean_absolut_val_nn = []
global specificity_app_N_val_nn
specificity_app_N_val_nn = []
global specificity_app_F_val_nn
specificity_app_F_val_nn = []
global specificity_app_H_val_nn
specificity_app_H_val_nn = []
global sensitivity_app_N_val_nn
sensitivity_app_N_val_nn = []
global sensitivity_app_F_val_nn
sensitivity_app_F_val_nn = []
global sensitivity_app_H_val_nn
sensitivity_app_H_val_nn = []

global acc_app_test_nn
acc_app_test_nn = []
global precision_app_test_nn
precision_app_test_nn = []
global recall_app_test_nn
recall_app_test_nn = []
global f1_score_app_test_nn
f1_score_app_test_nn = []
global mean_absolut_test_nn
mean_absolut_test_nn = []
global specificity_app_N_test_nn
specificity_app_N_test_nn = []
global specificity_app_F_test_nn
specificity_app_F_test_nn = []
global specificity_app_H_test_nn
specificity_app_H_test_nn = []
global sensitivity_app_N_test_nn
sensitivity_app_N_test_nn = []
global sensitivity_app_F_test_nn
sensitivity_app_F_test_nn = []
global sensitivity_app_H_test_nn
sensitivity_app_H_test_nn= []


global acc_app_global
acc_app_global = []
global precision_app_global
precision_app_global = []
global recall_app_global
recall_app_global = []
global f1_score_app_global
f1_score_app_global = []
global mean_absolut_app_global
mean_absolut_app_global = []
global specificity_app_N_global
specificity_app_N_global = []
global specificity_app_F_global
specificity_app_F_global = []
global specificity_app_H_global
specificity_app_H_global = []
global sensitivity_app_N_global
sensitivity_app_N_global = []
global sensitivity_app_F_global
sensitivity_app_F_global = []
global sensitivity_app_H_global
sensitivity_app_H_global = []

global acc_app_global_nn
acc_app_global_nn = []
global precision_app_global_nn
precision_app_global_nn = []
global recall_app_global_nn
recall_app_global_nn = []
global f1_score_app_global_nn
f1_score_app_global_nn = []
global mean_absolut_app_global_nn
mean_absolut_app_global_nn = []
global specificity_app_N_global_nn
specificity_app_N_global_nn = []
global specificity_app_F_global_nn
specificity_app_F_global_nn = []
global specificity_app_H_global_nn
specificity_app_H_global_nn = []
global sensitivity_app_N_global_nn
sensitivity_app_N_global_nn = []
global sensitivity_app_F_global_nn
sensitivity_app_F_global_nn = []
global sensitivity_app_H_global_nn
sensitivity_app_H_global_nn = []

global keysRead_max
keysRead_max = []
global keysRead_min
keysRead_min = []
global key_N
key_N = []
global key_F
key_F = []
global key_H
key_H = []

global window_ECG
window_ECG = {}
global window_EDA
window_EDA = {}
global window_EMGZ
window_EMGZ = {}
global window_EMGMF
window_EMGMF = {}

global allFeature_ECG
allFeature_ECG = {}
global allFeature_EMGZ
allFeature_EMGZ = {}
global allFeature_EMGMF
allFeature_EMGMF = {}
global allFeature_EDA
allFeature_EDA = {}


global storeInside
storeInside = {}


global featureVector
featureVector = {}
global dict_feature
dict_feature = {}

global dict_feature_ecg
dict_feature_ecg = {}
global dict_feature_eda
dict_feature_eda = {}
global dict_feature_emgmf
dict_feature_emgmf = {}
global dict_feature_emgz
dict_feature_emgz = {}

global dict_emotion
dict_emotion = {}

global feature
feature = {}
global hrv
hrv = {}


