############# library ##########
import os
import os.path
import pandas as pd
from pandas import DataFrame
import numpy as np

import csv

def write_info(id, df, feature, hrv):
    df.to_csv("DataFeature/ECG_feature_"+id+".csv", index=True, mode="a", line_terminator="\n")
    feature.to_csv("DataFeature/ECG_feature_" + id + "_feature.csv", index=True, mode="a", line_terminator="\n")
    hrv.to_csv("DataFeature/ECG_feature_" + id + "_hrv.csv", index=True, mode="a", line_terminator="\n")

'''def window_info(id, max_signal1,  name_file):
    print("right file")
    participants = {'Participante': [id], 'Janela de maior informação sinal 1': [max_signal1]}

    df = DataFrame(participants, columns=['Participante', 'Janela de maior informação sinal 1'])
    df.columns = ['Participante', 'Janela de maior informação sinal 1']
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Info")
    #df.to_csv("/home/gisela.pinto/Tese/Info/"+name_file, index=False, header=None, mode='a')  # Don't forget to add '.xlsx' at the end of the path
    df.to_csv(f_path+"/"+name_file, index=False, header=None, mode='a')


def window_info(signal, name_file):
    with open(name_file, 'a') as f:
        for key, value in signal.items():
            f.write("%s, %s\n" % (key, value))
            f.close()
'''

def window_info(key, signal, name_file):
    df = pd.DataFrame([signal], index=[key])  # create a dataframe from match list
    df.to_csv(name_file, sep=',', index=key, header=None, mode='a')  # create csv from df


def r_peaks_info(id, r_peak):
    participants = {'R peak': [r_peak]}
    df = DataFrame(participants, columns=['R peak'])
    df.columns = ['R peak']
    #desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    df.to_csv("Rpeaks_"+id+".csv", index=False, header=None, mode='a')  # Don't forget to add '.xlsx' at the end of the path




def feature_save(key, type, df, new_df, new_feature):
    #file = open("/home/giselapinto/Tese/"+type+".txt", 'a')
    file = open(type+".txt", 'a')

    if new_feature is not None:
        file.write("Participant: {}\n".format(key))
        file.write("Processed features: {}\n".format(df))
        file.write("new_df: {}\n".format(new_df))
        file.write("new_feature: {}\n".format(new_feature))
        file.write("\n\n\n")
    else:
        file.write("Participant: {}\n".format(key))
        file.write("df: {}\n".format(df))
        file.write("new_df: {}\n".format(new_df))
        file.write("\n\n\n")
    file.close()


def select_feature(type, key, method, feature, feature_selected, thresh):
    type = type.split("/")[1]
    type = type.split(".")[0]
    #file2write = open("/home/giselapinto/Tese/"+method+"_"+type+".txt", 'a')
    file2write = open(method+"_"+type+".txt", 'a')

    file2write.write(str(["Participant", "All features", "Selected features", "Threshold used"]))
    file2write.write(str([key, method, feature, feature_selected, thresh]))
    file2write.close()


def write_performance_single_spliter(accuracy, precision, recall, f1_score, mean_absolut_error,sensitivity_N_val, sensitivity_F_val, sensitivity_H_val, specificity_N_val, specificity_F_val,specificity_H_val, confusion, classification,
                                    accuracy_t, precision_t, recall_t, f1_score_t, mean_absolut_error_t, sensitivity_N_test, sensitivity_F_test, sensitivity_H_test, specificity_N_test, specificity_F_test, specificity_H_test, confusion_t, classification_t, counter, i, path):
    file2write = open(path + "classification_report_every_ShuffleSplit_"+counter+".txt", 'a+')
    file2write.write("-----------------------------------\t\t\t%s\t\t\t----------------------------------\n" % i)
    file2write.write("Classification of the validation dataset\n")
    file2write.write("Prevision Accuracy: {0:.2%}\n".format(accuracy))
    file2write.write("Prevision precisio: {0:.2%}\n".format(precision))
    file2write.write("Prevision recall: {0:.2%}\n".format(recall))
    file2write.write("Prevision f1 score: {0:.2%}\n".format(f1_score))
    file2write.write("Mean Absolut error: {0:.2%}\n".format(mean_absolut_error))
    file2write.write("sensitivity Neutral: {0:.2%}\n".format(sensitivity_N_val))
    file2write.write("sensitivity Fear: {0:.2%}\n".format(sensitivity_F_val))
    file2write.write("sensitivity Happy: {0:.2%}\n".format(sensitivity_H_val))
    file2write.write("specificty Neutral: {0:.2%}\n".format(specificity_N_val))
    file2write.write("specificty Fear: {0:.2%}\n".format(specificity_F_val))
    file2write.write("specificty Happy: {0:.2%}\n".format(specificity_H_val))
    file2write.write("Confusion Matrix: %s\n" % confusion)
    file2write.write("Classification report: %s\n" %classification)
    file2write.write("\n\n\n")
    file2write.write("Classification of the Test dataset\n")
    file2write.write("Prevision Accuracy: {0:.2%}\n".format(accuracy_t))
    file2write.write("Prevision precisio: {0:.2%}\n".format(precision_t))
    file2write.write("Prevision recall: {0:.2%}\n".format(recall_t))
    file2write.write("Prevision f1 score: {0:.2%}\n".format(f1_score_t))
    file2write.write("Mean Absolut error: {0:.2%}\n".format(mean_absolut_error_t))
    file2write.write("sensitivity Neutral: {0:.2%}\n".format(sensitivity_N_test))
    file2write.write("sensitivity Fear: {0:.2%}\n".format(sensitivity_F_test))
    file2write.write("sensitivity Happy: {0:.2%}\n".format(sensitivity_H_test))
    file2write.write("specificty Neutral: {0:.2%}\n".format(specificity_N_test))
    file2write.write("specificty Fear: {0:.2%}\n".format(specificity_F_test))
    file2write.write("specificty Happy: {0:.2%}\n".format(specificity_H_test))
    file2write.write("Confusion Matrix: %s\n" % confusion_t)
    file2write.write("Classification report: %s\n" % classification_t)
    file2write.write("---------------------------------------------------------------------------------------\n")
    file2write.write("\n\n\n")
    file2write.close()

def write_performance_global_spliter(acc_app_val, precision_app_val, recall_app_val, f1_score_app_val, mean_absolut_val,sensitivity_app_N_val, sensitivity_app_F_val, sensitivity_app_H_val, specificity_app_N_val, specificity_app_F_val,specificity_app_H_val, confusion_val, classification_val,
                                    acc_app_test, precision_app_test, recall_app_test, f1_score_app_test, mean_absolut_test, sensitivity_app_N_test, sensitivity_app_F_test, sensitivity_app_H_test, specificity_app_N_test, specificity_app_F_test, specificity_app_H_test, confusion_test, classification_test, counter, path):
    ###### write performance of global shuffle split#######
    file2write = open(path + "classification_report_global_ShuffleSplit.txt", 'a+')
    file2write.write("-----------------------------------\t\t\t%s\t\t\t----------------------------------\n"%counter)
    file2write.write("Classification validation dataset\n")
    file2write.write("Accuracy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(acc_app_val) , np.std(acc_app_val) , np.amax(acc_app_val) , np.amin(acc_app_val)) )
    file2write.write("Precision: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(precision_app_val) , np.std(precision_app_val) , np.amax(precision_app_val) , np.amin(precision_app_val)))
    file2write.write("Recall: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(recall_app_val) , np.std(recall_app_val) , np.amax(recall_app_val) , np.amin(recall_app_val)) )
    file2write.write("f1_score: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(f1_score_app_val) , np.std(f1_score_app_val) , np.amax(f1_score_app_val) , np.amin(f1_score_app_val)) )
    file2write.write("mean_absolut_error: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(mean_absolut_val) , np.std(mean_absolut_val) , np.amax(mean_absolut_val) , np.amin(mean_absolut_val)) )
    file2write.write("sensitivty Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_N_val) , np.std(sensitivity_app_N_val) , np.amax(sensitivity_app_N_val) , np.amin(sensitivity_app_N_val)) )
    file2write.write("sensitivty Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_F_val) , np.std(sensitivity_app_F_val) , np.amax(sensitivity_app_F_val) , np.amin(sensitivity_app_F_val)) )
    file2write.write("sensitivty Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_H_val) , np.std(sensitivity_app_H_val) , np.amax(sensitivity_app_H_val) , np.amin(sensitivity_app_H_val)) )
    file2write.write("specificity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_N_val) , np.std(specificity_app_N_val) , np.amax(specificity_app_N_val) , np.amin(specificity_app_N_val) ))
    file2write.write("specificity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_F_val) , np.std(specificity_app_F_val) , np.amax(specificity_app_F_val) , np.amin(specificity_app_F_val)) )
    file2write.write("specificity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_H_val) , np.std(specificity_app_H_val) , np.amax(specificity_app_H_val) , np.amin(specificity_app_H_val)))
    file2write.write("Confusion Matrix: %s \n" % confusion_val)
    file2write.write("Classification report: %s\n"% classification_val)
    file2write.write("\n\n\n")
    file2write.write("Classification test dataset\n")
    file2write.write("Accuracy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(acc_app_test) , np.std(acc_app_test) , np.amax(acc_app_test) , np.amin(acc_app_test)) )
    file2write.write("Precision: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(precision_app_test) , np.std(precision_app_test) , np.amax(precision_app_test) , np.amin(precision_app_test)) )
    file2write.write("Recall: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(recall_app_test) , np.std(recall_app_test) , np.amax(recall_app_test) , np.amin(recall_app_test)) )
    file2write.write("f1_score: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(f1_score_app_test) , np.std(f1_score_app_test) ,np.amax(f1_score_app_test) , np.amin(f1_score_app_test)) )
    file2write.write("mean_absolut_error: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(mean_absolut_test) , np.std(mean_absolut_test) , np.amax(mean_absolut_test) , np.amin(mean_absolut_test)) )
    file2write.write("sensitivty Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_N_test) , np.std(sensitivity_app_N_test) , np.amax(sensitivity_app_N_test) , np.amin(sensitivity_app_N_test)) )
    file2write.write("sensitivty Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_F_test) , np.std(sensitivity_app_F_test) , np.amax(sensitivity_app_F_test), np.amin(sensitivity_app_F_test)) )
    file2write.write("sensitivty Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_H_test) , np.std(sensitivity_app_H_test) , np.amax(sensitivity_app_H_test) , np.amin(sensitivity_app_H_test)) )
    file2write.write("specificity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_N_test) , np.std(specificity_app_N_test) , np.amax(specificity_app_N_test) , np.amin(specificity_app_N_test)) )
    file2write.write("specificity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_F_test) , np.std(specificity_app_F_test) , np.amax(specificity_app_F_test) , np.amin(specificity_app_F_test)) )
    file2write.write("specificity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_H_test) , np.std(specificity_app_H_test) , np.amax(specificity_app_H_test) , np.amin(specificity_app_H_test)) )
    file2write.write("Confusion Matrix: %s \n" % confusion_test)
    file2write.write("Classification report: %s\n" % classification_test)
    file2write.write("---------------------------------------------------------------------------------------\n")
    file2write.write("\n\n\n")
    file2write.close()

def write_performance_every_iteration(accuracy, precision, recall, f1_score, mean_absolut_error, confusion, classification, counter, path):
    file2write = open(path + "classification_report_every_iteration.txt", 'a+')
    file2write.write("-----------------------------------\t\t\t%s\t\t\t----------------------------------\n" % counter)
    file2write.write("Prevision Accuracy: %s\n" % accuracy)
    file2write.write("Prevision precisio: %s\n" % precision)
    file2write.write("Prevision recall: %s\n" % recall)
    file2write.write("Prevision f1 score: %s\n" % f1_score)
    file2write.write("Mean Absolut error: %s\n" % mean_absolut_error)
    file2write.write("Confusion Matrix: %s\n" % confusion)
    file2write.write(classification)
    file2write.write("---------------------------------------------------------------------------------------\n")
    file2write.write("\n\n\n")
    file2write.close()
