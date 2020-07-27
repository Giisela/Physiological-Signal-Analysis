from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from WriteFiles import *
# Importing the statistics module 
import statistics 
from Global_var import *


def clean_lists():
    del acc_app_val[:]
    del precision_app_val[:]
    del recall_app_val[:]
    del f1_score_app_val[:]
    del mean_absolut_val[:]
    del specificity_app_N_val[:]
    del specificity_app_F_val[:]
    del specificity_app_H_val[:]
    del sensitivity_app_N_val[:]
    del sensitivity_app_F_val[:]
    del sensitivity_app_H_val[:]

    del  acc_app_test[:]
    del  precision_app_test[:]
    del  recall_app_test[:]
    del  f1_score_app_test[:]
    del  mean_absolut_test[:]
    del specificity_app_N_test[:]
    del specificity_app_F_test[:]
    del specificity_app_H_test[:]
    del sensitivity_app_N_test[:]
    del sensitivity_app_F_test[:]
    del sensitivity_app_H_test[:]

    del  acc_app_val_nn[:]
    del  precision_app_val_nn[:]
    del  recall_app_val_nn[:]
    del  f1_score_app_val_nn[:]
    del  mean_absolut_val_nn[:]
    del specificity_app_N_val_nn[:]
    del specificity_app_F_val_nn[:]
    del specificity_app_H_val_nn[:]
    del sensitivity_app_N_val_nn[:]
    del sensitivity_app_F_val_nn[:]
    del sensitivity_app_H_val_nn[:]

    del  acc_app_test_nn[:]
    del  precision_app_test_nn[:]
    del  recall_app_test_nn[:]
    del  f1_score_app_test_nn[:]
    del  mean_absolut_test_nn[:]
    del specificity_app_N_test_nn[:]
    del specificity_app_F_test_nn[:]
    del specificity_app_H_test_nn[:]
    del sensitivity_app_N_test_nn[:]
    del sensitivity_app_F_test_nn[:]
    del sensitivity_app_H_test_nn[:]

    del  acc_app_global[:]
    del  precision_app_global[:]
    del  recall_app_global[:]
    del  f1_score_app_global[:]
    del  mean_absolut_app_global[:]
    del specificity_app_N_global[:]
    del specificity_app_F_global[:]
    del specificity_app_H_global[:]
    del sensitivity_app_N_global[:]
    del sensitivity_app_F_global[:]
    del sensitivity_app_H_global[:]

    del acc_app_global_nn[:]
    del precision_app_global_nn[:]
    del recall_app_global_nn[:]
    del f1_score_app_global_nn[:]
    del mean_absolut_app_global_nn[:]
    del specificity_app_N_global_nn[:]
    del specificity_app_F_global_nn[:]
    del specificity_app_H_global_nn[:]
    del sensitivity_app_N_global_nn[:]
    del sensitivity_app_F_global_nn[:]
    del sensitivity_app_H_global_nn[:]


def performance_global_finaly(y_pred_test, y_test, model, path, title):
    if model=='rf':
        classification_test = metrics.classification_report(y_test, y_pred_test)
        confusion_test = metrics.confusion_matrix(y_test, y_pred_test)

        print("accuracy global list:", acc_app_global)
        print("accuracy global mean:", np.mean(acc_app_global))
    
        print("Accuracy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(acc_app_global) , np.std(acc_app_global) , np.amax(acc_app_global) , np.amin(acc_app_global) ))
        print("Precision: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(precision_app_global) , np.std(precision_app_global) , np.amax(precision_app_global) , np.amin(precision_app_global) ))
        print("Recall: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}] \n".format(np.mean(recall_app_global) , np.std(recall_app_global) , np.amax(recall_app_global) , np.amin(recall_app_global)) )
        print("f1_score: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(f1_score_app_global) , np.std(f1_score_app_global) , np.amax(f1_score_app_global) , np.amin(f1_score_app_global)) )
        print("mean_absolut_error: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(mean_absolut_app_global) , np.std(mean_absolut_app_global) , np.amax(mean_absolut_app_global) , np.amin(mean_absolut_app_global)) )
        print("sensitivity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_N_global) , np.std(sensitivity_app_N_global) , np.amax(sensitivity_app_N_global) , np.amin(sensitivity_app_N_global)) )
        print("sensitivity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_F_global) , np.std(sensitivity_app_F_global) , np.amax(sensitivity_app_F_global) , np.amin(sensitivity_app_F_global)) )
        print("sensitivity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_H_global) , np.std(sensitivity_app_H_global) , np.amax(sensitivity_app_H_global) , np.amin(sensitivity_app_H_global)) )
        print("specificity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_N_global) , np.std(specificity_app_N_global) , np.amax(specificity_app_N_global) , np.amin(specificity_app_N_global) ))
        print("specificity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_F_global) , np.std(specificity_app_F_global) , np.amax(specificity_app_F_global) , np.amin(specificity_app_F_global)) )
        print("specificity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_H_global) , np.std(specificity_app_H_global) , np.amax(specificity_app_H_global) , np.amin(specificity_app_H_global)))
        print("Confusion:  %s \n"%confusion_test)



        ###### write performance of global shuffle split#######
        file2write = open(path + "classification_report_global.txt", 'a+')
        file2write.write("-----------------------------------\t\t\t\t\t\t----------------------------------\n")
        file2write.write("Classification validation dataset")
        file2write.write("Accuracy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(acc_app_global) , np.std(acc_app_global) , np.amax(acc_app_global) , np.amin(acc_app_global)) )
        file2write.write("Precision: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(precision_app_global) , np.std(precision_app_global) , np.amax(precision_app_global) , np.amin(precision_app_global)) )
        file2write.write("Recall: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}] \n".format(np.mean(recall_app_global) , np.std(recall_app_global) , np.amax(recall_app_global) , np.amin(recall_app_global)) )
        file2write.write("f1_score: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(f1_score_app_global) , np.std(f1_score_app_global) , np.amax(f1_score_app_global) , np.amin(f1_score_app_global)) )
        file2write.write("mean_absolut_error: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(mean_absolut_app_global) , np.std(mean_absolut_app_global) , np.amax(mean_absolut_app_global), np.amin(mean_absolut_app_global)) )
        file2write.write("sensitivity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_N_global) , np.std(sensitivity_app_N_global) , np.amax(sensitivity_app_N_global) , np.amin(sensitivity_app_N_global)) )
        file2write.write("sensitivity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_F_global) , np.std(sensitivity_app_F_global) , np.amax(sensitivity_app_F_global) , np.amin(sensitivity_app_F_global)) )
        file2write.write("sensitivity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_H_global) , np.std(sensitivity_app_H_global) , np.amax(sensitivity_app_H_global) , np.amin(sensitivity_app_H_global)))
        file2write.write("specificity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_N_global) , np.std(specificity_app_N_global) , np.amax(specificity_app_N_global) , np.amin(specificity_app_N_global) ))
        file2write.write("specificity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_F_global) , np.std(specificity_app_F_global) , np.amax(specificity_app_F_global) , np.amin(specificity_app_F_global)) )
        file2write.write("specificity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_H_global) , np.std(specificity_app_H_global) , np.amax(specificity_app_H_global) , np.amin(specificity_app_H_global)))
        file2write.write("Confusion Matrix: %s \n" % confusion_test)
        file2write.write("Classification report: %s\n"% classification_test)
        file2write.write("\n\n\n")
    
    else:

        classification = metrics.classification_report(y_test, y_pred_test)
        confusion_test = metrics.confusion_matrix(y_test, y_pred_test)
        

        print("Accuracy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(acc_app_global_nn) , np.std(acc_app_global_nn) , np.amax(acc_app_global_nn) , np.amin(acc_app_global_nn) ))
        print("Precision: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}] \n".format(np.mean(precision_app_global_nn) , np.std(precision_app_global_nn) , np.amax(precision_app_global_nn) , np.argmin(precision_app_global_nn)))
        print("Recall: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(recall_app_global_nn) , np.std(recall_app_global_nn) , np.amax(recall_app_global_nn) , np.amin(recall_app_global_nn)) )
        print("f1_score: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(f1_score_app_global_nn) , np.std(f1_score_app_global_nn) , np.amax(f1_score_app_global_nn) , np.amin(f1_score_app_global_nn)) )
        print("mean_absolut_error: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(mean_absolut_app_global_nn) , np.std(mean_absolut_app_global_nn) , np.amax(mean_absolut_app_global_nn) , np.amin(mean_absolut_app_global_nn)))
        print("sensitivity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_N_global_nn) , np.std(sensitivity_app_N_global_nn) , np.amax(sensitivity_app_N_global_nn) , np.amin(sensitivity_app_N_global_nn)))
        print("sensitivity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_F_global_nn) , np.std(sensitivity_app_F_global_nn) , np.amax(sensitivity_app_F_global_nn) , np.amin(sensitivity_app_F_global_nn)))
        print("sensitivity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_H_global_nn) , np.std(sensitivity_app_H_global_nn) , np.amax(sensitivity_app_H_global_nn) , np.amin(sensitivity_app_H_global_nn)))
        print("specificity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_N_global_nn) , np.std(specificity_app_N_global_nn) , np.amax(specificity_app_N_global_nn) , np.amin(specificity_app_N_global_nn)))
        print("specificity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_F_global_nn) , np.std(specificity_app_F_global_nn) , np.amax(specificity_app_F_global_nn) , np.amin(specificity_app_F_global_nn)))
        print("specificity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_H_global_nn) , np.std(specificity_app_H_global_nn) , np.amax(specificity_app_H_global_nn), np.amin(specificity_app_H_global_nn)))
        print("Confusion:  %s\n"%confusion_test)



        ###### write performance of global shuffle split#######
        file2write = open(path + "classification_report_global.txt", 'a+')
        file2write.write("-----------------------------------\t\t\t\t\t\t----------------------------------\n")
        file2write.write("Classification validation dataset")
        file2write.write("Accuracy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(acc_app_global_nn) , np.std(acc_app_global_nn) , np.amax(acc_app_global_nn) , np.amin(acc_app_global_nn) ))
        file2write.write("Precision: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}] \n".format(np.mean(precision_app_global_nn) , np.std(precision_app_global_nn) , np.amax(precision_app_global_nn) , np.argmin(precision_app_global_nn)))
        file2write.write("Recall: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(recall_app_global_nn) , np.std(recall_app_global_nn) , np.amax(recall_app_global_nn) , np.amin(recall_app_global_nn)) )
        file2write.write("f1_score: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(f1_score_app_global_nn) , np.std(f1_score_app_global_nn) , np.amax(f1_score_app_global_nn) , np.amin(f1_score_app_global_nn)) )
        file2write.write("mean_absolut_error: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(mean_absolut_app_global_nn) , np.std(mean_absolut_app_global_nn) , np.amax(mean_absolut_app_global_nn) , np.amin(mean_absolut_app_global_nn)))
        file2write.write("sensitivity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_N_global_nn) , np.std(sensitivity_app_N_global_nn) , np.amax(sensitivity_app_N_global_nn) , np.amin(sensitivity_app_N_global_nn)))
        file2write.write("sensitivity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_F_global_nn) , np.std(sensitivity_app_F_global_nn) , np.amax(sensitivity_app_F_global_nn) , np.amin(sensitivity_app_F_global_nn)))
        file2write.write("sensitivity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_H_global_nn) , np.std(sensitivity_app_H_global_nn) , np.amax(sensitivity_app_H_global_nn) , np.amin(sensitivity_app_H_global_nn)))
        file2write.write("specificity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_N_global_nn) , np.std(specificity_app_N_global_nn) , np.amax(specificity_app_N_global_nn) , np.amin(specificity_app_N_global_nn)))
        file2write.write("specificity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_F_global_nn) , np.std(specificity_app_F_global_nn) , np.amax(specificity_app_F_global_nn) , np.amin(specificity_app_F_global_nn)))
        file2write.write("specificity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_H_global_nn) , np.std(specificity_app_H_global_nn) , np.amax(specificity_app_H_global_nn), np.amin(specificity_app_H_global_nn)))
        file2write.write("Confusion Matrix: %s \n" % confusion_test)
        file2write.write("Classification report: %s\n"% classification)
        file2write.write("\n\n\n")




def performance_global_everyIteration_shuffle(y_pred_val,y_pred, y_val, y_test, model,  path, title, counter, i):
    if model=='rf':    
        accuracy_val = metrics.accuracy_score(y_val, y_pred_val)
        acc_app_val.append(accuracy_val)

        precision_val = metrics.precision_score(y_val, y_pred_val, average='micro')
        precision_app_val.append(precision_val)

        recall_val = metrics.recall_score(y_val, y_pred_val, average='micro')
        recall_app_val.append(recall_val)

        f1_score_val = metrics.f1_score(y_val, y_pred_val, average='micro')
        f1_score_app_val.append(f1_score_val)

        mean_absolut_error_val = metrics.mean_absolute_error(y_val, y_pred_val)
        mean_absolut_val.append(mean_absolut_error_val)

        accuracy_test = metrics.accuracy_score(y_test, y_pred)
        acc_app_test.append(accuracy_test)

        precision_test = metrics.precision_score(y_test, y_pred, average='micro')
        precision_app_test.append(precision_test)

        recall_test = metrics.recall_score(y_test, y_pred, average='micro')
        recall_app_test.append(recall_test)

        f1_score_test = metrics.f1_score(y_test, y_pred, average='micro')
        f1_score_app_test.append(f1_score_test)

        mean_absolut_error_test = metrics.mean_absolute_error(y_test, y_pred)
        mean_absolut_test.append(mean_absolut_error_test)

        confusion_val = metrics.confusion_matrix(y_val, y_pred_val)
        
        value_v = metrics.confusion_matrix(y_val, y_pred_val, labels=[0,1,2])

        sensitivity_N_val = (value_v[0][0]/(value_v[0][0]+ value_v[0][1]+value_v[0][2]))
        sensitivity_app_N_val.append(sensitivity_N_val)

        sensitivity_F_val = (value_v[1][0]/(value_v[1][0]+ value_v[1][1]+value_v[1][2]))
        sensitivity_app_F_val.append(sensitivity_F_val)

        sensitivity_H_val = (value_v[2][0]/(value_v[2][0]+ value_v[2][1]+value_v[2][2]))
        sensitivity_app_H_val.append(sensitivity_H_val)


        specificity_N_val = ((value_v[1][1]+value_v[1][2]+value_v[2][1]+value_v[2][2])/(value_v[1][1]+value_v[1][2]+value_v[2][1]+value_v[2][2]+value_v[1][0]+value_v[2][0]))
        specificity_app_N_val.append(specificity_N_val)

        specificity_F_val = ((value_v[0][0]+value_v[0][2]+value_v[2][0]+value_v[2][2])/(value_v[0][0]+value_v[0][2]+value_v[2][0]+value_v[2][2]+value_v[0][1]+value_v[2][1]))
        specificity_app_F_val.append(specificity_F_val)

        specificity_H_val = ((value_v[0][0]+value_v[0][2]+value_v[1][0]+value_v[1][1])/(value_v[0][0]+value_v[0][2]+value_v[1][0]+value_v[1][1]+value_v[0][2]+value_v[1][2]))
        specificity_app_H_val.append(specificity_H_val)


        classification_val = metrics.classification_report(y_val, y_pred_val)

        confusion_test = metrics.confusion_matrix(y_test, y_pred)
        value = metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2])


        sensitivity_N_test = (value[0][0]/(value[0][0]+ value[0][1]+value[0][2]))
        sensitivity_app_N_test.append(sensitivity_N_test)

        sensitivity_F_test = (value[1][0]/(value[1][0]+ value[1][1]+value[1][2]))
        sensitivity_app_F_test.append(sensitivity_F_test)

        sensitivity_H_test = (value[2][0]/(value[2][0]+ value[2][1]+value[2][2]))
        sensitivity_app_H_test.append(sensitivity_H_test)


        specificity_N_test = ((value[1][1]+value[1][2]+value[2][1]+value[2][2])/(value[1][1]+value[1][2]+value[2][1]+value[2][2]+value[1][0]+value[2][0]))
        specificity_app_N_test.append(specificity_N_test)

        specificity_F_test = ((value[0][0]+value[0][2]+value[2][0]+value[2][2])/(value[0][0]+value[0][2]+value[2][0]+value[2][2]+value[0][1]+value[2][1]))
        specificity_app_F_test.append(specificity_F_test)

        specificity_H_test = ((value[0][0]+value[0][2]+value[1][0]+value[1][1])/(value[0][0]+value[0][2]+value[1][0]+value[1][1]+value[0][2]+value[1][2]))
        specificity_app_H_test.append(specificity_H_test)


        classification_test = metrics.classification_report(y_test, y_pred)

        print("Accuracy_val: {:.2%}".format(accuracy_val))
        print("Precision_val: {:.2%}".format(precision_val))
        print("Recall_val: {:.2%}".format(recall_val))
        print("f1_score_val: {:.2%}".format(f1_score_val))
        print("mean_absolut_error_val: {:.2%}".format(mean_absolut_error_val))
        print("sensitivity Neutral: {:.2%}".format(sensitivity_N_val))
        print("sensitivity Fear: {:.2%}".format(sensitivity_F_val))
        print("sensitivity Happy: {:.2%}".format(sensitivity_H_val))
        print("specificty Neutral: {:.2%}".format(specificity_N_val))
        print("specificty Fear: {:.2%}".format(specificity_F_val))
        print("specificty Happy: {:.2%}".format(specificity_H_val))
        print("Confusion_val: ", confusion_val)

        print("Accuracy_test: {:.2%}".format(accuracy_test))
        print("Precision_test: {:.2%}".format(precision_test))
        print("Recall_test: {:.2%}".format(recall_test) )
        print("f1_score_test: {:.2%}".format(f1_score_test) )
        print("mean_absolut_error_test: {:.2%}".format(mean_absolut_error_test) )
        print("sensitivity Neutral: {:.2%}".format(sensitivity_N_test) )
        print("sensitivity Fear: {:.2%}".format(sensitivity_F_test) )
        print("sensitivity Happy:{:.2%} ".format(sensitivity_H_test) )
        print("specificty Neutral: {:.2%}".format(specificity_N_test) )
        print("specificty Fear: {:.2%}".format(specificity_F_test) )
        print("specificty Happy:{:.2%} ".format(specificity_H_test) )
        print("Confusion_test: ", confusion_test)

        ###### write performance of single line#######
        write_performance_single_spliter(accuracy_val, precision_val, recall_val, f1_score_val, mean_absolut_error_val, sensitivity_N_val, sensitivity_F_val, sensitivity_H_val, specificity_N_val, specificity_F_val,specificity_H_val, confusion_val, classification_val,
                                        accuracy_test, precision_test, recall_test, f1_score_test, mean_absolut_error_test, sensitivity_N_test, sensitivity_F_test, sensitivity_H_test, specificity_N_test, specificity_F_test, specificity_H_test, confusion_test, classification_test, counter,i, path)

        
    else:

        accuracy_val_nn = metrics.accuracy_score(y_val, y_pred_val)
        acc_app_val_nn.append(accuracy_val_nn)

        precision_val_nn = metrics.precision_score(y_val, y_pred_val, average='micro')
        precision_app_val_nn.append(precision_val_nn)

        recall_val_nn = metrics.recall_score(y_val, y_pred_val, average='micro')
        recall_app_val_nn.append(recall_val_nn)

        f1_score_val_nn = metrics.f1_score(y_val, y_pred_val, average='micro')
        f1_score_app_val_nn.append(f1_score_val_nn)

        mean_absolut_error_val_nn = metrics.mean_absolute_error(y_val, y_pred_val)
        mean_absolut_val_nn.append(mean_absolut_error_val_nn)

        accuracy_test_nn = metrics.accuracy_score(y_test, y_pred)
        acc_app_test_nn.append(accuracy_test_nn)

        precision_test_nn = metrics.precision_score(y_test, y_pred, average='micro')
        precision_app_test_nn.append(precision_test_nn)

        recall_test_nn = metrics.recall_score(y_test, y_pred, average='micro')
        recall_app_test_nn.append(recall_test_nn)

        f1_score_test_nn = metrics.f1_score(y_test, y_pred, average='micro')
        f1_score_app_test_nn.append(f1_score_test_nn)

        mean_absolut_error_test_nn = metrics.mean_absolute_error(y_test, y_pred)
        mean_absolut_test_nn.append(mean_absolut_error_test_nn)

        confusion_val_nn = metrics.confusion_matrix(y_val, y_pred_val)
        value_v = metrics.confusion_matrix(y_val, y_pred_val, labels=[0,1,2])

        sensitivity_N_val_nn = (value_v[0][0]/(value_v[0][0]+ value_v[0][1]+value_v[0][2]))
        sensitivity_app_N_val_nn.append(sensitivity_N_val_nn)

        sensitivity_F_val_nn = (value_v[1][0]/(value_v[1][0]+ value_v[1][1]+value_v[1][2]))
        sensitivity_app_F_val_nn.append(sensitivity_F_val_nn)

        sensitivity_H_val_nn = (value_v[2][0]/(value_v[2][0]+ value_v[2][1]+value_v[2][2]))
        sensitivity_app_H_val_nn.append(sensitivity_H_val_nn)


        specificity_N_val_nn = ((value_v[1][1]+value_v[1][2]+value_v[2][1]+value_v[2][2])/(value_v[1][1]+value_v[1][2]+value_v[2][1]+value_v[2][2]+value_v[1][0]+value_v[2][0]))
        specificity_app_N_val_nn.append(specificity_N_val_nn)

        specificity_F_val_nn = ((value_v[0][0]+value_v[0][2]+value_v[2][0]+value_v[2][2])/(value_v[0][0]+value_v[0][2]+value_v[2][0]+value_v[2][2]+value_v[0][1]+value_v[2][1]))
        specificity_app_F_val_nn.append(specificity_F_val_nn)

        specificity_H_val_nn = ((value_v[0][0]+value_v[0][2]+value_v[1][0]+value_v[1][1])/(value_v[0][0]+value_v[0][2]+value_v[1][0]+value_v[1][1]+value_v[0][2]+value_v[1][2]))
        specificity_app_H_val_nn.append(specificity_H_val_nn)


        classification_val_nn = metrics.classification_report(y_val, y_pred_val)
        

        confusion_test_nn = metrics.confusion_matrix(y_test, y_pred)
        value = metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2])


        sensitivity_N_test_nn = (value[0][0]/(value[0][0]+ value[0][1]+value[0][2]))
        sensitivity_app_N_test_nn.append(sensitivity_N_test_nn)

        sensitivity_F_test_nn = (value[1][0]/(value[1][0]+ value[1][1]+value[1][2]))
        sensitivity_app_F_test_nn.append(sensitivity_F_test_nn)

        sensitivity_H_test_nn = (value[2][0]/(value[2][0]+ value[2][1]+value[2][2]))
        sensitivity_app_H_test_nn.append(sensitivity_H_test_nn)


        specificity_N_test_nn = ((value[1][1]+value[1][2]+value[2][1]+value[2][2])/(value[1][1]+value[1][2]+value[2][1]+value[2][2]+value[1][0]+value[2][0]))
        specificity_app_N_test_nn.append(specificity_N_test_nn)

        specificity_F_test_nn = ((value[0][0]+value[0][2]+value[2][0]+value[2][2])/(value[0][0]+value[0][2]+value[2][0]+value[2][2]+value[0][1]+value[2][1]))
        specificity_app_F_test_nn.append(specificity_F_test_nn)

        specificity_H_test_nn = ((value[0][0]+value[0][2]+value[1][0]+value[1][1])/(value[0][0]+value[0][2]+value[1][0]+value[1][1]+value[0][2]+value[1][2]))
        specificity_app_H_test_nn.append(specificity_H_test_nn)


        classification_test_nn = metrics.classification_report(y_test, y_pred)

        print("Accuracy_val: {:.2%}".format(accuracy_val_nn ))
        print("Precision_val:{:.2%}".format(precision_val_nn ))
        print("Recall_val: {:.2%}".format(recall_val_nn ))
        print("f1_score_val: {:.2%}".format(f1_score_val_nn ))
        print("mean_absolut_error_val:{:.2%}".format( mean_absolut_error_val_nn ))
        print("sensitivity Neutral: {:.2%}".format(sensitivity_N_val_nn ))
        print("sensitivity Fear:{:.2%}".format(sensitivity_F_val_nn ))
        print("sensitivity Happy: {:.2%}".format(sensitivity_H_val_nn ))
        print("specificty Neutral:{:.2%}".format(specificity_N_val_nn ))
        print("specificty Fear: {:.2%}".format( specificity_F_val_nn ))
        print("specificty Happy:{:.2%}".format(specificity_H_val_nn ))
        print("Confusion_val:", confusion_val_nn)

        print("Accuracy_test:{:.2%}".format(accuracy_test_nn ))
        print("Precision_test: {:.2%}".format(precision_test_nn ))
        print("Recall_test: {:.2%}".format(recall_test_nn ))
        print("f1_score_test: {:.2%}".format(f1_score_test_nn ))
        print("mean_absolut_error_test:{:.2%}".format(mean_absolut_error_test_nn ))
        print("sensitivity Neutral:{:.2%}".format(sensitivity_N_test_nn ))
        print("sensitivity Fear:{:.2%}".format(sensitivity_F_test_nn ))
        print("sensitivity Happy: {:.2%}".format(sensitivity_H_test_nn ))
        print("specificty Neutral: {:.2%}".format(specificity_N_test_nn ))
        print("specificty Fear: {:.2%}".format(specificity_F_test_nn ))
        print("specificty Happy: {:.2%}".format(specificity_H_test_nn ))
        print("Confusion_test: ", confusion_test_nn)

        ###### write performance of single line#######
        write_performance_single_spliter(accuracy_val_nn, precision_val_nn, recall_val_nn, f1_score_val_nn, mean_absolut_error_val_nn, sensitivity_N_val_nn, sensitivity_F_val_nn, sensitivity_H_val_nn, specificity_N_val_nn, specificity_F_val_nn,specificity_H_val_nn, confusion_val_nn,classification_val_nn,
                                        accuracy_test_nn, precision_test_nn, recall_test_nn, f1_score_test_nn, mean_absolut_error_test_nn, sensitivity_N_test_nn, sensitivity_F_test_nn, sensitivity_H_test_nn, specificity_N_test_nn, specificity_F_test_nn, specificity_H_test_nn, confusion_test_nn, classification_test_nn, counter, i, path)


def performe_global_shuffle(y_pred, y_test, y_pred_val, y_val, model, path,title, counter):
    if model=='rf':
        
       
        acc_app_global.append(np.mean(acc_app_test))
        precision_app_global.append(np.mean(precision_app_test))
        recall_app_global.append(np.mean(recall_app_test))
        f1_score_app_global.append(np.mean(f1_score_app_test))
        mean_absolut_app_global.append(np.mean(mean_absolut_test))
        sensitivity_app_N_global.append(np.mean(sensitivity_app_N_test))
        sensitivity_app_F_global.append(np.mean(sensitivity_app_F_test))
        sensitivity_app_H_global.append(np.mean(sensitivity_app_H_test))

        specificity_app_N_global.append(np.mean(specificity_app_N_test))
        specificity_app_F_global.append(np.mean(specificity_app_F_test))
        specificity_app_H_global.append(np.mean(specificity_app_H_test))


        classification_val = metrics.classification_report(y_val, y_pred_val)
        classification_test = metrics.classification_report(y_test, y_pred)

        confusion_val = metrics.confusion_matrix(y_val, y_pred_val)

        confusion_test = metrics.confusion_matrix(y_test, y_pred)

        print("Accuracy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(acc_app_val) , np.std(acc_app_val) , np.amax(acc_app_val) , np.amin(acc_app_val)) )
        print("Precision: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(precision_app_val) , np.std(precision_app_val) , np.amax(precision_app_val) , np.amin(precision_app_val)))
        print("Recall: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(recall_app_val) , np.std(recall_app_val) , np.amax(recall_app_val) , np.amin(recall_app_val)) )
        print("f1_score: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(f1_score_app_val) , np.std(f1_score_app_val) , np.amax(f1_score_app_val) , np.amin(f1_score_app_val)) )
        print("mean_absolut_error: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(mean_absolut_val) , np.std(mean_absolut_val) , np.amax(mean_absolut_val) , np.amin(mean_absolut_val)) )
        print("sensitivty Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_N_val) , np.std(sensitivity_app_N_val) , np.amax(sensitivity_app_N_val) , np.amin(sensitivity_app_N_val)) )
        print("sensitivty Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_F_val) , np.std(sensitivity_app_F_val) , np.amax(sensitivity_app_F_val) , np.amin(sensitivity_app_F_val)) )
        print("sensitivty Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_H_val) , np.std(sensitivity_app_H_val) , np.amax(sensitivity_app_H_val) , np.amin(sensitivity_app_H_val)) )
        print("specificity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_N_val) , np.std(specificity_app_N_val) , np.amax(specificity_app_N_val) , np.amin(specificity_app_N_val) ))
        print("specificity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_F_val) , np.std(specificity_app_F_val) , np.amax(specificity_app_F_val) , np.amin(specificity_app_F_val)) )
        print("specificity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_H_val) , np.std(specificity_app_H_val) , np.amax(specificity_app_H_val) , np.amin(specificity_app_H_val)))
        print("Confusion: %s  \n"% confusion_val)

        print("Accuracy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(acc_app_test) , np.std(acc_app_test) , np.amax(acc_app_test) , np.amin(acc_app_test)) )
        print("Precision: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(precision_app_test) , np.std(precision_app_test) , np.amax(precision_app_test) , np.amin(precision_app_test)) )
        print("Recall: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(recall_app_test) , np.std(recall_app_test) , np.amax(recall_app_test) , np.amin(recall_app_test)) )
        print("f1_score: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(f1_score_app_test) , np.std(f1_score_app_test) ,np.amax(f1_score_app_test) , np.amin(f1_score_app_test)) )
        print("mean_absolut_error: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(mean_absolut_test) , np.std(mean_absolut_test) , np.amax(mean_absolut_test) , np.amin(mean_absolut_test)) )
        print("sensitivty Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_N_test) , np.std(sensitivity_app_N_test) , np.amax(sensitivity_app_N_test) , np.amin(sensitivity_app_N_test)) )
        print("sensitivty Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_F_test) , np.std(sensitivity_app_F_test) , np.amax(sensitivity_app_F_test), np.amin(sensitivity_app_F_test)) )
        print("sensitivty Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_H_test) , np.std(sensitivity_app_H_test) , np.amax(sensitivity_app_H_test) , np.amin(sensitivity_app_H_test)) )
        print("specificity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_N_test) , np.std(specificity_app_N_test) , np.amax(specificity_app_N_test) , np.amin(specificity_app_N_test)) )
        print("specificity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_F_test) , np.std(specificity_app_F_test) , np.amax(specificity_app_F_test) , np.amin(specificity_app_F_test)) )
        print("specificity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_H_test) , np.std(specificity_app_H_test) , np.amax(specificity_app_H_test) , np.amin(specificity_app_H_test)) )
        print("Confusion:  %s \n"%confusion_test)

        write_performance_global_spliter(acc_app_val, precision_app_val, recall_app_val, f1_score_app_val, mean_absolut_val,sensitivity_app_N_val, sensitivity_app_F_val, sensitivity_app_H_val, specificity_app_N_val, specificity_app_F_val,specificity_app_H_val, confusion_val, classification_val,
                                    acc_app_test, precision_app_test, recall_app_test, f1_score_app_test, mean_absolut_test, sensitivity_app_N_test, sensitivity_app_F_test, sensitivity_app_H_test, specificity_app_N_test, specificity_app_F_test, specificity_app_H_test, confusion_test, classification_test, counter, path)

    else:
        acc_app_global_nn.append(np.mean(acc_app_test_nn))

        precision_app_global_nn.append(np.mean(precision_app_test_nn))
        recall_app_global_nn.append(np.mean(recall_app_test_nn))
        f1_score_app_global_nn.append(np.mean(f1_score_app_test_nn))
        mean_absolut_app_global_nn.append(np.mean(mean_absolut_test_nn))

        sensitivity_app_N_global_nn.append(np.mean(sensitivity_app_N_test_nn))
        sensitivity_app_F_global_nn.append(np.mean(sensitivity_app_F_test_nn))
        sensitivity_app_H_global_nn.append(np.mean(sensitivity_app_H_test_nn))

        specificity_app_N_global_nn.append(np.mean(specificity_app_N_test_nn))
        specificity_app_F_global_nn.append(np.mean(specificity_app_F_test_nn))
        specificity_app_H_global_nn.append(np.mean(specificity_app_H_test_nn))


        classification_val_nn = metrics.classification_report(y_val, y_pred_val)
        classification_test_nn = metrics.classification_report(y_test, y_pred)

        confusion_val_nn = metrics.confusion_matrix(y_val, y_pred_val)

        confusion_test_nn = metrics.confusion_matrix(y_test, y_pred)
        

        print("Accuracy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(acc_app_val_nn) , np.std(acc_app_val_nn) , np.amax(acc_app_val_nn) , np.amin(acc_app_val_nn)) )
        print("Precision: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(precision_app_val_nn) , np.std(precision_app_val_nn) , np.amax(precision_app_val_nn) , np.amin(precision_app_val_nn)))
        print("Recall: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(recall_app_val_nn) , np.std(recall_app_val_nn) , np.amax(recall_app_val_nn) , np.amin(recall_app_val_nn) ))
        print("f1_score: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(f1_score_app_val_nn) , np.std(f1_score_app_val_nn) , np.amax(f1_score_app_val_nn) , np.amin(f1_score_app_val_nn) ))
        print("mean_absolut_error: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(mean_absolut_val_nn) , np.std(mean_absolut_val_nn) , np.amax(mean_absolut_val_nn) , np.amin(mean_absolut_val_nn)))
        print("sensitivty Neutral: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}] \n".format(np.mean(sensitivity_app_N_val_nn) , np.std(sensitivity_app_N_val_nn) , np.amax(sensitivity_app_N_val_nn), np.amin(sensitivity_app_N_val_nn)) )
        print("sensitivty Fear: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}] \n".format(np.mean(sensitivity_app_F_val_nn) , np.std(sensitivity_app_F_val_nn) , np.amax(sensitivity_app_F_val_nn), np.amin(sensitivity_app_F_val_nn)) )
        print("sensitivty Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_H_val_nn) , np.std(sensitivity_app_H_val_nn) , np.amax(sensitivity_app_H_val_nn) , np.amin(sensitivity_app_H_val_nn)) )
        print("specificity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_N_val_nn) , np.std(specificity_app_N_val_nn) , np.amax(specificity_app_N_val_nn) , np.amin(specificity_app_N_val_nn)) )
        print("specificity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_F_val_nn) , np.std(specificity_app_F_val_nn) , np.amax(specificity_app_F_val_nn) , np.amin(specificity_app_F_val_nn)) )
        print("specificity Happy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_H_val_nn) , np.std(specificity_app_H_val_nn) , np.amax(specificity_app_H_val_nn) , np.amin(specificity_app_H_val_nn)) )
        print("Confusion: %s \n"% confusion_val_nn)

        print("Accuracy: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(acc_app_test_nn) , np.std(acc_app_test_nn) , np.amax(acc_app_test_nn) , np.amin(acc_app_test_nn)) )
        print("Precision: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(precision_app_test_nn) , np.std(precision_app_test_nn) , np.amax(precision_app_test_nn) , np.amin(precision_app_test_nn)) )
        print("Recall: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(recall_app_test_nn) , np.std(recall_app_test_nn) , np.amax(recall_app_test_nn) , np.amin(recall_app_test_nn)) )
        print("f1_score: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(f1_score_app_test_nn) , np.std(f1_score_app_test_nn) , np.amax(f1_score_app_test_nn) , np.amin(f1_score_app_test_nn)) )
        print("mean_absolut_error: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(mean_absolut_test_nn) , np.std(mean_absolut_test_nn) , np.amax(mean_absolut_test_nn), np.amin(mean_absolut_test_nn)) )
        print("sensitivty Neutral: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}] \n".format(np.mean(sensitivity_app_N_test_nn) , np.std(sensitivity_app_N_test_nn) , np.amax(sensitivity_app_N_test_nn) , np.amin(sensitivity_app_N_test_nn)) )
        print("sensitivty Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(sensitivity_app_F_test_nn) , np.std(sensitivity_app_F_test_nn) , np.amax(sensitivity_app_F_test_nn) , np.amin(sensitivity_app_F_test_nn)) )
        print("sensitivty Happy: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}] \n".format(np.mean(sensitivity_app_H_test_nn) , np.std(sensitivity_app_H_test_nn) , np.amax(sensitivity_app_H_test_nn) , np.amin(sensitivity_app_H_test_nn)) )
        print("specificity Neutral: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_N_test_nn) , np.std(specificity_app_N_test_nn) , np.amax(specificity_app_N_test_nn) , np.amin(specificity_app_N_test_nn)) )
        print("specificity Fear: {:.2%} ({:.2%}) [Max:{:.2%}, Min:{:.2%}]\n".format(np.mean(specificity_app_F_test_nn) , np.std(specificity_app_F_test_nn) , np.amax(specificity_app_F_test_nn) , np.amin(specificity_app_F_test_nn)) )
        print("specificity Happy: {:.2%} ({:.2%})[Max:{:.2%}, Min:{:.2%}] \n".format(np.mean(specificity_app_H_test_nn) , np.std(specificity_app_H_test_nn) , np.amax(specificity_app_H_test_nn) , np.amin(specificity_app_H_test_nn)) )
        print("Confusion:  %s \n"%confusion_test_nn)

        write_performance_global_spliter(acc_app_val_nn, precision_app_val_nn, recall_app_val_nn, f1_score_app_val_nn, mean_absolut_val_nn,sensitivity_app_N_val_nn, sensitivity_app_F_val_nn, sensitivity_app_H_val_nn, specificity_app_N_val_nn, specificity_app_F_val_nn,specificity_app_H_val_nn, confusion_val_nn, classification_val_nn,
                                    acc_app_test_nn, precision_app_test_nn, recall_app_test_nn, f1_score_app_test_nn, mean_absolut_test_nn, sensitivity_app_N_test_nn, sensitivity_app_F_test_nn, sensitivity_app_H_test_nn, specificity_app_N_test_nn, specificity_app_F_test_nn, specificity_app_H_test_nn, confusion_test_nn, classification_test_nn, counter, path)



def performance_global_shuffle_allmix(y_pred, y_test, accuracy, precision, recall, f1_score, mean_absolut_error, path, title):
    confusion_global_spliter = metrics.confusion_matrix(y_pred, y_test)
    plot_confusion(confusion_global_spliter, target_names=["N", "F", "H"],
                   path=path + "Confusion_matrix_global_ShuffleSplit.png",
                   title=title, normalize=False)

    classification_global_spliter = metrics.classification_report(y_test, y_pred)

    print("Accuracy: %s (+/-) %s \n" % (statistics.mean(accuracy), statistics.stdev(accuracy)))
    print("Precision: %s (+/-) %s \n" % (statistics.mean(precision), statistics.stdev(precision)))
    print("Recall: %s (+/-) %s\n" % (statistics.mean(recall), statistics.stdev(recall)))
    print("f1_score: %s (+/-) %sv" % (statistics.mean(f1_score), statistics.stdev(f1_score)))
    print("mean_absolut_error: %s (+/-) %s\n" % (statistics.mean(mean_absolut_error), statistics.stdev(mean_absolut_error)))
    print("Confusion: \n", confusion_global_spliter)
    print("Classification report: \n", classification_global_spliter)

    ###### write performance of global shuffle split#######
    file2write = open(path + "classification_report_global_ShuffleSplit.txt", 'a+')
    file2write.write("-----------------------------------\t\t\t\t\t\t----------------------------------\n")
    file2write.write("Accuracy: %s (+/-) %s \n" % (statistics.mean(accuracy), statistics.stdev(accuracy)))
    file2write.write("Precision: %s (+/-) %s \n" % (statistics.mean(precision), statistics.stdev(precision)))
    file2write.write("Recall: %s (+/-) %s \n" % (statistics.mean(recall), statistics.stdev(recall)))
    file2write.write("Prevision f1 score: %s\n" % f1_score)
    file2write.write("f1_score: %s (+/-) %s \n" % (statistics.mean(f1_score), statistics.stdev(f1_score)))
    file2write.write("mean_absolut_error: %s (+/-) %s \n" % (statistics.mean(mean_absolut_error), statistics.stdev(mean_absolut_error)))
    file2write.write("Confusion Matrix: %s\n" % confusion_global_spliter)
    file2write.write(classification_global_spliter)
    file2write.write("---------------------------------------------------------------------------------------\n")
    file2write.write("\n\n\n")
    file2write.close()


    
def performance_every_shuffler_allmix(y_pred, y_test, accuracy, precision, recall, f1_score, mean_absolut_error, path, title):
    confusion = metrics.confusion_matrix(y_pred, y_test)
    plot_confusion(confusion, target_names=["N", "F", "H"],
                   path=path + "Confusion_matrix_every_ShuffleSplit.png",
                   title=title, normalize=False)

    classification = metrics.classification_report(y_test, y_pred)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1_score: ", f1_score)
    print("mean_absolut_error: ", mean_absolut_error)
    print("Confusion: ", confusion)

    ###### write performance of single line#######
    file2write = open(path + "classification_report_every_ShuffleSplit.txt", 'a+')
    file2write.write("---------------------------------------------------------------------\n")
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

def plot_confusion(cm,
                  target_names,
                  path,
                  title,
                  cmap=None,
                  normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
    plt.savefig(path)

