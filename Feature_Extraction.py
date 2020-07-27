############# library ##########
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import tools
from scipy import signal
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import glob
from itertools import zip_longest
import neurokit as nk
import matplotlib.pyplot as plt
from math import isnan
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score


############# files ###########
from Ploting_data import *
from Global_var import *
from WriteFiles import *


def classification(df, y):
    feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=len(df.keys()),
           forward=True,
           verbose=2,
           scoring='roc_auc',
           cv=4)

    features = feature_selector.fit(np.array(df), y)
    filtered_features= X.columns[list(features.k_feature_idx_)]
    return filtered_features
  
def variance_thresh(df, thresh):
    '''
    Removes all features with variance below thresh

    Parameters : df  : (DataFrame) Dataset
                 thresh   : (float) the number used to threshold the variance

    output: selected feature

    '''
    try:
        variance_thresh = VarianceThreshold(threshold=thresh)
        feature_set = variance_thresh.fit(df)
        data_set = feature_set.transform(df)
        features = []

        for key in df.keys():
            for x in data_set.T:
                if np.allclose(np.asarray(x), np.asarray(df[key].T)):
                    features.append(key)

        return features
    except:
        return []


'''def find_correlation(df, thresh):
    corrMatrix = df.corr()  #Cria a matrix de correlção das colunas através do pearson  
    #Calcula a correlação linear entre X e Y
   
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1)
    already_in = set()
    result = []
    print(corrMatrix.head())
    for col in corrMatrix: #Passa por cada coluna na matrix e seleciona os valores com tresh acima do referidi
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
   
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    print(result)
    return result, select_flat'''


def find_correlation(df, threshold):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater the dinamic threshold
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

    return to_drop

def select_best(df, thresh):
    try:
        # feature extraction
        k_best = SelectKBest(score_func=f_classif, k=thresh)
        # fit on train set
        fit = k_best.fit(df)
        # transform train set
        data_set = fit.transform(df)

        features = []
        for key in df.keys():
            for x in data_set.T:
                if np.allclose(np.asarray(x), np.asarray(df[key].T)):
                    features.append(key)

        return features
    except:
        return []

def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(list(initial_features))>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        print("MIN: ", min_p_value)
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features

def backward_elimination(data, target,significance_level = 0.05):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        print("features_with_constant ",features_with_constant)
        print("target: ",target)
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        print("p values ", p_values)
        max_p_value = p_values.max()
        print("MAX:", max_p_value)
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features

def downslamplig(df, grouping, sample):
    df_down = []
    #print(key)
    for down in grouper(df, grouping, None):
        #print("hr: ", down)
        #print("mean: ", np.mean(down))
        df_down.append(np.median(down))

    
    #print("hr 5:", len(df_down))
    resampled_signal = signal.decimate(df_down, sample)
    result = pd.Series(resampled_signal)


    '''plt.title(key)
    plt.plot(df_down, color='blue', linewidth=1.5)
    plt.plot(resampled_signal, color='orange', linewidth=1.5)
    plt.subplots_adjust(hspace=0.50)

    plt.xlabel("Sample per second")
    plt.grid()
    plt.savefig("Plot_Participant/Downsampling/"+id+"_"+type+"_"+key+".png")
    plt.close()
    #plt.show()'''


    return result

def downslamplig_counter(df, grouping, sample):
    df_down = []
    #print(key)
    for down in grouper(df, grouping, None):
        df_down.append(sum(down))
    #print("hr 5:", len(df_down))
    #resampled_signal = signal.decimate(df_down, sample)
    result = pd.Series(df_down)


    '''plt.title(key)
    plt.plot(df_down, color='blue', linewidth=1.5)
    plt.plot(resampled_signal, color='orange', linewidth=1.5)
    plt.subplots_adjust(hspace=0.50)

    plt.xlabel("Sample per second")
    plt.grid()
    plt.savefig("Plot_Participant/Downsampling/"+id+"_"+type+"_"+key+".png")
    plt.close()
    #plt.show()'''


    return result

def featureExtraction_metrics(name):
    path = 'Data_inc/Per_Emotion/*' + name + '*'
    for fp in glob.glob(os.path.join(path)):
        loadData = pickle.load(open(fp, "rb"))
        print(fp)
        '''key = fp.split("/")[2]
        key = key.split("_")[1]
        key = key.split(".")[0]
        print(key)'''
        df = None
        
        i = 0
        if "ECG" in fp:
            df = pd.DataFrame()
            df_1 = pd.DataFrame()

            df_1_N = pd.DataFrame.from_dict(loadData['ECG', "N"])
            df_1_N = df_1_N.drop("Class", axis=1)

            for id in df_1_N.keys():
                new = downslamplig(df_1_N[id], 1000, 1)
                df_1[id] = new
            df_1["Class"] = "N"
            df = df.append([df_1], ignore_index=True)



            df_1_F = pd.DataFrame.from_dict(loadData['ECG', "F"])
            df_1_F = df_1_F.drop("Class", axis=1)

            for id in df_1_F.keys():
                new = downslamplig(df_1_F[id], 1000, 1)
                df_1[id] = new
            df_1["Class"] = "F"
            df = df.append([df_1], ignore_index=True)



            df_1_H = pd.DataFrame.from_dict(loadData['ECG', "H"])
            df_1_H = df_1_H.drop("Class", axis=1)

            for id in df_1_H.keys():
                new = downslamplig(df_1_H[id], 1000, 1)
                df_1[id] = new
            df_1["Class"] = "H"
            df = df.append([df_1], ignore_index=True)
            print(df)
           

            '''for id in df_1.keys():
                new = downslamplig(df_1[id], 3000, 1)
                df[id] = new'''
            
            '''for id in df_1.keys():
                new = downslamplig(df_1[id], 6000, 1)
                df[id] = new'''

            '''
            feature = pd.DataFrame({"T_Waves": df["T_Waves"], "Q_Waves": df["Q_Waves"], "P_Waves": df["P_Waves"]})

            del df["T_Waves"]
            del df["Q_Waves"]
            del df["P_Waves"]

            for id in df.keys():
                new = downslamplig(df[id], 6000, 1)
                df[id] = new

            feature = feature.dropna(how='all', axis=0)
            '''


            df_file = open("AllFeature/Selected_Feature/feature_ECG.txt", "a+")
            df_file.write("%s\n" % (df.keys()))
            df_file.close()

        
        elif "EDA" in fp:
            df = pd.DataFrame()
            df_1 = pd.DataFrame()

            df_1_N = pd.DataFrame.from_dict(loadData['EDA', "N"])
            df_1_N = df_1_N.drop("Class", axis=1)

            for id in df_1_N.keys():
                if "SCR_Onsets" in id:
                    new = downslamplig_counter(df_1_N[id], 1000, 1)
                    df_1[id] = new
                else:
                    new = downslamplig(df_1_N[id], 1000, 1)
                    df_1[id] = new
            df_1["Class"] = "N"
            df = df.append([df_1], ignore_index=True)



            df_1_F = pd.DataFrame.from_dict(loadData['EDA', "F"])
            df_1_F = df_1_F.drop("Class", axis=1)

            for id in df_1_F.keys():
                if "SCR_Onsets" in id:
                    new = downslamplig_counter(df_1_F[id], 1000, 1)
                    df_1[id] = new
                else:
                    new = downslamplig(df_1_N[id], 1000, 1)
                    df_1[id] = new
            df_1["Class"] = "F"
            df = df.append([df_1], ignore_index=True)



            df_1_H = pd.DataFrame.from_dict(loadData['EDA', "H"])
            df_1_H = df_1_H.drop("Class", axis=1)

            for id in df_1_H.keys():
                if "SCR_Onsets" in id:
                    new = downslamplig_counter(df_1_H[id], 1000, 1)
                    df_1[id] = new
                else:
                    new = downslamplig(df_1_N[id], 1000, 1)
                    df_1[id] = new
            df_1["Class"] = "H"
            df = df.append([df_1], ignore_index=True)
            del df["SCR_Peaks_Amplitudes"]
            print(df)

            '''del df_1["SCR_Peaks_Amplitudes"]

            for id in df_1.keys():
                if "SCR_Onsets" in id:
                    new = downslamplig_counter(df_1[id], 3000, 1)
                    df[id] = new
                else:
                    new = downslamplig(df_1[id], 3000,1)
                    df[id] = new'''


            '''feature = pd.DataFrame({"SCR_Peaks_Amplitudes": df["SCR_Peaks_Amplitudes"], "SCR_Onsets": df["SCR_Onsets"]})

            del df["SCR_Peaks_Amplitudes"]
            del df["SCR_Onsets"]

            for id in df.keys():
                new = downslamplig(df[id], 6000, 1)
                df[id] = new

            feature = feature.dropna(how='all', axis=0)
            df = df.dropna(how='all', axis=0)'''


            df_file = open("AllFeature/Selected_Feature/feature_EDA.txt", "a+")
            df_file.write("%s\n" % (df.keys()))
            df_file.close()



        elif "EMGMF" in fp:
            df = pd.DataFrame()
            df_1 = pd.DataFrame()

            df_1_N = pd.DataFrame.from_dict(loadData['EMGMF', "N"])
            df_1_N = df_1_N.drop("Class", axis=1)

            for id in df_1_N.keys():
                if "EMG_Activation_MF" in id:
                    new = downslamplig_counter(df_1_N[id], 1000, 1)
                    df_1[id] = new
                else:
                    new = downslamplig(df_1_N[id], 1000, 1)
                    df_1[id] = new
            df_1["Class"] = "N"
            df = df.append([df_1], ignore_index=True)



            df_1_F = pd.DataFrame.from_dict(loadData['EMGMF', "F"])
            df_1_F = df_1_F.drop("Class", axis=1)

            for id in df_1_F.keys():
                if "EMG_Activation_MF" in id:
                    new = downslamplig_counter(df_1_F[id], 1000, 1)
                    df_1[id] = new
                else:
                    new = downslamplig(df_1_F[id], 1000, 1)
                    df_1[id] = new
            df_1["Class"] = "F"
            df = df.append([df_1], ignore_index=True)



            df_1_H = pd.DataFrame.from_dict(loadData['EMGMF', "H"])
            df_1_H = df_1_H.drop("Class", axis=1)

            for id in df_1_H.keys():
                if "EMG_Activation_MF" in id:
                    new = downslamplig_counter(df_1_H[id], 1000, 1)
                    df_1[id] = new
                else:
                    new = downslamplig(df_1_H[id], 1000, 1)
                    df_1[id] = new
            df_1["Class"] = "H"
            df = df.append([df_1], ignore_index=True)
            print(df)

            '''print(df_1)
            for id in df_1.keys():
                new = downslamplig(df_1[id], 3000, 1)
                df[id] = new'''
            

            '''feature = pd.DataFrame({"EMG_Pulse_Onsets_MF": df["EMG_Pulse_Onsets_MF"]})

            del df["EMG_Pulse_Onsets_MF"]

            for id in df.keys():
                new = downslamplig(df[id], 6000, 1)
                df[id] = new

            feature = feature.dropna(how='all', axis=0)
            df = df.dropna(how='all', axis=0)
            '''
            df_file = open("AllFeature/Selected_Feature/feature_EMGMF.txt", "a+")
            df_file.write("%s\n" % (df.keys()))
            df_file.close()


        elif "EMGZ" in fp:
            df = pd.DataFrame()
            df_1 = pd.DataFrame()

            df_1_N = pd.DataFrame.from_dict(loadData['EMGZ', "N"])
            df_1_N = df_1_N.drop("Class", axis=1)

            for id in df_1_N.keys():
                if "EMG_Activation_Z" in id:
                    new = downslamplig_counter(df_1_N[id], 1000, 1)
                    df_1[id] = new
                else:
                    new = downslamplig(df_1_N[id], 1000, 1)
                    df_1[id] = new
            df_1["Class"] = "N"
            df = df.append([df_1], ignore_index=True)



            df_1_F = pd.DataFrame.from_dict(loadData['EMGZ', "F"])
            df_1_F = df_1_F.drop("Class", axis=1)

            for id in df_1_F.keys():
                if "EMG_Activation_Z" in id:
                    new = downslamplig_counter(df_1_F[id], 1000, 1)
                    df_1[id] = new
                else:
                    new = downslamplig(df_1_F[id], 1000, 1)
                    df_1[id] = new
            df_1["Class"] = "F"
            df = df.append([df_1], ignore_index=True)



            df_1_H = pd.DataFrame.from_dict(loadData['EMGZ', "H"])
            df_1_H = df_1_H.drop("Class", axis=1)

            for id in df_1_H.keys():
                if "EMG_Activation_Z" in id:
                    new = downslamplig_counter(df_1_H[id], 1000, 1)
                    df_1[id] = new
                else:
                    new = downslamplig(df_1_H[id], 1000, 1)
                    df_1[id] = new
            df_1["Class"] = "H"
            df = df.append([df_1], ignore_index=True)
            print(df)

            '''


            del df["EMG_Pulse_Onsets_Z"]

            for id in df.keys():
                new = downslamplig(df[id], 6000, 1)
                df[id] = new

            feature = feature.dropna(how='all', axis=0)
            df = df.dropna(how='all', axis=0)'''

            df_file = open("AllFeature/Selected_Feature/feature_EMGZ.txt", "a+")
            df_file.write("%s\n" % (df.keys()))
            df_file.close()


        thresh_df = np.std(df)*0.2 # usar 20%
        hrv_file = open("AllFeature/Selected_Feature/" + name+".txt", "a+")
        print("std: %s, thresh:%s"%(np.std(df), thresh_df))
        hrv_file.write("Standard Deviation:\n%s Threshold:\n%s \n" % (np.std(df), thresh_df))
        hrv_file.close()
        
        
        y = np.array(df["Class"])

        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)

        df = df.drop("Class", axis=1)
     
        feat_df = open("AllFeature/Correlation3_"+name+".txt", "a+")
        find_corr_df = find_correlation(df, thresh_df)

        #corr_find_df = df.drop(find_corr_df, axis=1)
        feat_df.write("All Feature %s\n"%(df.keys()) )
        feat_df.write("\n Select features to:%s\n"%(find_corr_df))
        feat_df.close()
        

        feat_df = open("AllFeature/Variance3_"+name+".txt", "a+")
        variance_df = variance_thresh(df, thresh_df)
        feat_df.write("All Feature %s\n"%(df.keys()) )
        feat_df.write("\n Select features to:%s\n"%(variance_df))
        feat_df.close()
        
        feat_df = open("AllFeature/Select_K_best3_"+name+".txt", "a+")
        feat_df.write("All Feature \t\t\t Select k Best removing: \n")
        select_df = select_best(df, thresh_df)
        print("feature df: %s, Selected_df:%s" % (df.keys(), select_df))
        feat_df.write("All Feature %s\n" % (df.keys()))
        feat_df.close()

        
        feat_df = open("AllFeature/Forward3_"+name+".txt", "a+")  
        forward_df = forward_selection(df, y)
        feat_df.write("All Feature %s\n" % (df.keys()))
        feat_df.write("Forward Selection Selected %s\n" % (forward_df))
        feat_df.close()
        
    
        feat_df = open("AllFeature/Backward3_"+name+".txt", "a+")
        feat_df.write(" \t\t\t Backward Selection Selected: \n")
        backward_df = backward_elimination(df, y)    
        feat_df.write("All Feature %s \n" % (df.keys()))
        feat_df.write("Backward Selection Selected:%s" % (backward_df))
        feat_df.close()

def plots_box(type, key):
    #box_ploting(type,key)

    box_ploting_all(type, key)

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def featureExtraction_person(name, key):
    path = 'Data_inc/*' + name + '*'

    Eda_df = ["EDA_Phasic", "EDA_Tonic", "SCR_Onsets", "SCR_Peaks_Indexes"]

    EMG_df = ["EMG_Activation_Z", "EMG_Envelope_Z", "EMG_Pulse_Onsets_Z"]

    EMG_df_MF = ["EMG_Activation_MF", "EMG_Envelope_MF", "EMG_Pulse_Onsets_MF"]

    ECG_df = ["Heart_Rate", "ECG_RR_Interval", 'ECG_HRV_HF', 'ECG_HRV_LF', 'ECG_HRV_ULF', 'ECG_HRV_VHF', 'ECG_HRV_VLF', "T_Waves", "P_Waves", "Q_Waves"]


    for fp in glob.glob(os.path.join(path)):  # , '*.mat' os.path.join

        if not key in dict_feature:
            dict_feature[key] = {}


        loadData = pickle.load(open(fp, "rb"))
        df = None

        i = 0
        if "ECG" in fp:
            df_N = pd.DataFrame()
            df_F = pd.DataFrame()
            df_H = pd.DataFrame()
            feature = pd.DataFrame()

            
            df_1_N = pd.DataFrame.from_dict(loadData['ECG', "ID1_N"])

            df_1_F = pd.DataFrame.from_dict(loadData['ECG', "ID1_F"])

            df_1_H = pd.DataFrame.from_dict(loadData['ECG', "ID1_H"])



            #df_1["ECG_RR_Interval"].fillna(np.nanmean(df_1["ECG_RR_Interval"]), inplace=True)
            #df_1["ECG_RR_Interval"].interpolate(method='polynomial', order=2).ffill().bfill()

        
            for id in df_1_N.keys():
                new = downslamplig(df_1_N[id], 1000, 1)
                df_N[id] = new 

            for id in df_1_F.keys():
                new = downslamplig(df_1_F[id], 1000, 1)
                df_F[id] = new 

            for id in df_1_H.keys():
                new = downslamplig(df_1_H[id], 1000, 1)
                df_H[id] = new            
            
            #sprint(df)

            plt.title("T Waves")
            plt.plot(df_N["T_Waves"],"blue", label="Neutral")
            plt.plot(df_F["T_Waves"],"orange", label="Fear")
            plt.plot(df_H["T_Waves"],"green", label="Happy")
            plt.legend()
            plt.grid()
            #plt.savefig("Plot_Participant/Aggregados/"+k[0]+"_"+k[1]+"_"+keyVec+".png")
            #plt.close()
            plt.show()


            break

            print(len(df))
            new_df = {n: df[n] for n in ECG_df}

            print(len(new_df))
            


        elif "EDA" in fp:
            df = pd.DataFrame()
            feature = pd.DataFrame()
            df_1 = pd.DataFrame.from_dict(loadData["EDA", key])
            
            df_1["SCR_Onsets"] = df_1["SCR_Onsets"].fillna(0)

            del df_1["SCR_Peaks_Amplitudes"]

            for id in df_1.keys():
                if "SCR_Onsets" in id:
                    new = downslamplig_counter(df_1[id], 6000, 1)
                    df[id] = new
                else:
                    new = downslamplig(df_1[id], 6000, 1)
                    df[id] = new
            print(df)
            print(len(df))
            new_df = {n: df[n] for n in Eda_df}
            print(len(new_df))


        elif "EMGMF" in fp:
            df =pd.DataFrame()
            df_1 = pd.DataFrame.from_dict(loadData["EMGMF", key])
            df_1["EMG_Envelope_MF"].interpolate(method='polynomial', order=5)


            for id in df_1.keys():
                if "EMG_Activation_MF" in id:
                    new = downslamplig_counter(df_1[id], 6000, 1)
                    df[id] = new
                else:
                    new = downslamplig(df_1[id], 6000, 1)
                    df[id] = new
            
            print(df)
            print(len(df))
            new_df = {n: df[n] for n in EMG_df_MF}
            print(len(new_df))

        elif "EMGZ" in fp:

            df = pd.DataFrame()
            df_1 = pd.DataFrame.from_dict(loadData["EMGZ", key])
            df_1["EMG_Envelope_Z"].interpolate(method='polynomial', order=5)

            for id in df_1.keys():
                if "EMG_Activation_Z" in id:
                    new = downslamplig_counter(df_1[id], 6000, 1)
                    df[id] = new
                else:
                    new = downslamplig(df_1[id], 6000, 1)
                    df[id] = new

            print(df)
            print(len(df))
            new_df = {n: df[n] for n in EMG_df}
            print(len(new_df))

        dict_feature[key] = {**new_df, **dict_feature[key]}

        output = open("Data_inc/split_person/All_60s.pkl", 'wb')
        pickle.dump(dict_feature, output, protocol=4)

def featureExtraction_emotion(name):
    path = 'Data_inc/Emotion/*' + name + '*'
    for fp in glob.glob(os.path.join(path)):  # , '*.mat' os.path.join
        print(fp)
        key_participant = fp.split("/")[2]
        key = key_participant.split(".")[0]
        key = key.split("_")[1]
        

        if not name in dict_emotion and not key in dict_emotion:
            dict_emotion[name, key] = {}
        loadData = pickle.load(open(fp, "rb"))
        df = None

        i = 0
        if "ECG" in fp:
            df = pd.DataFrame.from_dict(loadData['ECG', key])
            del df["Class"]
            #feature = pd.DataFrame({"P_Waves": df["P_Waves"]})

            #del df["T_Waves"]
            #del df["Q_Waves"]
            #del df["P_Waves"]

            '''for id in df.keys():
                new = downslamplig(df[id], 6000, 1)
                df[id] = new'''
            

            #feature = feature.dropna(how='all', axis=0)
            #df = df.dropna(how='all', axis=0)

            dict_emotion["ECG", key] = {**df}
            print(dict_emotion["ECG", key])


        elif "EDA" in fp:
            df = pd.DataFrame()
            df = pd.DataFrame.from_dict(loadData["EDA", key])
            del df["Class"]
            #df["SCR_Peaks_Amplitudes"].fillna(0, inplace=True)

            #df["SCR_Peaks_Amplitudes"] = df["SCR_Peaks_Amplitudes"].dropna()
            #df["SCR_Peaks_Amplitudes"].interpolate(method='polynomial', order=2)



            #feature = pd.DataFrame({"SCR_Peaks_Amplitudes": df_1["SCR_Peaks_Amplitudes"], "SCR_Onsets": df_1["SCR_Onsets"]})


            '''del df_1["SCR_Peaks_Amplitudes"]
            del df_1["SCR_Onsets"]

            for id in df.keys():

                new = downslamplig(df[id], 6000, 1)
                df[id] = new

            feature = feature.dropna(how='all', axis=0)
            df = df.dropna(how='all', axis=0)
            '''
            dict_emotion["EDA", key] = {**df}


        elif "EMGMF" in fp:
            df = pd.DataFrame.from_dict(loadData["EMGMF", key])
            del df["Class"]

            #feature = pd.DataFrame({"EMG_Pulse_Onsets_MF": df["EMG_Pulse_Onsets_MF"]})


            #del df["EMG_Pulse_Onsets_MF"]

            '''for id in df.keys():
                if "SCR_Onsets" in id:
                    new = downslamplig_counter(df_1[id], 6000, 1)
                    df[id] = new
                else_
                    new = downslamplig(df[id], 6000, 1)
                    df[id] = new'''


            dict_emotion["EMGMF", key] = {**df}



        elif "EMGZ" in fp:
            df = pd.DataFrame.from_dict(loadData["EMGZ", key])
            del df["Class"]

            #feature = pd.DataFrame({"EMG_Pulse_Onsets_Z": df["EMG_Pulse_Onsets_Z"]})


            #del df["EMG_Pulse_Onsets_Z"]

            '''for id in df.keys():
                new = downslamplig(df[id], 6000, 1)
                df[id] = new'''

            dict_emotion["EMGZ", key] = {**df}

        output = open('Data_inc/For_split/'+name+'.pkl', 'wb')
        pickle.dump(dict_emotion, output, protocol=4)

def featureExtraction_person_all():
    for key in keysRead:
        print(key)
        featureExtraction_person("ECG", key)
        #featureExtraction_person("EMGZ", key)
        #featureExtraction_person("EMGMF", key)
        #featureExtraction_person("EDA", key)

def featureExtraction_emotion_all():
    featureExtraction_emotion("ECG")
    featureExtraction_emotion("EMGZ")
    featureExtraction_emotion("EMGMF")
    featureExtraction_emotion("EDA")


    plots_box("ECG", "N")
    plots_box("EMGZ", "N")
    plots_box("EMGMF", "N")
    plots_box("EDA", "N")

def extract_features_all_metrics():
    featureExtraction_metrics("ECG")
    featureExtraction_metrics("EMGZ")
    featureExtraction_metrics("EMGMF")
    featureExtraction_metrics("EDA")

def grouper(iterable, n, fillvalue=None):
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def extract_features():
    featureVector1 = {}

    Eda_df = ["EDA_Phasic", "EDA_Tonic"]
    Eda_feature = ["SCR_Peaks_Amplitudes", "SCR_Onsets"]  # SCR_Onsets

    EMG_df = ["EMG_Activation", "EMG_Envelope"]
    EMG_Feature = ["EMG_Pulse_Onsets"]  # EMG_Pulse_Onsets

    #EMG_df_MF = ["EMG_Activation_MF", "EMG_Envelope_MF"]
    #EMG_Feature_MF = ["EMG_Pulse_Onsets_MF"]  # EMG_Pulse_Onsets

    ECG_Feature = ["T_Waves", "P_Waves", "Q_Waves"]
    ECG_df = ["Heart_Rate", "ECG_RR_Interval", 'ECG_HRV_HF', 'ECG_HRV_LF', 'ECG_HRV_ULF', 'ECG_HRV_VHF', 'ECG_HRV_VLF']

    i=0
    for key in keysRead:
        print(key)
        '''
        try:
            print("ECG")
            dict_feature = nk.ecg_process(window_ECG[key],  sampling_rate=1000)
            del window_ECG[key]
            df_ECG = dict_feature
            #output = open('/home/gisela.pinto/Tese/Data_pkl/ECG.pkl', 'wb')
            output = open('Data_pkl/ECG_'+key+'.pkl', 'wb')
            pickle.dump(df_ECG, output, protocol=4)
            print("Writing in pkl")


        except Exception as e:
            print("ola")
            print("ola")
            print(e)
            continue
        
        ######     EMGZ        #######
        print("EMGZ")
        dict_feature = nk.emg_process(window_EMGZ[key], sampling_rate=1000)
    
        #new_dict_feature_EMGZ = {df: dict_feature['df'][df] for df in EMG_df}
        #new_feature_EMGZ = {df: dict_feature['EMG'][df] for df in EMG_Feature}
        del window_EMGZ[key]
        df_EMGZ = dict_feature
        #output = open('/home/gisela.pinto/Tese/Data_pkl/EMGZ.pkl', 'wb')
        output = open('Data_pkl/EMGZ_'+key+'.pkl', 'wb')
        pickle.dump(df_EMGZ, output, protocol=4)
        print("Writing in pkl")



        ######     EMGMF        #######
        print("EMGMF")
        dict_feature = nk.emg_process(window_EMGMF[key], sampling_rate=1000)
        #new_dict_feature_EMGMF = {df: dict_feature['df'][df] for df in EMG_df}
        #new_feature_EMGMF = {df: dict_feature['EMG'][df] for df in EMG_Feature}
        del window_EMGMF[key]
        df_EMGMF = dict_feature
        #output = open('/home/gisela.pinto/Tese/Data_pkl/EMGMF.pkl', 'wb')
        output = open('Data_pkl/EMGMF_'+key+'.pkl', 'wb')
        pickle.dump(df_EMGMF, output, protocol=4)
        print("Writing in pkl")
        '''


        ######     EDA        #######
        print("EDA")
        dict_feature = nk.eda_process(window_EDA[key], sampling_rate=1000)
        #new_dict_feature_EDA = {df: dict_feature['df'][df] for df in Eda_df}
        #new_feature_EDA = {df: dict_feature['EDA'][df] for df in Eda_feature}
    
        del window_EDA[key]
        df_EDA = dict_feature
        #output = open('/home/gisela.pinto/Tese/Data_pkl/EDA.pkl', 'wb')
        output = open('Data_pkl/EDA_'+key+'.pkl', 'wb')
        pickle.dump(df_EDA, output, protocol=4)
        print("Writing in pkl")

        
        '''
        df = {**new_dict_feature_ECG, **new_dict_ECG, **new_dict_feature_EMGZ, **new_feature_EMGZ,
              **new_dict_feature_EMGMF, **new_feature_EMGMF, **new_dict_feature_EDA, **new_feature_EDA}
        
        for k in df.keys():
            try:
                df[k] = df[k][~np.isnan(df[k])]
                df[k] = np.median(df[k])
            except Exception as e:
                print(e)
                print("ola")
                print(df[k])

        temp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in df.items()]))
        temp.fillna(temp.mean(), inplace=True)

        i = i + 1
        featureVector[key] = []
        featureVector[key].append(temp.values[0])

        print("feature: ",featureVector[key])
        '''
        
    #output = open('/home/gisela.pinto/Tese/Data_inc/NoPCA.pkl', 'wb')
    #output = open('Data_inc/AllFeature.pkl', 'wb')
    #pickle.dump(featureVector, output, protocol=4)
