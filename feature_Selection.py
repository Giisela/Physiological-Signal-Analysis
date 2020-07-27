############# library ##########
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pickle
############# files ###########
from WriteFiles import *
from Global_var import *


def select_per_emotion(name):
    path = 'Data_pkl/*' + name + '*'

    print(glob.glob(os.path.join(path)))

    Eda_df = ["EDA_Phasic", "EDA_Tonic", "SCR_Onsets", "SCR_Peaks_Indexes", "Class"]
    Eda_feature = ["SCR_Peaks_Amplitudes"]  # SCR_Onsets

    EMG_df = ["EMG_Activation_Z", "EMG_Envelope_Z", "EMG_Pulse_Onsets_Z", "Class"]

    EMG_df_MF = ["EMG_Activation_MF", "EMG_Envelope_MF", "EMG_Pulse_Onsets_MF", "Class"]

    ECG_Feature = ["T_Waves", "P_Waves", "Q_Waves", "Class"]
    ECG_df = ["Heart_Rate", "ECG_RR_Interval", 'ECG_HRV_HF', 'ECG_HRV_LF', 'ECG_HRV_ULF', 'ECG_HRV_VHF', 'ECG_HRV_VLF']


    for fp in glob.glob(os.path.join(path)):  # , '*.mat' os.path.join
        print(fp)
        key_participant = fp.split("/")[1]
        key_participant = key_participant.split(".")[0]
        key = key_participant.split("_")[1]+"_"+key_participant.split("_")[2]
        key_letter = key_participant.split("_")[2]

        if not name in featureVector and not key_letter in featureVector:
            featureVector[name, key_letter] = {}

        loadData = pickle.load(open(fp, "rb"))
        df = None
        feature = None
        hrv = None
        if "ECG" in fp:
            print(key)
            feature = pd.DataFrame()
            df = pd.DataFrame.from_dict(loadData["df"])
            df["ECG_RR_Interval"].fillna(np.nanmean(df["ECG_RR_Interval"]), inplace=True)
            df["ECG_RR_Interval"].interpolate(method='polynomial', order=2)

            df.fillna(df.mean(), inplace=True)

            val = loadData["ECG"]["Average_Signal_Quality"]
            loadData["ECG"]["Average_Signal_Quality"] = []
            loadData["ECG"]["Average_Signal_Quality"].append(val)

            del loadData["ECG"]["HRV"]
            del loadData["ECG"]["Probable_Lead"]

            feature_1 = pd.DataFrame.from_dict(loadData["ECG"], orient='index').T
            feature_1.fillna(np.nanmean(feature_1), inplace=True)

            feature_value=[]
            feature_value1=[]
            feature_value2=[]
            p_waves = np.array(feature_1["P_Waves"].dropna(), dtype=int)
            t_waves = np.array(feature_1["T_Waves"].dropna(), dtype=int)
            q_waves = np.array(feature_1["Q_Waves"].dropna(), dtype=int)
            for index, value in enumerate(window_ECG[key]):
                if index in p_waves:
                    feature_value.append(value)
                else:
                    feature_value.append(np.nan)
                
                if index in t_waves:
                    feature_value1.append(value)
                else:
                    feature_value1.append(np.nan)
                
                if index in q_waves:
                    feature_value2.append(value)
                else:
                    feature_value2.append(np.nan)
            

            del feature_1["P_Waves"]
            del feature_1["T_Waves"]
            del feature_1["Q_Waves"]

            feature = pd.DataFrame({"P_Waves":feature_value,"T_Waves":feature_value1, "Q_Waves":feature_value2, "Class":key_letter})
            feature = feature.fillna(method='ffill')
            feature = feature.fillna(method='bfill')
            print(df)
            print(feature)

            new_df = {n: df[n] for n in ECG_df}
            new_feature = {n: feature[n] for n in ECG_Feature}


            '''if "N" in key_letter:
                featureVector[name, "N"] = {**new_df, **new_feature, **featureVector[name, "N"]}
                print("feature ECG: ", featureVector[name, key_letter])
                output = open('Data_inc/Emotion/ECG_N.pkl', 'wb')
            elif "F" in key_letter:
                featureVector[name, "F"] = {**new_df, **new_feature, **featureVector[name, "F"]}
                print("feature ECG: ", featureVector[name, key_letter])
                output = open('Data_inc/Emotion/ECG_F.pkl', 'wb')
            else:
                featureVector[name, "H"] = {**new_df, **new_feature, **featureVector[name, "H"]}
                print("feature ECG: ", featureVector[name, key_letter])
                output = open('Data_inc/Emotion/ECG_H.pkl', 'wb')
            '''
            featureVector[name, key_letter] = {**new_df, **new_feature, **featureVector[name,key_letter]}
            print("feature ECG: ", featureVector[name, key_letter])
            output = open('Data_inc/Per_Emotion/ECG.pkl', 'wb')


        elif "EDA" in fp:
            
            df = pd.DataFrame.from_dict(loadData["df"])
            #df["EDA_Tonic"].interpolate(method='polynomial', order=2)
            #df["EDA_Phasic"].interpolate(method='polynomial', order=2)
            df["SCR_Onsets"] = df["SCR_Onsets"].fillna(0)
            
            feature= pd.DataFrame.from_dict(loadData["EDA"])

            print(feature["SCR_Peaks_Indexes"])

            feature_value=[]
            pulse_onsets = np.array(feature["SCR_Peaks_Indexes"].dropna(), dtype=int)
            for index, value in enumerate(window_EDA[key]):
                if index in pulse_onsets:
                    print("here")
                    feature_value.append(value)
                else:
                    feature_value.append(np.nan)

            print(feature_value)

            del feature["SCR_Peaks_Indexes"]

            df["SCR_Peaks_Indexes"] = feature_value
            df["SCR_Peaks_Indexes"] = df["SCR_Peaks_Indexes"].fillna(method='ffill')
            df["SCR_Peaks_Indexes"] = df["SCR_Peaks_Indexes"].fillna(method='bfill')
            
            df["Class"] = key_letter
            print(df)
            print(feature)

            new_df = {n: df[n] for n in Eda_df}
            new_feature = {n: feature[n] for n in Eda_feature}

            '''if "N" in key:
                featureVector[name, "N"] = {**new_df, **new_feature, **featureVector[name, "N"]}
                #print("feature EDA: ", featureVector[name, key])
                output = open('Data_inc/Emotion/EDA_N.pkl', 'wb')

            elif "F" in key:
                featureVector[name, "F"] = {**new_df, **new_feature, **featureVector[name, "F"]}
                #print("feature EDA: ", featureVector[name, key])
                output = open('Data_inc/Emotion/EDA_F.pkl', 'wb')
            else:
                featureVector[name, "H"] = {**new_df, **new_feature, **featureVector[name, "H"]}
                # print("feature EDA: ", featureVector[name, key])
                output = open('Data_inc/Emotion/EDA_H.pkl', 'wb')'''
            featureVector[name, key_letter] = {**new_df, **new_feature, **featureVector[name,key_letter]}
            print("feature EDA: ", featureVector[name, key_letter])
            output = open('Data_inc/Per_Emotion/EDA.pkl', 'wb')

        elif "EMG" in fp:
            df = pd.DataFrame.from_dict(loadData["df"])

            feature = pd.DataFrame.from_dict(loadData["EMG"])

            if "EMGZ" in fp:
                almost_df = pd.DataFrame()
                almost_feature_1 = pd.DataFrame()
                almost_feature = pd.DataFrame()
                for k in df.keys():
                    almost_df[k + "_Z"] = df[k]

                for k in feature.keys():
                    almost_feature_1[k + "_Z"] = feature[k]
                
                almost_df["EMG_Envelope_Z"].interpolate(method='polynomial', order=3)
                almost_df["EMG_Activation_Z"].interpolate(method='polynomial', order=1)
                almost_df["Class"] = key_letter
                print(almost_feature_1["EMG_Pulse_Onsets_Z"])

                feature_value=[]
                pulse_onsets = np.array(almost_feature_1["EMG_Pulse_Onsets_Z"].dropna(), dtype=int)
                for index, value in enumerate(window_EMGZ[key]):
                    if index in pulse_onsets:
                        feature_value.append(value)
                    else:
                        feature_value.append(np.nan)

                del almost_feature_1["EMG_Pulse_Onsets_Z"]
                
                almost_df["EMG_Pulse_Onsets_Z"] = feature_value
                almost_df["EMG_Pulse_Onsets_Z"] = almost_df["EMG_Pulse_Onsets_Z"].fillna(method='ffill')
                almost_df["EMG_Pulse_Onsets_Z"] = almost_df["EMG_Pulse_Onsets_Z"].fillna(method='bfill')
                
                new_df = {n: almost_df[n] for n in EMG_df}

                print(new_df)

                '''if "N" in key:
                    featureVector[name, "N"] = {**new_df, **featureVector[name, "N"]}
                    # print("feature EDA: ", featureVector[name, key])
                    output = open('Data_inc/Emotion/EMGZ_N.pkl', 'wb')

                elif "F" in key:
                    featureVector[name, "F"] = {**new_df, **featureVector[name, "F"]}
                    # print("feature EDA: ", featureVector[name, key])
                    output = open('Data_inc/Emotion/EMGZ_F.pkl', 'wb')
                else:
                    featureVector[name, "H"] = {**new_df, **featureVector[name, "H"]}
                    # print("feature EDA: ", featureVector[name, key])
                    output = open('Data_inc/Emotion/EMGZ_H.pkl', 'wb')'''
                featureVector[name, key_letter] = {**new_df, **featureVector[name,key_letter]}
                print("feature EMGZ: ", featureVector[name, key_letter])
                output = open('Data_inc/Per_Emotion/EMGZ.pkl', 'wb')
            else:
                almost_df = pd.DataFrame()
                almost_feature_1 = pd.DataFrame()
                almost_feature = pd.DataFrame()
                for k in df.keys():
                    almost_df[k + "_MF"] = df[k]

                for k in feature.keys():
                    almost_feature_1[k + "_MF"] = feature[k]
                almost_df["Class"] = key_letter
                almost_df["EMG_Envelope_MF"].interpolate(method='polynomial', order=3)
                almost_df["EMG_Activation_MF"].interpolate(method='polynomial', order=1)

                print(almost_feature_1["EMG_Pulse_Onsets_MF"])

                feature_value=[]
                pulse_onsets = np.array(almost_feature_1["EMG_Pulse_Onsets_MF"].dropna(), dtype=int)
                for index, value in enumerate(window_EMGMF[key]):
                    if index in pulse_onsets:
                        feature_value.append(value)
                    else:
                        feature_value.append(np.nan)

                del almost_feature_1["EMG_Pulse_Onsets_MF"]

                almost_df["EMG_Pulse_Onsets_MF"] = feature_value
                almost_df["EMG_Pulse_Onsets_MF"] = almost_df["EMG_Pulse_Onsets_MF"].fillna(method='ffill')
                almost_df["EMG_Pulse_Onsets_MF"] = almost_df["EMG_Pulse_Onsets_MF"].fillna(method='bfill')


                new_df = {n: almost_df[n] for n in EMG_df_MF}

                print(new_df)

                '''if "N" in key:
                    featureVector[name, "N"] = {**new_df, **featureVector[name, "N"]}
                    # print("feature EDA: ", featureVector[name, key])
                    output = open('Data_inc/Emotion/EMGMF_N.pkl', 'wb')

                elif "F" in key:
                    featureVector[name, "F"] = {**new_df, **featureVector[name, "F"]}
                    # print("feature EDA: ", featureVector[name, key])
                    output = open('Data_inc/Emotion/EMGMF_F.pkl', 'wb')
                else:
                    featureVector[name, "H"] = {**new_df, **featureVector[name, "H"]}
                    # print("feature EDA: ", featureVector[name, key])
                    output = open('Data_inc/Emotion/EMGMF_H.pkl', 'wb')'''
                featureVector[name, key_letter] = {**new_df, **featureVector[name,key_letter]}
                print("feature EMGMF: ", featureVector[name, key_letter])
                output = open('Data_inc/Per_Emotion/EMGMF.pkl', 'wb')

        pickle.dump(featureVector, output, protocol=4)

def select(name):
    path = 'Data_pkl/*' + name + '*'

    print(glob.glob(os.path.join(path)))

    Eda_df = ["EDA_Phasic", "EDA_Tonic", "SCR_Onsets", "SCR_Peaks_Indexes"]
    Eda_feature = ["SCR_Peaks_Amplitudes"]  # SCR_Onsets

    EMG_df = ["EMG_Activation_Z", "EMG_Envelope_Z", "EMG_Pulse_Onsets_Z"]

    EMG_df_MF = ["EMG_Activation_MF", "EMG_Envelope_MF", "EMG_Pulse_Onsets_MF"]

    ECG_Feature = ["T_Waves", "P_Waves", "Q_Waves"]
    ECG_df = ["Heart_Rate", "ECG_RR_Interval", 'ECG_HRV_HF', 'ECG_HRV_LF', 'ECG_HRV_ULF', 'ECG_HRV_VHF', 'ECG_HRV_VLF']



    for fp in glob.glob(os.path.join(path)):  # , '*.mat' os.path.join
        print(fp)
        key = fp.split("/")[1]
        key = key.split(".")[0]
        key = key.split("_")[1]+"_"+key.split("_")[2]

        if not name in featureVector and not key in featureVector:
            featureVector[name, key] = {}

        loadData = pickle.load(open(fp, "rb"))
        df = None
        feature = None
        hrv = None
        if "ECG" in fp:
            print(key)
            df = pd.DataFrame(loadData["df"])
            df.fillna(np.nanmean(df), inplace=True)
        
            df["ECG_RR_Interval"].interpolate(method='polynomial', order=2).ffill().bfill()
            df["ECG_RR_Interval"].fillna(np.nanmean(df["ECG_RR_Interval"]), inplace=True)
            

            val = loadData["ECG"]["Average_Signal_Quality"]
            loadData["ECG"]["Average_Signal_Quality"] = []
            loadData["ECG"]["Average_Signal_Quality"].append(val)

            del loadData["ECG"]["HRV"]
            del loadData["ECG"]["Probable_Lead"]

            feature = pd.DataFrame.from_dict(loadData["ECG"], orient='index').T
            '''feature_value=[]
            feature_value1=[]
            feature_value2=[]
            p_waves = np.array(feature_1["P_Waves"].dropna(), dtype=int)
            t_waves = np.array(feature_1["T_Waves"].dropna(), dtype=int)
            q_waves = np.array(feature_1["Q_Waves"].dropna(), dtype=int)
            for index, value in enumerate(window_ECG[key]):
                if index in p_waves:
                    feature_value.append(value)
                else:
                    feature_value.append(np.nan)
                
                if index in t_waves:
                    feature_value1.append(value)
                else:
                    feature_value1.append(np.nan)
                
                if index in q_waves:
                    feature_value2.append(value)
                else:
                    feature_value2.append(np.nan)
            

            del feature_1["P_Waves"]
            del feature_1["T_Waves"]
            del feature_1["Q_Waves"]
            '''
            #feature = pd.DataFrame({"P_Waves":feature_value,"T_Waves":feature_value1, "Q_Waves":feature_value2})
            #feature = feature.fillna(method='ffill')
            #feature = feature.fillna(method='bfill')
            #print(feature)
            #print(df)

            #feature["P_Waves"].interpolate(method='nearest')

            new_df = {n: df[n] for n in ECG_df}
            new_feature = {n: feature[n] for n in ECG_Feature}

            featureVector[name, key] = {**new_df, **new_feature, **featureVector[name, key]}
            

            #print("feature ECG: ", featureVector[name, key])
            #output = open('Data_inc/ECG_participants.pkl', 'wb')


        elif "EDA" in fp:
            df = pd.DataFrame.from_dict(loadData["df"])
            #df["EDA_Tonic"].interpolate(method='polynomial', order=2)
            #df["EDA_Phasic"].interpolate(method='polynomial', order=2)
            df["SCR_Onsets"] = df["SCR_Onsets"].fillna(0)
            
            feature = pd.DataFrame.from_dict(loadData["EDA"])

            print(feature["SCR_Peaks_Indexes"])

            feature_value=[]
            pulse_onsets = np.array(feature["SCR_Peaks_Indexes"].dropna(), dtype=int)
            for index, value in enumerate(window_EDA[key]):
                if index in pulse_onsets:
                    print("here")
                    feature_value.append(value)
                else:
                    feature_value.append(np.nan)

            print(feature_value)

            del feature["SCR_Peaks_Indexes"]

            df["SCR_Peaks_Indexes"] = feature_value
            df["SCR_Peaks_Indexes"] = df["SCR_Peaks_Indexes"].fillna(method='ffill')
            df["SCR_Peaks_Indexes"] = df["SCR_Peaks_Indexes"].fillna(method='bfill')
            

            new_df = {n: df[n] for n in Eda_df}
            new_feature = {n: feature[n] for n in Eda_feature}


            featureVector[name, key] = {**new_df, **new_feature, **featureVector[name, key]}
            #print("feature EDA: ", featureVector[name, key])
            output = open('Data_inc/EDA_participants.pkl', 'wb')

        elif "EMG" in fp:
            df = pd.DataFrame(loadData["df"])
            feature = pd.DataFrame(loadData["EMG"])

            if "EMGZ" in fp:
                almost_df = pd.DataFrame()
                almost_feature_1 = pd.DataFrame()
                almost_feature = pd.DataFrame()
                for k in df.keys():
                    almost_df[k + "_Z"] = df[k]

                for k in feature.keys():
                    almost_feature_1[k + "_Z"] = feature[k]

                print(almost_df["EMG_Envelope_Z"])
                
                almost_df["EMG_Envelope_Z"].interpolate(method='polynomial', order=5).ffill().bfill()


                feature_value=[]
                pulse_onsets = np.array(almost_feature_1["EMG_Pulse_Onsets_Z"].dropna(), dtype=int)
                for index, value in enumerate(window_EMGZ[key]):
                    if index in pulse_onsets:
                        feature_value.append(value)
                    else:
                        feature_value.append(np.nan)

                del almost_feature_1["EMG_Pulse_Onsets_Z"]
                
                almost_df["EMG_Pulse_Onsets_Z"] = feature_value
                almost_df["EMG_Pulse_Onsets_Z"] = almost_df["EMG_Pulse_Onsets_Z"].fillna(method='ffill')
                almost_df["EMG_Pulse_Onsets_Z"] = almost_df["EMG_Pulse_Onsets_Z"].fillna(method='bfill')
                
                new_df = {n: almost_df[n] for n in EMG_df}
                featureVector[name, key] = {**new_df, **featureVector[name, key]}
                print(featureVector[name, key])

                #print("feature EMGZ: ", featureVector[name, key])
                output = open('Data_inc/EMGZ_participants.pkl', 'wb')

            else:
                almost_df = pd.DataFrame()
                almost_feature_1 = pd.DataFrame()
                almost_feature = pd.DataFrame()
                for k in df.keys():
                    almost_df[k + "_MF"] = df[k]

                for k in feature.keys():
                    almost_feature_1[k + "_MF"] = feature[k]
                
                
                print(almost_df["EMG_Envelope_MF"])
                almost_df["EMG_Envelope_MF"].interpolate(method='polynomial', order=5).ffill().bfill()

                feature_value=[]
                pulse_onsets = np.array(almost_feature_1["EMG_Pulse_Onsets_MF"].dropna(), dtype=int)
                for index, value in enumerate(window_EMGMF[key]):
                    if index in pulse_onsets:
                        feature_value.append(value)
                    else:
                        feature_value.append(np.nan)

                del almost_feature_1["EMG_Pulse_Onsets_MF"]

                almost_df["EMG_Pulse_Onsets_MF"] = feature_value
                almost_df["EMG_Pulse_Onsets_MF"] = almost_df["EMG_Pulse_Onsets_MF"].fillna(method='ffill')
                almost_df["EMG_Pulse_Onsets_MF"] = almost_df["EMG_Pulse_Onsets_MF"].fillna(method='bfill')


                new_df = {n: almost_df[n] for n in EMG_df_MF}
                featureVector[name, key] = {**new_df, **featureVector[name, key]}
                print(featureVector[name, key])
                #print("feature EMGMF: ", featureVector[name, key])
                output = open('Data_inc/EMGMF_participants.pkl', 'wb')

        #pickle.dump(featureVector, output, protocol=4)


def selectAll():
    select("ECG")
    #select("EMGMF")
    #select("EMGZ")
    #select("EDA")

    #select_per_emotion("ECG")
    #select_per_emotion("EMGZ")
    #select_per_emotion("EMGMF")
    #select_per_emotion("EDA")

    new = featureVector.copy()
    features = {}
    for k in new.keys():
        print(k)

        name = k

        try:
            features[name].update(new[k])
        except:
            features[name]= new[k]

    #print(features.keys())
    for k in new:
        print(k[0])
        print(k[1])

        for keyVec in featureVector[k].keys():
            #print(len(featureVector[k].keys()))
            #print(featureVector[k][keyVec])
            plt.title(keyVec)
            plt.plot(featureVector["ECG","ID1_N"]["Heart_Rate"],"b", label="Neutral")
            plt.plot(featureVector["ECG","ID1_F"]["Heart_Rate"],"yellow", label="Fear")
            plt.plot(featureVector["ECG","ID1_H"]["Heart_Rate"],"green", label="Happy")
            #plt.savefig("Plot_Participant/Aggregados/"+k[0]+"_"+k[1]+"_"+keyVec+".png")
            #plt.close()
            plt.show()
    #output = open('AllFeature/SelectALL.pkl', 'wb')
    #pickle.dump(featureVector, output, protocol=4)'''


