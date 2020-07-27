import matplotlib.pyplot as plt
from Global_var import *
import numpy as np
from matplotlib.widgets import SpanSelector

def box_ploting(type, key):
    for keyVec in dict_feature[type, key].keys():
        print(dict_feature[type, key][keyVec])
        key_H = key.split("_")[0]+"_H"
        key_F = key.split("_")[0] + "_F"
        fig, ax = plt.subplots()
        ax.set_title([key.split("_")[0]+"_"+keyVec])
        ax.boxplot([dict_feature[type,key][keyVec], dict_feature[type,key_F][keyVec], dict_feature[type,key_H][keyVec]], labels=[key, key_F, key_H], showfliers=False)
        plt.savefig("box_ploting/Per_participants_per_feature/"+type+"_"+key.split("_")[0]+"_"+keyVec+".png")

def box_ploting_all(type_s, key):

    for keyVec in dict_emotion[type_s, key].keys():
        print(keyVec)
        fig, ax = plt.subplots()
        ax.set_title([keyVec])
        ax.boxplot([dict_emotion[type_s, key][keyVec], dict_emotion[type_s, "F"][keyVec], dict_emotion[type_s, "H"][keyVec]], labels=["N", "F", "H"], showfliers=False)
        plt.savefig("Plot_Participant/Box_plot_perEmotion/"+type_s+"_"+keyVec+".png")


def segment(x, y):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(211)

    ax.plot(x, '-')
    ax.plot(y)
    ax.set_ylim(-2, 2)
   
    # set useblit True on gtkagg for enhanced performance
    span = SpanSelector(ax, onselect(y[0], y[1]), 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))

    plt.show()


def onselect(xmin, xmax):
    thisx = xmin
    thisy = xmax
    line2.set_data(thisx, thisy)
    ax2.set_xlim(thisx[0], thisx[-1])
    ax2.set_ylim(thisy.min(), thisy.max())
    fig.canvas.draw_idle()


    