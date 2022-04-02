# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 22:17:32 2022

@author: muham
"""

import numpy as np
import os
import time
import pandas as pd

participantList = []
data=[]
final_list = []

path=r'C:/Users/muham/Desktop/Mlabs/Final DataSet/prove theorem/Each_frame_Emotion/3'
for excelsheet in os.listdir(path):
    excel_path=os.path.join(path,excelsheet)
    print(excel_path)
    dataframe1 = pd.read_excel(excel_path)
    data.append(list(dataframe1.Participant.unique()))
    #print(data)
    
#print(data)
#print(np.shape(data))

for value1 in data:
    for value2 in value1:
        participantList.append(value2)

#print(participantList)
#print(np.shape(participantList))

    for value in participantList:
        #print(dataframe1.loc[dataframe1['Participant'] == value].loc[:,'Emotion'].to_numpy())
        eachFrameEmotion = dataframe1.loc[dataframe1['Participant'] == value].loc[:,'Emotion'].to_numpy()
        print("participant = {}, eachFrameEmotion length = {}".format(value,len(eachFrameEmotion)))
        positive=['neutral','happy','surprise']
        negative=['angry','fear','sad','disguist']
        posFirst=0
        negFirst=0
        posSecond=0
        negSecond=0
        for index in range(len(eachFrameEmotion)):
            if(index<=len(eachFrameEmotion)//5):
                if(eachFrameEmotion[index] in positive):
                    posFirst=posFirst+1
                if(eachFrameEmotion[index] in negative):
                    negFirst=negFirst+1
            if(index>len(eachFrameEmotion)//5):
                if(eachFrameEmotion[index] in positive):
                    posSecond=posSecond+1
                if(eachFrameEmotion[index] in negative):
                    negSecond=negSecond+1
        
        #print("participant = {}, posFirst = {}, negFirst={} , posSecond={} , negSecond={}".format(value,posFirst,negFirst,posSecond,negSecond))
            
        first=100*posFirst/(posFirst+negFirst)
        second=100*posSecond/(posSecond+negSecond)
            
        #print("{} {} {} {}")
        final_list.append([value,first,second])   
        
#print(final_list)
df = pd.DataFrame(final_list, columns = ['participant','first', 'second'])
df.to_excel("Prove theorm3.xlsx",sheet_name='Sheet_name_1')