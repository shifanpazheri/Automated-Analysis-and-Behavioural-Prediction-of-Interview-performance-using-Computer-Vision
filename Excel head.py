# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:34:51 2022

@author: muham
"""

import numpy as np
import os
import time
import pandas as pd


dataframeHeadScore = pd.read_excel(r"C:/Users/muham/Desktop/Head.xlsx")
print(dataframeHeadScore)
array_part = np.array(dataframeHeadScore['Participant'])
array_score = np.array(dataframeHeadScore['Overall'])
print(list(array_part).index("pp37"))
print(array_part)
print(array_score)

data = []

path=r'C:/Users/muham/Desktop/Mlabs/Final DataSet/CSV/CSV'
for excelsheet in os.listdir(path):
    excel_path=os.path.join(path,excelsheet)
    dataframe1 = pd.read_csv(excel_path)
    print(excel_path)
    #print(dataframe1)
    #print(dataframe1.loc[:,'p28_y'])
    
    head_tracking = dataframe1.loc[:,'p28_y'].to_numpy()
    
    print(head_tracking)
    participant = dataframe1.loc[2,"Participant"]
    print(participant)
    
    stableCount=0
    transientCount=0
    extremeCount=0
    
    for index in range(len(head_tracking)//10):
        if(index > 2 and index < len(head_tracking)):
            if(max(head_tracking[index-2:index+3]) - min(head_tracking[index-2:index+3]) <= 2):
                stableCount=stableCount+1
            elif((max(head_tracking[index-2:index+3])== head_tracking[index]) or (min(head_tracking[index-2:index+3]) == head_tracking[index])):
                extremeCount=extremeCount+1
            else:
                transientCount=transientCount+1
    
    #print(stableCount)
    #print(transientCount)
    #print(extremeCount)
    
    data.append([participant+"_1",100*stableCount/(stableCount+transientCount+extremeCount),100*extremeCount/(stableCount+transientCount+extremeCount),array_score[list(array_part).index(participant.lower())]])
    #print(data)
    
    for index in range(len(head_tracking)//10,2*len(head_tracking)//10):
        if(index > 2 and index < len(head_tracking)):
            if(max(head_tracking[index-2:index+3]) - min(head_tracking[index-2:index+3]) <= 2):
                stableCount=stableCount+1
            elif((max(head_tracking[index-2:index+3])== head_tracking[index]) or (min(head_tracking[index-2:index+3]) == head_tracking[index])):
                extremeCount=extremeCount+1
            else:
                transientCount=transientCount+1
    
    #print(stableCount)
    #print(transientCount)
    #print(extremeCount)
    
    data.append([participant+"_2",100*stableCount/(stableCount+transientCount+extremeCount),100*extremeCount/(stableCount+transientCount+extremeCount),array_score[list(array_part).index(participant.lower())]])
    #print(data)
    
    for index in range(2*len(head_tracking)//10,3*len(head_tracking)//10):
        if(index > 2 and index < len(head_tracking)):
            if(max(head_tracking[index-2:index+3]) - min(head_tracking[index-2:index+3]) <= 2):
                stableCount=stableCount+1
            elif((max(head_tracking[index-2:index+3])== head_tracking[index]) or (min(head_tracking[index-2:index+3]) == head_tracking[index])):
                extremeCount=extremeCount+1
            else:
                transientCount=transientCount+1
    
    #print(stableCount)
    #print(transientCount)
    #print(extremeCount)
    
    data.append([participant+"_3",100*stableCount/(stableCount+transientCount+extremeCount),100*extremeCount/(stableCount+transientCount+extremeCount),array_score[list(array_part).index(participant.lower())]])
    #print(data)
    
    for index in range(3*len(head_tracking)//10,4*len(head_tracking)//10):
        if(index > 2 and index < len(head_tracking)):
            if(max(head_tracking[index-2:index+3]) - min(head_tracking[index-2:index+3]) <= 2):
                stableCount=stableCount+1
            elif((max(head_tracking[index-2:index+3])== head_tracking[index]) or (min(head_tracking[index-2:index+3]) == head_tracking[index])):
                extremeCount=extremeCount+1
            else:
                transientCount=transientCount+1
    
    #print(stableCount)
    #print(transientCount)
    #print(extremeCount)
    
    data.append([participant+"_4",100*stableCount/(stableCount+transientCount+extremeCount),100*extremeCount/(stableCount+transientCount+extremeCount),array_score[list(array_part).index(participant.lower())]])
    #print(data)
    
    for index in range(4*len(head_tracking)//10,5*len(head_tracking)//10):
        if(index > 2 and index < len(head_tracking)):
            if(max(head_tracking[index-2:index+3]) - min(head_tracking[index-2:index+3]) <= 2):
                stableCount=stableCount+1
            elif((max(head_tracking[index-2:index+3])== head_tracking[index]) or (min(head_tracking[index-2:index+3]) == head_tracking[index])):
                extremeCount=extremeCount+1
            else:
                transientCount=transientCount+1
    
    #print(stableCount)
    #print(transientCount)
    #print(extremeCount)
    
    data.append([participant+"_5",100*stableCount/(stableCount+transientCount+extremeCount),100*extremeCount/(stableCount+transientCount+extremeCount),array_score[list(array_part).index(participant.lower())]])
    #print(data)
    
    for index in range(5*len(head_tracking)//10,6*len(head_tracking)//10):
        if(index > 2 and index < len(head_tracking)):
            if(max(head_tracking[index-2:index+3]) - min(head_tracking[index-2:index+3]) <= 2):
                stableCount=stableCount+1
            elif((max(head_tracking[index-2:index+3])== head_tracking[index]) or (min(head_tracking[index-2:index+3]) == head_tracking[index])):
                extremeCount=extremeCount+1
            else:
                transientCount=transientCount+1
    
    #print(stableCount)
    #print(transientCount)
    #print(extremeCount)
    
    data.append([participant+"_6",100*stableCount/(stableCount+transientCount+extremeCount),100*extremeCount/(stableCount+transientCount+extremeCount),array_score[list(array_part).index(participant.lower())]])
    #print(data)
    
    for index in range(6*len(head_tracking)//10,7*len(head_tracking)//10):
        if(index > 2 and index < len(head_tracking)):
            if(max(head_tracking[index-2:index+3]) - min(head_tracking[index-2:index+3]) <= 2):
                stableCount=stableCount+1
            elif((max(head_tracking[index-2:index+3])== head_tracking[index]) or (min(head_tracking[index-2:index+3]) == head_tracking[index])):
                extremeCount=extremeCount+1
            else:
                transientCount=transientCount+1
    
    #print(stableCount)
    #print(transientCount)
    #print(extremeCount)
    
    data.append([participant+"_7",100*stableCount/(stableCount+transientCount+extremeCount),100*extremeCount/(stableCount+transientCount+extremeCount),array_score[list(array_part).index(participant.lower())]])
    #print(data)
    
    for index in range(7*len(head_tracking)//10,8*len(head_tracking)//10):
        if(index > 2 and index < len(head_tracking)):
            if(max(head_tracking[index-2:index+3]) - min(head_tracking[index-2:index+3]) <= 2):
                stableCount=stableCount+1
            elif((max(head_tracking[index-2:index+3])== head_tracking[index]) or (min(head_tracking[index-2:index+3]) == head_tracking[index])):
                extremeCount=extremeCount+1
            else:
                transientCount=transientCount+1
    
    #print(stableCount)
    #print(transientCount)
    #print(extremeCount)
    
    data.append([participant+"_8",100*stableCount/(stableCount+transientCount+extremeCount),100*extremeCount/(stableCount+transientCount+extremeCount),array_score[list(array_part).index(participant.lower())]])
    #print(data)
    
    for index in range(8*len(head_tracking)//10,9*len(head_tracking)//10):
        if(index > 2 and index < len(head_tracking)):
            if(max(head_tracking[index-2:index+3]) - min(head_tracking[index-2:index+3]) <= 2):
                stableCount=stableCount+1
            elif((max(head_tracking[index-2:index+3])== head_tracking[index]) or (min(head_tracking[index-2:index+3]) == head_tracking[index])):
                extremeCount=extremeCount+1
            else:
                transientCount=transientCount+1
    
    #print(stableCount)
    #print(transientCount)
    #print(extremeCount)
    
    data.append([participant+"_9",100*stableCount/(stableCount+transientCount+extremeCount),100*extremeCount/(stableCount+transientCount+extremeCount),array_score[list(array_part).index(participant.lower())]])
    #print(data)
    
    for index in range(9*len(head_tracking)//10,10*len(head_tracking)//10):
        if(index > 2 and index < len(head_tracking)):
            if(max(head_tracking[index-2:index+3]) - min(head_tracking[index-2:index+3]) <= 2):
                stableCount=stableCount+1
            elif((max(head_tracking[index-2:index+3])== head_tracking[index]) or (min(head_tracking[index-2:index+3]) == head_tracking[index])):
                extremeCount=extremeCount+1
            else:
                transientCount=transientCount+1
    
    #print(stableCount)
    #print(transientCount)
    #print(extremeCount)
    
    data.append([participant+"_10",100*stableCount/(stableCount+transientCount+extremeCount),100*extremeCount/(stableCount+transientCount+extremeCount),array_score[list(array_part).index(participant.lower())]])
    #print(data)
    

df = pd.DataFrame(data, columns = ['participant','stablePercent', 'extremePercent','Overall'])
print(df)

df.to_excel("Head gesture tracking.xlsx",sheet_name='Sheet_name_1')
    
    
    
