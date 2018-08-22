#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 07:42:53 2018

@author: shaydineen
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('ETF_Opportunity_Set/SameTrain/spy.csv')
data = dataset.iloc[:,1:].values
hot_encoder_ready_data = data[:,0]
#Buy is denoted as 1 and sell is denoted as 0

def close_hot_encoder(dataset):
    
    onehot = []
    
    for i in range(len(dataset)-1):
        
        if dataset[i+1]>=dataset[i]:
            onehot.append(1)
        else:
            onehot.append(0)
    
    return onehot

onehot = close_hot_encoder(hot_encoder_ready_data)

filepath = 'ETF_Opportunity_Set/SameTrain/spy_onehot.csv'

df = pd.DataFrame(onehot)
df.to_csv(filepath, index=False)
