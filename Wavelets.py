#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:30:19 2018

@author: shaydineen
"""

import pywt

import pandas as pd
import numpy as np

#dataset = pd.read_excel('SP500.xlsx')
dataset = pd.read_csv('ETF_Opportunity_Set/IWM/iwn.csv')
dataset = dataset.iloc[:, 1:].values

coeffs = pywt.dwt2(data = dataset, wavelet = 'sym4', axes = (-1,1)) #-1,1 and 1,-1equals 2078,10
print(coeffs[0].shape)
coeffs2 = pywt.dwt2(data = coeffs[0], wavelet = 'sym4', axes = (-1, -1))
print(coeffs2[0].shape)

filepath = 'iwm_wavelet.csv'
df = pd.DataFrame(coeffs2[0])
df.to_csv(filepath, index=False)


pywt.dwt2()