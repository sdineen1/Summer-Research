#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 15:11:30 2018

@author: shaydineen
"""


  
import json
import numpy as np
import pandas as pd

# =============================================================================
# Converting MACD JSON file to CSV
# =============================================================================

macd_list = []
with open('Data/macd.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            print(j)
            print(tech_analysis[j])
            macd_list.append([j, tech_analysis[j]]) #where j is the date and tech_analysis[j] is the data at date j

macd = []
for i in range(9, len(macd_list)):
    date = macd_list[i][0]
    macd_value = macd_list[i][1]['MACD']
    macd_signal = macd_list[i][1]['MACD_Signal']
    macd_hist = macd_list[i][1]['MACD_Hist']

    macd.append([date, macd_value, macd_signal, macd_hist])
    
macd = np.array(macd)

filepath = 'macd.csv'

macdDF = pd.DataFrame(macd)

macdDF.to_csv(filepath, index=False)

# =============================================================================
# Converting CCI JSON file to CSV
# =============================================================================

cci_list = []
with open('Data/cci.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            cci_list.append([j, tech_analysis[j]])

cci = []
for i in range(6, len(cci_list)):
    date = cci_list[i][0]
    cci_value = cci_list[i][1]['CCI']

    cci.append([date, cci_value])
    
cci = np.array(cci)

filepath = 'Data/cci.csv'

cciDF = pd.DataFrame(cci)

cciDF.to_csv(filepath, index=False)

# =============================================================================
# Converting ATR JSON file to CSV
# =============================================================================

atr_list = []
with open('Data/atr.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            atr_list.append([j, tech_analysis[j]])

atr = []
for i in range(6, len(atr_list)):
    date = atr_list[i][0]
    atr_value = atr_list[i][1]['ATR']

    atr.append([date, atr_value])
    
atr = np.array(atr)

filepath = 'atr.csv'

atrDF = pd.DataFrame(atr)

atrDF.to_csv(filepath, index=False)

# =============================================================================
# Converting BOLL JSON file to CSV
# =============================================================================

boll_list = []
with open('Data/boll.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            boll_list.append([j, tech_analysis[j]])

boll = []
for i in range(10, len(boll_list)):
    date = boll_list[i][0]
    boll_mid_band = boll_list[i][1]['Real Middle Band']
    boll_up_band = boll_list[i][1]['Real Upper Band']
    boll_lower_band = boll_list[i][1]['Real Lower Band']

    boll.append([date, boll_mid_band, boll_up_band, boll_lower_band])
    
boll = np.array(boll)

filepath = 'boll.csv'

bollDF = pd.DataFrame(boll)

bollDF.to_csv(filepath, index=False)

# =============================================================================
# Converting EMA20 JSON file to CSV
# =============================================================================

ema20_list = []
with open('Data/ema20.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            ema20_list.append([j, tech_analysis[j]])

ema20 = []
for i in range(7, len(ema20_list)):
    date = ema20_list[i][0]
    ema20_value = ema20_list[i][1]['EMA']

    ema20.append([date, ema20_value])
    
ema20 = np.array(ema20)

filepath = 'ema20.csv'

ema20DF = pd.DataFrame(ema20)

ema20DF.to_csv(filepath, index=False)

# =============================================================================
# Converting 5 Day Moving Average JSON file to CSV
# =============================================================================

ma5_list = []
with open('Data/ma5.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            ma5_list.append([j, tech_analysis[j]])

ma5 = []
for i in range(7, len(ma5_list)):
    date = ma5_list[i][0]
    ma5_value = ma5_list[i][1]['SMA']

    ma5.append([date, ma5_value])
    
ma5 = np.array(ma5)

filepath = 'ma5.csv'

ma5DF = pd.DataFrame(ma5)

ma5DF.to_csv(filepath, index=False)

# =============================================================================
# Converting 5 Day Moving Average JSON file to CSV
# =============================================================================

ma10_list = []
with open('Data/ma10.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            ma10_list.append([j, tech_analysis[j]])

ma10 = []
for i in range(7, len(ma10_list)):
    date = ma10_list[i][0]
    ma10_value = ma10_list[i][1]['SMA']

    ma10.append([date, ma10_value])
    
ma10 = np.array(ma10)

filepath = 'ma10.csv'

ma10DF = pd.DataFrame(ma10)

ma10DF.to_csv(filepath, index=False)

# =============================================================================
# Converting 6 month momentum JSON file to CSV
# =============================================================================

mom6_list = []
with open('Data/mom6.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            mom6_list.append([j, tech_analysis[j]])

mom6 = []
for i in range(7, len(mom6_list)):
    date = mom6_list[i][0]
    mom6_value = mom6_list[i][1]['MOM']

    mom6.append([date, mom6_value])
    
mom6 = np.array(mom6)

filepath = 'mom6.csv'

mom6DF = pd.DataFrame(mom6)

mom6DF.to_csv(filepath, index=False)

# =============================================================================
# Converting 10 month momentum JSON file to CSV
# =============================================================================

mom12_list = []
with open('Data/mom12.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            mom12_list.append([j, tech_analysis[j]])

mom12 = []
for i in range(7, len(mom12_list)):
    date = mom12_list[i][0]
    mom12_value = mom12_list[i][1]['MOM']

    mom12.append([date, mom12_value])
    
mom12 = np.array(mom12)

filepath = 'mom12.csv'

mom12DF = pd.DataFrame(mom12)

mom12DF.to_csv(filepath, index=False)

# =============================================================================
# Converting ROC JSON file to CSV
# =============================================================================

roc_list = []
with open('Data/roc.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            roc_list.append([j, tech_analysis[j]])

roc = []
for i in range(7, len(roc_list)):
    date = roc_list[i][0]
    roc_value = roc_list[i][1]['ROC']

    roc.append([date, roc_value])
    
roc = np.array(roc)

filepath = 'roc.csv'

rocDF = pd.DataFrame(roc)

rocDF.to_csv(filepath, index=False)

# =============================================================================
# Converting STOCH JSON file to CSV
# =============================================================================

stoch_list = []
with open('Data/stoch.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            stoch_list.append([j, tech_analysis[j]])

stoch = []
for i in range(10, len(stoch_list)):
    date = stoch_list[i][0]
    slowK = stoch_list[i][1]['SlowK']
    slowD = stoch_list[i][1]['SlowD']

    stoch.append([date, slowK, slowD])
    
stoch = np.array(stoch)

filepath = 'stoch.csv'

stochDF = pd.DataFrame(stoch)

stochDF.to_csv(filepath, index=False)

# =============================================================================
# Converting WVAD JSON file to CSV
# =============================================================================


wvad_list = []
with open('Data/wvad.json') as data_file:    
    data = json.load(data_file)
    for tech_analysis in data.values(): #.values()
        for j in tech_analysis:
            wvad_list.append([j, tech_analysis[j]])

wvad = []
for i in range(6, len(wvad_list)):
    date = wvad_list[i][0]
    willr = wvad_list[i][1]['WILLR']

    wvad.append([date, willr])
    
wvad = np.array(wvad)

filepath = 'Data/wvad.csv'

wvadDF = pd.DataFrame(wvad)

wvadDF.to_csv(filepath, index=False)