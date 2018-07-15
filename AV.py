#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 10:48:32 2018

@author: shaydineen
"""

import requests
import alpha_vantage


API_URL = "https://www.alphavantage.co/query"

data = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "^GSPC",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


outFile = 'Data/testData.csv'

page = requests.get(url = API_URL, params = data)

with open(outFile, 'w') as oF:
    oF.write(page.text)
    
macd = {
       'function' : 'MACD',
       'symbol' : '^GSPC',
       "outputsize" : "full",
       "datatype" : "csv",
       'interval' : 'daily',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

macd_request = requests.get(url = API_URL, params = macd)

macd_file = 'Data/macd.json' 
with open(macd_file, 'w') as oF:
    oF.write(macd_request.text)
    
#RSI and stochastic

cci = {
       'function' : 'CCI',
       'symbol' : '^GSPC',
       'interval' : 'daily',
       'time_period' : '30', #investopedia says this is a standard value
       'apikey' : '0ZSSUD2LJQV6MK6M'}

cci_request = requests.get(url = API_URL, params = cci)
cci_file = 'Data/cci.json'

with open(cci_file, 'w') as oF:
    oF.write(cci_request.text)
    
atr = {
       'function' : 'ATR',
       'symbol' : '^GSPC',
       'interval' : 'daily',
       'time_period' : '14', #investopedia says this is a standard value
       'apikey' : '0ZSSUD2LJQV6MK6M'}

atr_request = requests.get(url = API_URL, params = atr)
atr_file = 'Data/atr.json'

with open(atr_file, 'w') as oF:
    oF.write(atr_request.text)
    
boll = {
        'function' : 'BBANDS',
        'symbol' : '^GSPC',
        'interval' : 'daily',
        'time_period' : '20', #investopedia says this is a standard value
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

boll_request = requests.get(url = API_URL, params = boll)
boll_file = 'Data/boll.json'

with open(boll_file, 'w') as oF:
    oF.write(boll_request.text)
    
ema20 = {
        'function' : 'EMA',
        'symbol' : '^GSPC',
        'interval' : 'daily',
        'time_period' : '20', #what was used in actual paper
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

ema20_request = requests.get(url = API_URL, params = ema20)
ema20_file = 'Data/ema20.json'

with open(ema20_file, 'w') as oF:
    oF.write(ema20_request.text)
    
ma5 = {
       'function' : 'sma', 
       'symbol' : '^GSPC',
       'interval' : 'daily',
       'time_period' : '5', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma5_request = requests.get(url = API_URL, params = ma5)
ma5_file = 'Data/ma5.json'

with open(ma5_file, 'w') as oF:
    oF.write(ma5_request.text)
    
ma10 = {
       'function' : 'sma', 
       'symbol' : '^GSPC',
       'interval' : 'daily',
       'time_period' : '10', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma10_request = requests.get(url = API_URL, params = ma10)
ma10_file = 'Data/ma10.json'

with open(ma10_file, 'w') as oF:
    oF.write(ma10_request.text)

mom6 = {
        'function' : 'MOM',
        'symbol' : '^GSPC',
        'interval' : 'daily',
        'time_period' : '127', #they said they used six months and on average there are 252 trading days in a year so half of that is 127
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

mom6_request = requests.get(url = API_URL, params = mom6)
mom6_file = 'Data/mom6.json'

with open(mom6_file, 'w') as oF:
    oF.write(mom6_request.text)

mom12 = {
        'function' : 'MOM',
        'symbol' : '^GSPC',
        'interval' : 'daily',
        'time_period' : '252', #they said they used twelve months and on average there are 252 trading days in a year
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

mom12_request = requests.get(url = API_URL, params = mom12)
mom12_file = 'Data/mom12.json'

with open(mom12_file, 'w') as oF:
    oF.write(mom12_request.text)
    
roc = {
        'function' : 'ROC',
        'symbol' : '^GSPC',
        'interval' : 'daily',
        'time_period' : '127', #using the same value as six month momentum
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

roc_request = requests.get(url = API_URL, params = roc)
roc_file = 'Data/roc.json'

with open(roc_file, 'w') as oF:
    oF.write(roc_request.text)
    
wvad = {
        'function' : 'WILLR',
        'symbol' : '^GSPC',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

wvad_request = requests.get(url = API_URL, params = wvad)
wvad_file = 'Data/wvad.json'

with open(wvad_file, 'w') as oF:
    oF.write(wvad_request.text)

stoch = {
        'function' : 'STOCH',
        'symbol' : '^GSPC',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

stoch_request = requests.get(url = API_URL, params = stoch)
stoch_file = 'Data/stoch.json'

with open(stoch_file, 'w') as oF:
    oF.write(stoch_request.text)
    
us_dollar = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "DXY",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


usdx_file = 'Data/usdx.csv'

usdx_request = requests.get(url = API_URL, params = us_dollar)

with open(usdx_file, 'w') as oF:
    oF.write(usdx_request.text)
    
tnx = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "^TNX",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


tnx_file = 'Data/tnx.csv'

tnx_request = requests.get(url = API_URL, params = tnx)

with open(tnx_file, 'w') as oF:
    oF.write(tnx_request.text)

vix = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "VIX",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


vix_file = 'Data/VIX.csv'

vix_request = requests.get(url = API_URL, params = vix)

with open(vix_file, 'w') as oF:
    oF.write(vix_request.text)

rsi= {
       'function' : 'RSI',
       'symbol' : '^GSPC',
       'interval' : 'daily',
       'time_period' : '14',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

rsi_file = 'Data/rsi.json'
rsi_request = requests.get(url = API_URL, params = rsi)

with open(rsi_file, 'w') as oF:
    oF.write(rsi_request.text)
    
    
spy = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "SPY",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


spy_file = 'ETF Opportunity Set/spy.csv'

spy_request = requests.get(url = API_URL, params = spy)

with open(spy_file, 'w') as oF:
    oF.write(spy_request.text)

stoch_spy = {
        'function' : 'STOCH',
        'symbol' : 'SPY',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

stoch_spy_request = requests.get(url = API_URL, params = stoch_spy)
stoch_spy_file = 'ETF Opportunity Set/SPY/stoch_spy.json'

with open(stoch_spy_file, 'w') as oF:
    oF.write(stoch_spy_request.text)

ema20_spy = {
        'function' : 'EMA',
        'symbol' : 'SPY',
        'interval' : 'daily',
        'time_period' : '20', #what was used in actual paper
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

ema20_spy_request = requests.get(url = API_URL, params = ema20_spy)
ema20_spy_file = 'ETF Opportunity Set/SPY/ema20_spy.json'

with open(ema20_spy_file, 'w') as oF:
    oF.write(ema20_spy_request.text)

ma5_spy = {
       'function' : 'sma', 
       'symbol' : 'SPY',
       'interval' : 'daily',
       'time_period' : '5', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma5_spy_request = requests.get(url = API_URL, params = ma5_spy)
ma5_spy_file = 'ETF_Opportunity_Set/SPY/ma5_spy.json'

with open(ma5_spy_file, 'w') as oF:
    oF.write(ma5_spy_request.text)
    
ma10_spy = {
       'function' : 'sma', 
       'symbol' : 'SPY',
       'interval' : 'daily',
       'time_period' : '10', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma10_spy_request = requests.get(url = API_URL, params = ma10_spy)
ma10_spy_file = 'ETF_Opportunity_Set/SPY/ma10_spy.json'

with open(ma10_spy_file, 'w') as oF:
    oF.write(ma10_spy_request.text)

wvad_spy = {
        'function' : 'WILLR',
        'symbol' : 'SPY',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

wvad_spy_request = requests.get(url = API_URL, params = wvad_spy)
wvad_spy_file = 'ETF_Opportunity_Set/SPY/wvad_spy.json'

with open(wvad_spy_file, 'w') as oF:
    oF.write(wvad_spy_request.text)


rsi_spy= {
       'function' : 'RSI',
       'symbol' : 'SPY',
       'interval' : 'daily',
       'time_period' : '14',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

rsi_spy_file = 'ETF_Opportunity_Set/SPY/rsi_spy.json'
rsi_spy_request = requests.get(url = API_URL, params = rsi_spy)

with open(rsi_spy_file, 'w') as oF:
    oF.write(rsi_spy_request.text)

#ma5 ma10 and RSI
    
# =============================================================================
# IWM
# =============================================================================
iwm = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "IWM",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


iwm_file = 'ETF_Opportunity_Set/IWN/iwn.csv'

iwn_request = requests.get(url = API_URL, params = iwm)

with open(iwm_file, 'w') as oF:
    oF.write(iwn_request.text)

stoch_iwm = {
        'function' : 'STOCH',
        'symbol' : 'IWM',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

stoch_iwm_request = requests.get(url = API_URL, params = stoch_iwm)
stoch_iwm_file = 'ETF_Opportunity_Set/IWN/stoch_iwn.json'

with open(stoch_iwm_file, 'w') as oF:
    oF.write(stoch_iwm_request.text)

ema20_iwm = {
        'function' : 'EMA',
        'symbol' : 'IWM',
        'interval' : 'daily',
        'time_period' : '20', #what was used in actual paper
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

ema20_iwm_request = requests.get(url = API_URL, params = ema20_iwm)
ema20_iwm_file = 'ETF_Opportunity_Set/IWN/ema20_iwm.json'

with open(ema20_iwm_file, 'w') as oF:
    oF.write(ema20_iwm_request.text)

ma5_iwm = {
       'function' : 'sma', 
       'symbol' : 'IWM',
       'interval' : 'daily',
       'time_period' : '5', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma5_iwm_request = requests.get(url = API_URL, params = ma5_iwm)
ma5_iwm_file = 'ETF_Opportunity_Set/IWN/ma5_iwm.json'

with open(ma5_iwm_file, 'w') as oF:
    oF.write(ma5_iwm_request.text)
    
ma10_iwm = {
       'function' : 'sma', 
       'symbol' : 'IWM',
       'interval' : 'daily',
       'time_period' : '10', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma10_iwm_request = requests.get(url = API_URL, params = ma10_iwm)
ma10_iwm_file = 'ETF_Opportunity_Set/IWN/ma10_iwm.json'

with open(ma10_iwm_file, 'w') as oF:
    oF.write(ma10_iwm_request.text)

wvad_iwm = {
        'function' : 'WILLR',
        'symbol' : 'IWM',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

wvad_iwm_request = requests.get(url = API_URL, params = wvad_iwm)
wvad_iwm_file = 'ETF_Opportunity_Set/IWN/wvad_iwm.json'

with open(wvad_iwm_file, 'w') as oF:
    oF.write(wvad_iwm_request.text)


rsi_iwm= {
       'function' : 'RSI',
       'symbol' : 'IWM',
       'interval' : 'daily',
       'time_period' : '14',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

rsi_iwm_file = 'ETF_Opportunity_Set/IWN/rsi_iwm.json'
rsi_iwm_request = requests.get(url = API_URL, params = rsi_iwm)

with open(rsi_iwm_file, 'w') as oF:
    oF.write(rsi_iwm_request.text)


# =============================================================================
# EEM
# =============================================================================

eem = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "EEM",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


eem_file = 'ETF_Opportunity_Set/EEM/eem.csv'

eem_request = requests.get(url = API_URL, params = eem)

with open(eem_file, 'w') as oF:
    oF.write(eem_request.text)

stoch_eem = {
        'function' : 'STOCH',
        'symbol' : 'EEM',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

stoch_eem_request = requests.get(url = API_URL, params = stoch_eem)
stoch_eem_file = 'ETF_Opportunity_Set/EEM/stoch_eem.json'

with open(stoch_eem_file, 'w') as oF:
    oF.write(stoch_eem_request.text)

ema20_eem = {
        'function' : 'EMA',
        'symbol' : 'EEM',
        'interval' : 'daily',
        'time_period' : '20', #what was used in actual paper
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

ema20_eem_request = requests.get(url = API_URL, params = ema20_eem)
ema20_eem_file = 'ETF_Opportunity_Set/EEM/ema20_eem.json'

with open(ema20_eem_file, 'w') as oF:
    oF.write(ema20_eem_request.text)

ma5_eem = {
       'function' : 'sma', 
       'symbol' : 'EEM',
       'interval' : 'daily',
       'time_period' : '5', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma5_eem_request = requests.get(url = API_URL, params = ma5_eem)
ma5_eem_file = 'ETF_Opportunity_Set/EEM/ma5_eem.json'

with open(ma5_eem_file, 'w') as oF:
    oF.write(ma5_eem_request.text)
    
ma10_eem = {
       'function' : 'sma', 
       'symbol' : 'EEM',
       'interval' : 'daily',
       'time_period' : '10', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma10_eem_request = requests.get(url = API_URL, params = ma10_eem)
ma10_eem_file = 'ETF_Opportunity_Set/EEM/ma10_eem.json'

with open(ma10_eem_file, 'w') as oF:
    oF.write(ma10_eem_request.text)

wvad_eem = {
        'function' : 'WILLR',
        'symbol' : 'EEM',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

wvad_eem_request = requests.get(url = API_URL, params = wvad_eem)
wvad_eem_file = 'ETF_Opportunity_Set/EEM/wvad_eem.json'

with open(wvad_eem_file, 'w') as oF:
    oF.write(wvad_eem_request.text)


rsi_eem= {
       'function' : 'RSI',
       'symbol' : 'EEM',
       'interval' : 'daily',
       'time_period' : '14',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

rsi_eem_file = 'ETF_Opportunity_Set/EEM/rsi_eem.json'
rsi_eem_request = requests.get(url = API_URL, params = rsi_eem)

with open(rsi_eem_file, 'w') as oF:
    oF.write(rsi_eem_request.text)

# =============================================================================
# TLT
# =============================================================================

tlt = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "TLT",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


tlt_file = 'ETF_Opportunity_Set/TLT/tlt.csv'

tlt_request = requests.get(url = API_URL, params = tlt)

with open(tlt_file, 'w') as oF:
    oF.write(tlt_request.text)

stoch_tlt = {
        'function' : 'STOCH',
        'symbol' : 'TLT',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

stoch_tlt_request = requests.get(url = API_URL, params = stoch_tlt)
stoch_tlt_file = 'ETF_Opportunity_Set/TLT/stoch_tlt.json'

with open(stoch_tlt_file, 'w') as oF:
    oF.write(stoch_tlt_request.text)

ema20_tlt = {
        'function' : 'EMA',
        'symbol' : 'TLT',
        'interval' : 'daily',
        'time_period' : '20', #what was used in actual paper
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

ema20_tlt_request = requests.get(url = API_URL, params = ema20_tlt)
ema20_tlt_file = 'ETF_Opportunity_Set/TLT/ema20_tlt.json'

with open(ema20_tlt_file, 'w') as oF:
    oF.write(ema20_tlt_request.text)

ma5_tlt = {
       'function' : 'sma', 
       'symbol' : 'TLT',
       'interval' : 'daily',
       'time_period' : '5', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma5_tlt_request = requests.get(url = API_URL, params = ma5_tlt)
ma5_tlt_file = 'ETF_Opportunity_Set/TLT/ma5_tlt.json'

with open(ma5_tlt_file, 'w') as oF:
    oF.write(ma5_tlt_request.text)
    
ma10_tlt = {
       'function' : 'sma', 
       'symbol' : 'TLT',
       'interval' : 'daily',
       'time_period' : '10', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma10_tlt_request = requests.get(url = API_URL, params = ma10_tlt)
ma10_tlt_file = 'ETF_Opportunity_Set/TLT/ma10_tlt.json'

with open(ma10_tlt_file, 'w') as oF:
    oF.write(ma10_tlt_request.text)

wvad_tlt = {
        'function' : 'WILLR',
        'symbol' : 'TLT',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

wvad_tlt_request = requests.get(url = API_URL, params = wvad_tlt)
wvad_tlt_file = 'ETF_Opportunity_Set/TLT/wvad_tlt.json'

with open(wvad_tlt_file, 'w') as oF:
    oF.write(wvad_tlt_request.text)


rsi_tlt= {
       'function' : 'RSI',
       'symbol' : 'TLT',
       'interval' : 'daily',
       'time_period' : '14',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

rsi_tlt_file = 'ETF_Opportunity_Set/TLT/rsi_tlt.json'
rsi_tlt_request = requests.get(url = API_URL, params = rsi_tlt)

with open(rsi_tlt_file, 'w') as oF:
    oF.write(rsi_tlt_request.text)

# =============================================================================
# LQD
# =============================================================================

lqd = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "LQD",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


lqd_file = 'ETF_Opportunity_Set/LQD/lqd.csv'

lqd_request = requests.get(url = API_URL, params = lqd)

with open(lqd_file, 'w') as oF:
    oF.write(lqd_request.text)

stoch_lqd = {
        'function' : 'STOCH',
        'symbol' : 'LQD',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

stoch_lqd_request = requests.get(url = API_URL, params = stoch_lqd)
stoch_lqd_file = 'ETF_Opportunity_Set/LQD/stoch_lqd.json'

with open(stoch_lqd_file, 'w') as oF:
    oF.write(stoch_lqd_request.text)

ema20_lqd = {
        'function' : 'EMA',
        'symbol' : 'LQD',
        'interval' : 'daily',
        'time_period' : '20', #what was used in actual paper
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

ema20_lqd_request = requests.get(url = API_URL, params = ema20_lqd)
ema20_lqd_file = 'ETF_Opportunity_Set/LQD/ema20_lqd.json'

with open(ema20_lqd_file, 'w') as oF:
    oF.write(ema20_lqd_request.text)

ma5_lqd = {
       'function' : 'sma', 
       'symbol' : 'LQD',
       'interval' : 'daily',
       'time_period' : '5', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma5_lqd_request = requests.get(url = API_URL, params = ma5_lqd)
ma5_lqd_file = 'ETF_Opportunity_Set/LQD/ma5_lqd.json'

with open(ma5_lqd_file, 'w') as oF:
    oF.write(ma5_lqd_request.text)
    
ma10_lqd = {
       'function' : 'sma', 
       'symbol' : 'LQD',
       'interval' : 'daily',
       'time_period' : '10', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma10_lqd_request = requests.get(url = API_URL, params = ma10_lqd)
ma10_lqd_file = 'ETF_Opportunity_Set/LQD/ma10_lqd.json'

with open(ma10_lqd_file, 'w') as oF:
    oF.write(ma10_lqd_request.text)

wvad_lqd = {
        'function' : 'WILLR',
        'symbol' : 'LQD',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

wvad_lqd_request = requests.get(url = API_URL, params = wvad_lqd)
wvad_lqd_file = 'ETF_Opportunity_Set/LQD/wvad_lqd.json'

with open(wvad_lqd_file, 'w') as oF:
    oF.write(wvad_lqd_request.text)


rsi_lqd= {
       'function' : 'RSI',
       'symbol' : 'LQD',
       'interval' : 'daily',
       'time_period' : '14',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

rsi_lqd_file = 'ETF_Opportunity_Set/LQD/rsi_lqd.json'
rsi_lqd_request = requests.get(url = API_URL, params = rsi_lqd)

with open(rsi_lqd_file, 'w') as oF:
    oF.write(rsi_lqd_request.text)
    
# =============================================================================
# TIP
# =============================================================================

tip = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "TIP",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


tip_file = 'ETF_Opportunity_Set/TIP/tip.csv'

tip_request = requests.get(url = API_URL, params = tip)

with open(tip_file, 'w') as oF:
    oF.write(tip_request.text)

stoch_tip = {
        'function' : 'STOCH',
        'symbol' : 'TIP',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

stoch_tip_request = requests.get(url = API_URL, params = stoch_tip)
stoch_tip_file = 'ETF_Opportunity_Set/TIP/stoch_tip.json'

with open(stoch_tip_file, 'w') as oF:
    oF.write(stoch_tip_request.text)

ema20_tip = {
        'function' : 'EMA',
        'symbol' : 'TIP',
        'interval' : 'daily',
        'time_period' : '20', #what was used in actual paper
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

ema20_tip_request = requests.get(url = API_URL, params = ema20_tip)
ema20_tip_file = 'ETF_Opportunity_Set/TIP/ema20_tip.json'

with open(ema20_tip_file, 'w') as oF:
    oF.write(ema20_tip_request.text)

ma5_tip = {
       'function' : 'sma', 
       'symbol' : 'TIP',
       'interval' : 'daily',
       'time_period' : '5', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma5_tip_request = requests.get(url = API_URL, params = ma5_tip)
ma5_tip_file = 'ETF_Opportunity_Set/TIP/ma5_tip.json'

with open(ma5_tip_file, 'w') as oF:
    oF.write(ma5_tip_request.text)
    
ma10_tip = {
       'function' : 'sma', 
       'symbol' : 'TIP',
       'interval' : 'daily',
       'time_period' : '10', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma10_tip_request = requests.get(url = API_URL, params = ma10_tip)
ma10_tip_file = 'ETF_Opportunity_Set/TIP/ma10_tip.json'

with open(ma10_tip_file, 'w') as oF:
    oF.write(ma10_tip_request.text)

wvad_tip = {
        'function' : 'WILLR',
        'symbol' : 'TIP',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

wvad_tip_request = requests.get(url = API_URL, params = wvad_tip)
wvad_tip_file = 'ETF_Opportunity_Set/TIP/wvad_tip.json'

with open(wvad_tip_file, 'w') as oF:
    oF.write(wvad_tip_request.text)


rsi_tip= {
       'function' : 'RSI',
       'symbol' : 'TIP',
       'interval' : 'daily',
       'time_period' : '14',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

rsi_tip_file = 'ETF_Opportunity_Set/TIP/rsi_tip.json'
rsi_tip_request = requests.get(url = API_URL, params = rsi_tip)

with open(rsi_tip_file, 'w') as oF:
    oF.write(rsi_tip_request.text)

# =============================================================================
# IYR
# =============================================================================

iyr = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "IYR",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


iyr_file = 'ETF_Opportunity_Set/IYR/iyr.csv'

iyr_request = requests.get(url = API_URL, params = iyr)

with open(iyr_file, 'w') as oF:
    oF.write(iyr_request.text)

stoch_iyr = {
        'function' : 'STOCH',
        'symbol' : 'IYR',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

stoch_iyr_request = requests.get(url = API_URL, params = stoch_iyr)
stoch_iyr_file = 'ETF_Opportunity_Set/IYR/stoch_iyr.json'

with open(stoch_iyr_file, 'w') as oF:
    oF.write(stoch_iyr_request.text)

ema20_iyr = {
        'function' : 'EMA',
        'symbol' : 'IYR',
        'interval' : 'daily',
        'time_period' : '20', #what was used in actual paper
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

ema20_iyr_request = requests.get(url = API_URL, params = ema20_iyr)
ema20_iyr_file = 'ETF_Opportunity_Set/IYR/ema20_iyr.json'

with open(ema20_iyr_file, 'w') as oF:
    oF.write(ema20_iyr_request.text)

ma5_iyr = {
       'function' : 'sma', 
       'symbol' : 'IYR',
       'interval' : 'daily',
       'time_period' : '5', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma5_iyr_request = requests.get(url = API_URL, params = ma5_iyr)
ma5_iyr_file = 'ETF_Opportunity_Set/IYR/ma5_iyr.json'

with open(ma5_iyr_file, 'w') as oF:
    oF.write(ma5_iyr_request.text)
    
ma10_iyr = {
       'function' : 'sma', 
       'symbol' : 'IYR',
       'interval' : 'daily',
       'time_period' : '10', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma10_iyr_request = requests.get(url = API_URL, params = ma10_iyr)
ma10_iyr_file = 'ETF_Opportunity_Set/IYR/ma10_iyr.json'

with open(ma10_iyr_file, 'w') as oF:
    oF.write(ma10_iyr_request.text)

wvad_iyr = {
        'function' : 'WILLR',
        'symbol' : 'IYR',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

wvad_iyr_request = requests.get(url = API_URL, params = wvad_iyr)
wvad_iyr_file = 'ETF_Opportunity_Set/IYR/wvad_iyr.json'

with open(wvad_iyr_file, 'w') as oF:
    oF.write(wvad_iyr_request.text)


rsi_iyr= {
       'function' : 'RSI',
       'symbol' : 'IYR',
       'interval' : 'daily',
       'time_period' : '14',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

rsi_iyr_file = 'ETF_Opportunity_Set/IYR/rsi_iyr.json'
rsi_iyr_request = requests.get(url = API_URL, params = rsi_iyr)

with open(rsi_iyr_file, 'w') as oF:
    oF.write(rsi_iyr_request.text)

# =============================================================================
# GLD
# =============================================================================

gld = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "GLD",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


gld_file = 'ETF_Opportunity_Set/GLD/gld.csv'

gld_request = requests.get(url = API_URL, params = gld)

with open(gld_file, 'w') as oF:
    oF.write(gld_request.text)

stoch_gld = {
        'function' : 'STOCH',
        'symbol' : 'GLD',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

stoch_gld_request = requests.get(url = API_URL, params = stoch_gld)
stoch_gld_file = 'ETF_Opportunity_Set/GLD/stoch_gld.json'

with open(stoch_gld_file, 'w') as oF:
    oF.write(stoch_gld_request.text)

ema20_gld = {
        'function' : 'EMA',
        'symbol' : 'GLD',
        'interval' : 'daily',
        'time_period' : '20', #what was used in actual paper
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

ema20_gld_request = requests.get(url = API_URL, params = ema20_gld)
ema20_gld_file = 'ETF_Opportunity_Set/GLD/ema20_gld.json'

with open(ema20_gld_file, 'w') as oF:
    oF.write(ema20_gld_request.text)

ma5_gld = {
       'function' : 'sma', 
       'symbol' : 'GLD',
       'interval' : 'daily',
       'time_period' : '5', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma5_gld_request = requests.get(url = API_URL, params = ma5_gld)
ma5_gld_file = 'ETF_Opportunity_Set/GLD/ma5_gld.json'

with open(ma5_gld_file, 'w') as oF:
    oF.write(ma5_gld_request.text)
    
ma10_gld = {
       'function' : 'sma', 
       'symbol' : 'GLD',
       'interval' : 'daily',
       'time_period' : '10', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma10_gld_request = requests.get(url = API_URL, params = ma10_gld)
ma10_gld_file = 'ETF_Opportunity_Set/GLD/ma10_gld.json'

with open(ma10_gld_file, 'w') as oF:
    oF.write(ma10_gld_request.text)

wvad_gld = {
        'function' : 'WILLR',
        'symbol' : 'GLD',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

wvad_gld_request = requests.get(url = API_URL, params = wvad_gld)
wvad_gld_file = 'ETF_Opportunity_Set/GLD/wvad_gld.json'

with open(wvad_gld_file, 'w') as oF:
    oF.write(wvad_gld_request.text)


rsi_gld = {
       'function' : 'RSI',
       'symbol' : 'GLD',
       'interval' : 'daily',
       'time_period' : '14',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

rsi_gld_file = 'ETF_Opportunity_Set/GLD/rsi_gld.json'
rsi_gld_request = requests.get(url = API_URL, params = rsi_gld)

with open(rsi_gld_file, 'w') as oF:
    oF.write(rsi_gld_request.text)

# =============================================================================
# OIH
# =============================================================================

oih = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "OIH",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


oih_file = 'ETF_Opportunity_Set/OIH/oih.csv'

oih_request = requests.get(url = API_URL, params = oih)

with open(oih_file, 'w') as oF:
    oF.write(oih_request.text)

stoch_oih = {
        'function' : 'STOCH',
        'symbol' : 'OIH',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

stoch_oih_request = requests.get(url = API_URL, params = stoch_oih)
stoch_oih_file = 'ETF_Opportunity_Set/OIH/stoch_oih.json'

with open(stoch_oih_file, 'w') as oF:
    oF.write(stoch_oih_request.text)

ema20_oih = {
        'function' : 'EMA',
        'symbol' : 'OIH',
        'interval' : 'daily',
        'time_period' : '20', #what was used in actual paper
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

ema20_oih_request = requests.get(url = API_URL, params = ema20_oih)
ema20_oih_file = 'ETF_Opportunity_Set/OIH/ema20_oih.json'

with open(ema20_oih_file, 'w') as oF:
    oF.write(ema20_oih_request.text)

ma5_oih = {
       'function' : 'sma', 
       'symbol' : 'OIH',
       'interval' : 'daily',
       'time_period' : '5', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma5_oih_request = requests.get(url = API_URL, params = ma5_oih)
ma5_oih_file = 'ETF_Opportunity_Set/OIH/ma5_oih.json'

with open(ma5_oih_file, 'w') as oF:
    oF.write(ma5_oih_request.text)
    
ma10_oih = {
       'function' : 'sma', 
       'symbol' : 'OIH',
       'interval' : 'daily',
       'time_period' : '10', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma10_oih_request = requests.get(url = API_URL, params = ma10_oih)
ma10_oih_file = 'ETF_Opportunity_Set/OIH/ma10_oih.json'

with open(ma10_oih_file, 'w') as oF:
    oF.write(ma10_oih_request.text)

wvad_oih = {
        'function' : 'WILLR',
        'symbol' : 'OIH',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

wvad_oih_request = requests.get(url = API_URL, params = wvad_oih)
wvad_oih_file = 'ETF_Opportunity_Set/OIH/wvad_oih.json'

with open(wvad_oih_file, 'w') as oF:
    oF.write(wvad_oih_request.text)


rsi_oih= {
       'function' : 'RSI',
       'symbol' : 'OIH',
       'interval' : 'daily',
       'time_period' : '14',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

rsi_oih_file = 'ETF_Opportunity_Set/OIH/rsi_oih.json'
rsi_oih_request = requests.get(url = API_URL, params = rsi_oih)

with open(rsi_oih_file, 'w') as oF:
    oF.write(rsi_oih_request.text)

# =============================================================================
# FXE
# =============================================================================

fxe = {
    "function" : "TIME_SERIES_DAILY",
    "symbol" : "FXE",
    "outputsize" : "full",
    "datatype" : "csv",
    "apikey" : "0ZSSUD2LJQV6MK6M",
    }


fxe_file = 'ETF_Opportunity_Set/FXE/fxe.csv'

fxe_request = requests.get(url = API_URL, params = fxe)

with open(fxe_file, 'w') as oF:
    oF.write(fxe_request.text)

stoch_fxe = {
        'function' : 'STOCH',
        'symbol' : 'FXE',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

stoch_fxe_request = requests.get(url = API_URL, params = stoch_fxe)
stoch_fxe_file = 'ETF_Opportunity_Set/FXE/stoch_fxe.json'

with open(stoch_fxe_file, 'w') as oF:
    oF.write(stoch_fxe_request.text)

ema20_fxe = {
        'function' : 'EMA',
        'symbol' : 'FXE',
        'interval' : 'daily',
        'time_period' : '20', #what was used in actual paper
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

ema20_fxe_request = requests.get(url = API_URL, params = ema20_fxe)
ema20_fxe_file = 'ETF_Opportunity_Set/FXE/ema20_fxe.json'

with open(ema20_fxe_file, 'w') as oF:
    oF.write(ema20_fxe_request.text)

ma5_fxe = {
       'function' : 'sma', 
       'symbol' : 'FXE',
       'interval' : 'daily',
       'time_period' : '5', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma5_fxe_request = requests.get(url = API_URL, params = ma5_fxe)
ma5_fxe_file = 'ETF_Opportunity_Set/FXE/ma5_fxe.json'

with open(ma5_fxe_file, 'w') as oF:
    oF.write(ma5_fxe_request.text)
    
ma10_fxe = {
       'function' : 'sma', 
       'symbol' : 'FXE',
       'interval' : 'daily',
       'time_period' : '10', #what was used in actual paper
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

ma10_fxe_request = requests.get(url = API_URL, params = ma10_fxe)
ma10_fxe_file = 'ETF_Opportunity_Set/FXE/ma10_fxe.json'

with open(ma10_fxe_file, 'w') as oF:
    oF.write(ma10_fxe_request.text)

wvad_fxe = {
        'function' : 'WILLR',
        'symbol' : 'FXE',
        'interval' : 'daily',
        'time_period' : '14', #according to investopedia, 14 is usually used
        'series_type' : 'close',
        'apikey' : '0ZSSUD2LJQV6MK6M'}

wvad_fxe_request = requests.get(url = API_URL, params = wvad_fxe)
wvad_fxe_file = 'ETF_Opportunity_Set/FXE/wvad_fxe.json'

with open(wvad_fxe_file, 'w') as oF:
    oF.write(wvad_fxe_request.text)


rsi_fxe= {
       'function' : 'RSI',
       'symbol' : 'FXE',
       'interval' : 'daily',
       'time_period' : '14',
       'series_type' : 'close',
       'apikey' : '0ZSSUD2LJQV6MK6M'}

rsi_fxe_file = 'ETF_Opportunity_Set/FXE/rsi_fxe.json'
rsi_fxe_request = requests.get(url = API_URL, params = rsi_fxe)

with open(rsi_fxe_file, 'w') as oF:
    oF.write(rsi_fxe_request.text)