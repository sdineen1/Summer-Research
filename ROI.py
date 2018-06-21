#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 08:43:11 2018

@author: shaydineen
"""

import pandas as pd
import numpy as np
import math as math

#still needs to see if we were wrong
def ROI(predicted, actual):
    
    portfolio = 100000
    stock_held = 0
    money_lost = 0
    for i in range (0, len(actual)):
        
        if predicted[i+1]>actual[i]:
            if portfolio/actual[i]>=1:
                stock_held = stock_held + math.floor(portfolio/actual[i])
                portfolio = portfolio%actual[i]
                #if (actual[i+1]<actual[i]):
        elif predicted[i+1]<actual[i]:
            if stock_held >= 0:
                portfolio = portfolio + actual[i]*stock_held
               stock_held = 0
    if stock_held>0:
        portfolio = portfolio + stock_held* actual[len(actual)-1]
    ROI = portfolio/100000
    return ROI, 

def ROI_pre_to_pre(predicted, actual):
    
    portfolio = 100000
    stock_held = 0
    for i in range (0, len(actual)):
        
        if predicted[i+1]>actual[i]:
            if portfolio/actual[i]>=1:
                stock_held = stock_held + math.floor(portfolio/actual[i])
                portfolio = portfolio%actual[i]
        elif predicted[i+1]<actual[i]:
            if stock_held >= 0:
                portfolio = portfolio + actual[i]*stock_held
                stock_held = 0
    ROI = portfolio/100000
    return ROI
