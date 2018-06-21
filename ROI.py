#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 08:43:11 2018

@author: shaydineen
"""

import pandas as pd
import numpy as np
import math as math

def ROI(predicted, actual):
    
    #Portfolio is an arbitrary amount of money
    portfolio = 100000
    #Stock held will be the variable that holds the number of shares of the S&P 500 that we hold
    stock_held = 0
    #For-loop is used to interate through each trading day in the test set
    for i in range (0, len(actual)-1):
        
        #This if- statement says that if the predicted price for tomorrow is greater than the price today, we enter the body of the if-statment
        if predicted[i+1]>actual[i]:
            #This if statement ensures that we have money to actually buy shares of the S&P 500.  If the portfolio (the amount of cash we have on hand) divided by the price of today's S&P 500 price is less than one that means we dont have enough cash on hand to buy shares 
            #If portfolio divided by day i's closing price is greater than 1, that means we have enough money to buy shares
            if portfolio/actual[i]>=1:
                #since we are buying, the number of shares that we have will go up
                #stock_held will go up by the number of shares that we already have plus the floor function (rounding down) of the portfolio divided by the closing price at day i
                stock_held = stock_held + math.floor(portfolio/actual[i])
                #since we took money out of the portfolio, the amount of money that we will now have in it is the remainder between portfolio and the price of share
                portfolio = portfolio%actual[i]
        
        #this else-if statement says that if the predicted price for tomorrow is less than today's price we will enter the loop
        elif predicted[i+1]<actual[i]:
            #this if statment ensures that we have shares to actually sell
            if stock_held > 0:
                #portfolio now becomes whatever we had in our portfolio before plus the price today that we are selling at times the number of shares
                portfolio = portfolio + actual[i]*stock_held
                #since we sell all of our shares, stock_held gets reset to 0
                stock_held = 0
    
    #This if statement is used to cash in all of our shares after after all of the test trading days
    if stock_held>0:
        #This statement says that the portfolio will be increased by the number of shares times the closing price for the last trading day
        portfolio = portfolio + stock_held* actual[len(actual)-1]
        #Since we cashed in all of our stocks, stock held is set back to 0
        stock_held = 0
    
    #ROI is simply calculated as the amount of money we have in our portfolio divided by our intial investment of 100000
    ROI = portfolio/100000
    return ROI, portfolio


def ROI_MAX(actual):
#The syntax is exactly the same as above but this calculates the max ROI that could be achieved if we had a special oracle telling us the exact stock price tomorrow 
    portfolio = 100000
    stock_held = 0
    for i in range (0, len(actual)-1):
        
        if actual[i+1]>actual[i]:
            if portfolio/actual[i]>=1:
                stock_held = stock_held + math.floor(portfolio/actual[i])
                portfolio = portfolio%actual[i]
        elif actual[i+1]<actual[i]:
            if stock_held > 0:
                portfolio = portfolio + actual[i]*stock_held
                stock_held = 0
    if stock_held>0:
        portfolio = portfolio + stock_held* actual[len(actual)-1]
        stock_held=0
    ROI = portfolio/100000
    return ROI


output = pd.read_csv('LSTMoutput.csv')
predicted = output.iloc[:,0].values
actual = output.iloc[:,1].values

predicted, actual = np.array(predicted), np.array(actual)

roi, portfolio = ROI(predicted=predicted, actual = actual)
max_roi = ROI_MAX(actual=actual)
