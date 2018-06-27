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

response = requests.get(API_URL, params=data)
print(response)
