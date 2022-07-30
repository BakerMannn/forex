############################################################################
"""Packages Import"""
#Data Manipulation/Import
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
import openpyxl
#Data Visulization
import plotly.express as px
import seaborn as sns  
import matplotlib.pyplot as plt
#Modeling
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

############################################################################
"""Global Variables"""
ticker = 'EURUSD=X'

############################################################################
"""Data Import"""
yf_data = yf.Ticker(ticker)
price_data = yf_data.history(period='max')

############################################################################
"""Data Cleaning/Manipulation"""
#Remove Uncessary Columns
price_data.drop(['Volume','Dividends','Stock Splits'], axis=1, inplace=True)

#Target Column
price_data['close_offset'] = price_data['Close'].shift(-1)
price_data['close_higher'] = price_data['close_offset'] > price_data['Close']
############################################################################
"""Feature Engineering"""
#TA Variables'''
close = price_data['Close']
open = price_data['Open']
high = price_data['High']
low = price_data['Low']

#Overlap Studies
price_data['EMA_21']=ta.EMA(close,timeperiod=21)
price_data['EMA_50']=ta.EMA(close,timeperiod=50)
price_data['EMA_100']=ta.EMA(close,timeperiod=100)
price_data['EMA_200']=ta.EMA(close,timeperiod=200)
price_data['SAR'] = ta.SAR(high,low)

#Momentum Indicators
price_data['ADX'] = ta.ADX(high,low,close,timeperiod=14)
price_data['AROON'] = ta.AROONOSC(high,low,timeperiod=14)
price_data['CCI'] = ta.CCI(high,low,close,timeperiod=14)
price_data['ROC'] = ta.ROC(close,timeperiod=10)
price_data['RSI'] = ta.RSI(close,timeperiod=14)
stoch_list = ta.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
price_data['STOCH_K']=stoch_list[0]
price_data['STOCH_D']=stoch_list[1]
stoch_RSI_list = ta.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
price_data['STOCHRSI_K']=stoch_RSI_list[0]
price_data['STOCHRSI_D']=stoch_RSI_list[1]

#Volatility
price_data['ATR'] = ta.ATR(high,low,close,timeperiod=14)
price_data['NATR'] = ta.NATR(high,low,close,timeperiod=14)

#Candle Recognition
price_data['THREEBLACKCROWS'] = ta.CDL3BLACKCROWS(open,high,low,close)
price_data['BREAKAWAY'] = ta.CDLBREAKAWAY(open,high,low,close)
price_data['DOJI'] = ta.CDLDOJI(open,high,low,close)
price_data['DOJISTAR'] = ta.CDLDOJISTAR(open,high,low,close)
price_data['EVENINGDOJISTAR'] = ta.CDLEVENINGSTAR(open,high,low,close,penetration = 0)
price_data['ECENINGSTAR'] = ta.CDLEVENINGDOJISTAR(open,high,low,close,penetration = 0)
price_data['GRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(open,high,low,close)
price_data['HAMMER'] = ta.CDLHAMMER(open,high,low,close)
price_data['HANGINGMAN'] = ta.CDLHANGINGMAN(open,high,low,close)
price_data['HARAMI'] = ta.CDLHARAMI(open,high,low,close)
price_data['HARAMICROSS'] = ta.CDLHARAMICROSS(open,high,low,close)
price_data['INVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(open,high,low,close)
price_data['MORINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(open,high,low,close,penetration = 0)
price_data['MORNINGSTAR'] = ta.CDLMORNINGSTAR(open,high,low,close,penetration = 0)
price_data['SHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(open,high,low,close)
price_data['SPINNINGTOP'] = ta.CDLSPINNINGTOP(open,high,low,close)

#Remove N/As
price_data.dropna(inplace=True)
price_data.to_excel('price_data.xlsx')

############################################################################
"""Model Training"""
#Pipeline
pipeline = make_pipeline(SimpleImputer(),
                         StandardScaler(),
                         PCA(),
                         GradientBoostingRegressor()
                         )

#Model Train

############################################################################
"""Model Evaluation"""


