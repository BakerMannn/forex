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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

############################################################################
"""Ticker Archive"""
EUR/USD = 'EURUSD=X'
Bitcoin  = 'BTC-USD'
S&P = 'SPY'

############################################################################
"""Global Variables"""
ticker = S&P
drop_labels = ['close_offset', 'close_higher']
label = 'close_higher'

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
"""Data Split"""
#Split Data
X = price_data.drop(columns=drop_labels)
y = price_data[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=555)

############################################################################
"""Base Models Training"""
#Instantiate Models
xgb_clf = xgb.XGBClassifier()
rf_clf = RandomForestClassifier()

#XGB Grid/Train
xgb_pipeline = Pipeline([
                    ('imputer', SimpleImputer()),
                    ('standard_scaler', StandardScaler()),
                    ('pca', PCA(n_components=0.95)),
                    ('xgb_clf', xgb_clf)])

xgb_param_grid = {'xgb_clf__max_depth':[2,3,5,7,10],
                  'xgb_clf__n_estimators':[10,100,500],}

xgb_grid = GridSearchCV(xgb_pipeline, 
                    xgb_param_grid, 
                    cv=5, 
                    scoring='accuracy',
                    n_jobs = -1)

xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_
print(f'XGB Accuracy Score: {xgb_grid.best_score_}')

#Random Forest Grid/Train
rf_pipeline = Pipeline([
                    ('imputer', SimpleImputer()),
                    ('standard_scaler', StandardScaler()),
                    ('pca', PCA(n_components=0.95)),
                    ('rf_clf', rf_clf)])

rf_param_grid = {'rf_clf__max_depth':[2,3,5,7,10],
                  'rf_clf__n_estimators':[10,100,500],}

rf_grid = GridSearchCV(rf_pipeline, 
                    rf_param_grid, 
                    cv=5, 
                    scoring='accuracy',
                    n_jobs = -1)

rf_grid.fit(X_train, y_train)

rf_model = rf_grid.best_estimator_
print(f'RF Accuracy Score: {rf_grid.best_score_}')
############################################################################
"""Ensemble Training"""
#Ensemble
voting_clf = VotingClassifier(estimators=[('xgb', xgb_model),
                                          ('rf', rf_model)],)

#Ensmble Pipeline
ensemble_pipeline = Pipeline([
                    ('imputer', SimpleImputer()),
                    ('standard_scaler', StandardScaler()),
                    ('pca', PCA(n_components=0.95)),
                    ('clf', voting_clf)])

#Parameter Grid
param_grid = {'clf__voting':['hard', 'soft']}

#Grid Search
ensemble_grid = GridSearchCV(ensemble_pipeline, 
                    param_grid, 
                    cv=5, 
                    scoring='accuracy',
                    n_jobs = -1)

#Model Train
ensemble_grid.fit(X_train, y_train)
ensemble_model = ensemble_grid.best_estimator_
print(f'Train Accuracy Score: {ensemble_grid.best_score_}')
############################################################################
"""Model Evaluation"""
#Ensemble Score
xgb_y_pred = xgb_model.predict(X_test)
xgb_score = accuracy_score(y_test, xgb_y_pred)

#Random Forest Score
rf_y_pred = rf_model.predict(X_test)
rf_score = accuracy_score(y_test, rf_y_pred)

#Ensemble Score
ensemble_y_pred = ensemble_model.predict(X_test)
ensemble_score = accuracy_score(y_test, ensemble_y_pred)

#Test Scores
print(f'XGB Test Accuracy Score: {xgb_score}')
print(f'Random Forest Test Accuracy Score: {rf_score}')
print(f'Ensemble Test Accuracy Score: {ensemble_score}') #Ensemble lower in accuracy potentially due to low diversity of input models

