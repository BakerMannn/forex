############################################################################
"""To Do"""
#Regression Logic
#Typical Price Movement Distributions
#Best estimator logic for reg
#Buy or Sell Function

############################################################################
"""Packages Import"""
#Data Manipulation/Import
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

############################################################################
"""Ticker Archive"""
eurusd = 'EURUSD=X'
bitcoin  = 'BTC-USD'
sp500 = 'SPY'
google = 'GOOGL'

############################################################################
"""Global Variables"""
#Misc
ticker = input('Ticker:')
drop_labels = ['close_offset', 'close_higher']
label = 'close_higher'
trade_prob_quantile = 0.85

#Model Training
cv = 5
scoring = 'accuracy'
n_jobs = -1
pca_n = 0.95

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

############################################################################
"""Data Split"""
#Split Data
X = price_data.drop(columns=drop_labels)
y = price_data[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=555)

############################################################################
"""Base Models Training"""
#Instantiate Models
xgb_clf = xgb.XGBClassifier(seed=555)
rf_clf = RandomForestClassifier(random_state=555)
lr_clf = LogisticRegression(random_state=555)

#XGB Grid/Train
xgb_pipeline = Pipeline([
                    ('imputer', SimpleImputer()),
                    ('standard_scaler', StandardScaler()),
                    ('pca', PCA(n_components=pca_n)),
                    ('xgb_clf', xgb_clf)])

xgb_param_grid = {'xgb_clf__max_depth':[2,3,5,7,10],
                  'xgb_clf__n_estimators':[10,100,500],}

xgb_grid = GridSearchCV(xgb_pipeline, 
                    xgb_param_grid, 
                    cv=cv, 
                    scoring=scoring,
                    n_jobs = n_jobs)

xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_
print(f'XGB Train Accuracy Score: {xgb_grid.best_score_}')

#Random Forest Grid/Train
rf_pipeline = Pipeline([
                    ('imputer', SimpleImputer()),
                    ('standard_scaler', StandardScaler()),
                    ('pca', PCA(n_components=pca_n)),
                    ('rf_clf', rf_clf)])

rf_param_grid = {'rf_clf__max_depth':[2,3,5,7,10],
                  'rf_clf__n_estimators':[10,100,500],}

rf_grid = GridSearchCV(rf_pipeline, 
                    rf_param_grid, 
                    cv=cv, 
                    scoring=scoring,
                    n_jobs =n_jobs)

rf_grid.fit(X_train, y_train)

rf_model = rf_grid.best_estimator_
print(f'RF Train Accuracy Score: {rf_grid.best_score_}')

#Logistic Regression Grid/Train
lr_pipeline = Pipeline([
                    ('imputer', SimpleImputer()),
                    ('standard_scaler', StandardScaler()),
                    ('pca', PCA(n_components=pca_n)),
                    ('lr_clf', lr_clf)])

lr_param_grid = {'lr_clf__C': [0.1, 1, 10, 100],
                  'lr_clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'lr_clf__penalty': ['none', 'l1', 'l2', 'elasticnet']}

lr_grid = GridSearchCV(lr_pipeline, 
                    lr_param_grid, 
                    cv=cv, 
                    scoring=scoring,
                    n_jobs = n_jobs)

lr_grid.fit(X_train, y_train)

lr_model = lr_grid.best_estimator_
print(f'Log Reg Train Accuracy Score: {lr_grid.best_score_}')

############################################################################
"""Ensemble Training"""
#Ensemble
voting_clf = VotingClassifier(estimators=[('xgb', xgb_model),
                                          ('rf', rf_model),
                                          ('lr', lr_model)])

#Ensmble Pipeline
ensemble_pipeline = Pipeline([
                    ('imputer', SimpleImputer()),
                    ('standard_scaler', StandardScaler()),
                    ('pca', PCA(n_components=pca_n)),
                    ('clf', voting_clf)])

#Parameter Grid
param_grid = {'clf__voting':['hard', 'soft']}

#Grid Search
ensemble_grid = GridSearchCV(ensemble_pipeline, 
                    param_grid, 
                    cv=cv, 
                    scoring=scoring,
                    n_jobs = n_jobs)

#Model Train
ensemble_grid.fit(X_train, y_train)
ensemble_model = ensemble_grid.best_estimator_
print(f'Ensemble Train Accuracy Score: {ensemble_grid.best_score_}')

############################################################################
"""Model Evaluation"""
#Dummy Estimator
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=555)
dummy_clf.fit(X_train, y_train)

dummy_y_pred = dummy_clf.predict(X_test)
dummy_score = accuracy_score(y_test, dummy_y_pred)
dummy_report = classification_report(y_test, dummy_y_pred)

#XGB Score
xgb_y_pred = xgb_model.predict(X_test)
xgb_score = accuracy_score(y_test, xgb_y_pred)
xgb_report = classification_report(y_test, xgb_y_pred)

#Random Forest Score
rf_y_pred = rf_model.predict(X_test)
rf_score = accuracy_score(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred)

#Logistic Regression Score
lr_y_pred = lr_model.predict(X_test)
lr_score = accuracy_score(y_test, lr_y_pred)
lr_report = classification_report(y_test, lr_y_pred)

#Ensemble Score
ensemble_y_pred = ensemble_model.predict(X_test)
ensemble_score = accuracy_score(y_test, ensemble_y_pred)
ensemble_report = classification_report(y_test, ensemble_y_pred)

#Test Scores
model_scores = {xgb_model:xgb_score, 
                rf_model:rf_score, 
                lr_model:lr_score, 
                ensemble_model:ensemble_score}

model_reports = [('Dummy', dummy_report), 
                ('XGB', xgb_report), 
                ('Random Forest', rf_report), 
                ('Log Reg', lr_report), 
                ('Ensemble', ensemble_report)]

for name, report in model_reports:
    print(f'{name} Classification Report:\n'
          f'{report}')

############################################################################
"""Decision Function"""
best_clf = max(model_scores, key=model_scores.get)

#Probablity Function
prob_df = X_test.copy()
prob_df[['prob_sell', 'prob_buy']] = best_clf.predict_proba(X_test)
upper_limit = prob_df['prob_buy'].quantile(trade_prob_quantile)
prob_df['decision'] = prob_df['prob_buy'].apply(lambda x: 'buy' if x> upper_limit else ('sell' if x < (1-upper_limit) else 'no_trade'))
trades = prob_df['decision'].value_counts()

#Cumulative Returns
cumulative_returns_df = prob_df.copy()
cumulative_returns_df['decision'].replace(['buy', 'sell', 'no_trade'], [1,-1,0], inplace=True)
cumulative_returns_df['close_offset'] = cumulative_returns_df['Close'].shift(-1)
cumulative_returns_df.dropna(inplace=True)

cumulative_returns_df = cumulative_returns_df.assign(PL = lambda x: (x['close_offset'] - x['Close']) * x['decision'])

total_pl = cumulative_returns_df['PL'].sum()
print(f'Cumulative Returns: {round(total_pl,2)}')
print(f'Buy Threshold: {round(upper_limit,3)}')
print(f'Sell Threshold: {round((1-upper_limit),3)}')
print(trades)
############################################################################
"""Model EDA"""
# #Sell Probability Distribution
# sns.kdeplot(data=cumulative_returns_df,
#             x='prob_sell')
# plt.show()

# #Buy Probability Distribution
# sns.kdeplot(data=cumulative_returns_df,
#             x='prob_buy')
# plt.show()