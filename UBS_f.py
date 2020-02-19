# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 01:58:22 2020

@author: aesnj
"""
import pandas as pd
from datetime import datetime, date, time, timedelta
from dateutil.relativedelta import relativedelta
from datetime import timedelta as td
import numpy as np
from numpy import inf
import math
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder 
import tsfresh

import csv
import arrow
import requests
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.font_manager as font_manager
pd.core.common.is_list_like = pd.api.types.is_list_like

#from pandas_datareader import data as pdr
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error,f1_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
#cntk.tests.test_utils.set_device_from_pytest_env()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
#%matplotlib inline

trd = []
prc=[]
trd = pd.read_csv('C:/Users/aesnj/OneDrive/Documents/Assignment/Home Assignment/data/eurusd-trades.csv') 
prc = pd.read_csv('C:/Users/aesnj/OneDrive/Documents/Assignment/Home Assignment/data/eurusd-prices.csv') 

trd['tym'] = pd.to_datetime(trd['time']) #, format='%H:%M:%S', utc=True)
prc['tym'] = pd.to_datetime(prc['time']) #'', format='%H:%M:%S.000', utc=True)

trd['tym'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').time())
prc['tym'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').time())

#%%assuming price says same for consecutive trades with no matchinng price provided at timestamp,
#find closest time to trade within prices to dteremine price trade at for each trade, and build dataset for trade
timp=[]
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


for i in range(0, len(trd)):
   
    if trd.tym[i]  == prc.tym[i]:
        timp.append(trd.tym[i])
        #timp == prc.time[i]
    else:
        timp.append(( nearest(prc.tym, trd.tym[i] )))
        print(i)
           
#%%

# Formatting time variables into date.time 
prc['tym'] = prc['tym'].dt.time
trd['tym'] = trd['tym'].dt.time

trd['tym'].apply(lambda x: datetime.strptime(str(x), '%H:%M:%S').time())
prc['tym'].apply(lambda x: datetime.strptime(str(x), '%H:%M:%S').time())


#%% Finding Time and price of trading from price info

timp2 = pd.DataFrame(timp)
timp2.columns = ['sec']
timp2 = timp2['sec'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').time())
#%% 
#aggregate for same time, trader, quantity and deal side, and hot step encode
    
trd['prc_tym'] = timp2        
test = trd.groupby(['time','side','counterPartyId','tym'])['tradeQuantity'].sum() 

#%% Looking at aggregated results, the effect of time pass without price change is elminated, to preserve data the two
#price and trade data should be merged to reflect time with no trade happening and time where between trada prices are
#same (quantity not enough to move market price)

def date_range(start_date, end_date, increment, period):
    result = []
    nxt = start_date
    delta = relativedelta(**{period:increment})
    while nxt <= end_date:
        result.append(nxt)
        nxt += delta
    return result  

#%% function for returning 0 values as small numbers to be handled by logs in instances of comp. 
#vs occurence(actual 0), so all replacements must be done prior to one hot encoding
#replacing zeros except for delta time
def replaceZeroes(data):
  min_nonzero = data[data != 0.].min() # np.min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data
#replacing nans except for delta time, where nan is first turned to zero then to a small number
def replace_nan(data): 
    for col in data.columns:
        data[col].replace('nan', np.nan, inplace=True)
        data.replace(np.nan, 0)
    data = data.where(data.notnull(), 0) 
    return data
#Function to replace nan with min as above when needed
def replace_nan_min(data): 
    for col in data.columns:
        data[col].replace('nan', np.nan, inplace=True)
        data.replace(np.nan, 0)
    data = data.where(data.notnull(), 0) 
    min_nonzero = data[data != 0].min() # np.min(data[np.nonzero(data)])
    data = data.where(~(data==0),min_nonzero, axis = 0)
    #data[data == 0] = min_nonzero
    return data
#%%
prc2=[]
trd2=[]
prc3= []

prc2 = pd.DataFrame(prc[['tym','bidPrice','offerPrice']])

trd2 = pd.DataFrame(trd[['tym','side','tradeQuantity','counterPartyId', 'prc_tym']])
#%%
#Calculate difference in times between pricing changes and timesteps between trades
prc3=pd.DataFrame(prc2)
prc3['tym'] = prc2['tym'].apply(lambda x: datetime.strptime(str(x), '%H:%M:%S').time())

prc3['d_prc_t'] = pd.to_timedelta(prc3['tym'].astype(str)).diff(+1).dt.total_seconds()
prc3['d_prc_off'] = prc3['offerPrice'].diff()
prc3['d_prc_bid'] = prc3['bidPrice'].diff()
# at this stage which should have no "computatinal"/lack of data nans so all can be replaced with 0

prc3 = replace_nan(prc3) #[prc3.columns.difference(['d_prc_t'])])
#prc3 = replaceZeroes(prc3[prc3.columns.difference(['d_prc_t'])])
#%% look at log returns of both prices, inf is replaced with 0. Ideally a poison distirbution should be used in 
#future, at the very end a value smaller than other values will be replacing 0
prc3['d_prc_off_log'] = replace_nan_min(pd.DataFrame(np.log(prc3['d_prc_off'] /replaceZeroes(prc3['d_prc_off']) .shift(1)))) #.fillna(0)
prc3['d_prc_bid_log'] = replace_nan_min(pd.DataFrame(np.log(prc3['d_prc_bid'] /replaceZeroes(prc3['d_prc_bid']) .shift(1)))) #.fillna(0)
prc3['d_off_log'] = replace_nan_min(pd.DataFrame(np.log(prc3['offerPrice'] /replaceZeroes(prc3['offerPrice']) .shift(1)))) #.fillna(0)
prc3['d_bid_log'] = replace_nan_min(pd.DataFrame(np.log(prc3['bidPrice'] /replaceZeroes(prc3['bidPrice']) .shift(1)))) #.fillna(0)

prc3['off_log'] = np.log(prc3['offerPrice'])#.fillna(0)
prc3['bid_log'] = np.log(prc3['bidPrice'])#.fillna(0)

prc3_1 = prc3.shift(1)
prc3_1.columns = [str(col) + '_shft' for col in prc3_1.columns]
prc3_1 = replace_nan(prc3_1)
prc3 = pd.concat([prc3, prc3_1], axis=1)

#%% if the replacement above isnt applied, infinity values arise which can be removed using the bellow.
#prc3['d_prc_bid_log'] = prc3['d_prc_bid_log'].replace([np.inf, -np.inf], 0)
#prc3['d_prc_off_log'] = prc3['d_prc_off_log'].replace([np.inf, -np.inf], 0)
#prc3 = prc3.fillna(0)
#%%
prc2 = prc3
#%%look at trading quantity diff between trades and trade times
trd3=[]
trd3=pd.DataFrame(trd2)
trd3['tym'] = trd2['tym'].apply(lambda x: datetime.strptime(str(x), '%H:%M:%S').time())
trd3['d_tr_t'] = pd.to_timedelta(trd3['tym'].astype(str)).diff(+1).dt.total_seconds()

trd3['prc_tym'] = trd2['prc_tym'].apply(lambda x: datetime.strptime(str(x), '%H:%M:%S').time())
trd3['d_prc_t'] = pd.to_timedelta(trd3['prc_tym'].astype(str)).diff(+1).dt.total_seconds()
# expect variables toascertain lack of Nan values - will be difficault to trace later
trd3 = replace_nan(trd3)
#%% one approach of summing quantities (shown bellow) is calculated - not used in alpha - can skip
tst = pd.DataFrame()
tst = trd3[['tradeQuantity','side']]
tst = pd.concat([tst ,pd.get_dummies(tst ['side'], prefix='side')],axis=1)
#%% map offer and bid quantities to appropriate columns after one hot encoding 

mask1 = (tst['side_BID'] != 0)
z_valid = tst[mask1]
mask2 = (tst['side_OFFER'] != 0)
z_valid2 = tst[mask2]

tst['bid_q'] = 0
tst.loc[mask1, 'bid_q'] = z_valid['tradeQuantity'] 
tst['off_q'] = 0
tst.loc[mask2, 'off_q'] = z_valid2['tradeQuantity'] 

trd3[['bid_q','off_q']]=tst[['bid_q','off_q']]

trd3['d_bid_q'] = trd3['bid_q'].diff()
trd3['d_off_q'] = trd3['off_q'].diff()
trd3 = replace_nan(trd3)
#%% one hot encode side and quantity to features - for counterpartyID, if running without TPU or GCP skip - as done in final results

cat_columns = ["side"]
trd3= pd.get_dummies(trd3, prefix_sep="__",
                              columns=cat_columns)
#%% One hot encoding Traders
#uncomment the bellow if you decide to use counterpartyID one hot encode as part of the model- extra 300 features
#cat_columns2 = ["counterPartyId"]

#trd3= pd.get_dummies(trd3, prefix_sep="_",
                              #columns=cat_columns2)
#trd3 = trd3.drop(['tradeQuantity'], axis =1)
#trd2 = trd3

#%%
#define function to split time to three columns hour,minute,second
trd4=[]
trd4=trd3

prc4 =prc2

def el_time(data, col):
    
    data[str(col) + "_h"] = pd.DataFrame(data[col].apply(lambda x: datetime.strptime(str(x), '%H:%M:%S').hour))
    data[str(col) + "_m"] = pd.DataFrame(data[col].apply(lambda x: datetime.strptime(str(x), '%H:%M:%S').minute))
    data[str(col) + "_s"] = pd.DataFrame(data[col].apply(lambda x: datetime.strptime(str(x), '%H:%M:%S').second))


   # data[col] = data[col].dt.time
    
    #data[str(col) + '_'+'h'] = [data[col].hour for x in  data[col]]
    #data[str(col) + '_'+'m'] = [data[col].minute for x in data [col]]
    #data[str(col) + '_'+'s'] = [pdata[col].second for x in data [col]]
    return data

cols=['tym','prc_tym']    
for col in cols:
    trd4 = el_time(trd3,col)
    
trd4 = trd4.drop(['prc_tym'],axis=1)
trd4['TR']=1


trd5 = trd4
counter =pd.DataFrame()
counter['numb'] = trd5.groupby('tym').count().tradeQuantity
maybe = pd.DataFrame(pd.merge(counter,trd5, on='tym', how='right', sort = True))
maybe['cnt'] = maybe.groupby(['tym']).cumcount()+1
tym_man = maybe[['tym','numb','tym_h','tym_m','tym_s','cnt']]
tym_man['mlis'] = (tym_man['cnt']-1)*(1000/tym_man['numb'])
#tym_man['tym'] + timedelta(milliseconds=tym_man['mlis'] )

tym_man['tym2'] = tym_man.apply(lambda x: time(x.tym_h, x.tym_m, x.tym_s, int(x.mlis)), axis=1)

#tym_man['tym'].apply(lambda x: datetime.strptime(str(x), '%H:%M:%S').time())
trd5['tym']= tym_man['tym2']
    
#%% change datetime format to allow delta

d1 = prc2.tym[0]
d2 = prc2.tym[len(prc)-1]

d1 = datetime.strptime(str(prc2.tym[0]), '%H:%M:%S')
d2 = datetime.strptime(str(prc2.tym[len(prc)-1]), '%H:%M:%S')

#%% Merge the price and trade dataset together, creating one large dataset with all info

all_tym = pd.DataFrame(date_range(d1, d2, 1, 'seconds'))
all_tym2 = pd.DataFrame(pd.date_range(d1, d2, freq='10ms')) #.time)
#all_tym2 = all_tym['tym'].apply(lambda x: pd.date_range(datetime.strptime(str(x), '%H:%M:%S'), periods=100, freq='10ms'))
#slist=[]
#slist = [st for row in all_tym2 for st in row]
#timer= pd.DataFrame()
#timer = pd.DataFrame(slist)
#timer.columns = ['tim']
#slist2 = timer['tim'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S.%f').time())

#tester.apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S.%f').time())

#pd.DataFrame(date_range(d1, d2, 1,'ms'))
all_tym.columns = ['tym']
all_tym2.columns = ['tym']

all_tym['tym'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').time())
#all_tym2['tym'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S.%f').time())

#fulldate = prc2['tym'].strftime("%H:%M:%S.%f")
#prc2['tym'].apply(lambda x: datetime.strftime((x), ("%H:%M:%S.%f")))
                  
#.time())

all_tym['tym'] = all_tym['tym'].dt.time
all_tym2['tym'] = all_tym2['tym'].dt.time
all_prc = pd.DataFrame()
all_prc = pd.DataFrame(pd.merge(prc2,all_tym, on='tym', how='right', sort = True))
all_prc2 = pd.DataFrame(pd.merge(prc2,all_tym2, on='tym', how='right', sort = True))
#%%
#Forward fill our prices column to replace the "None" values with the correct long/short prices to represent the "holding" of our prices
all_prc[['bidPrice','offerPrice']] = all_prc[['bidPrice','offerPrice']].fillna(method='ffill')
all_prc2[['bidPrice','offerPrice']] = all_prc2[['bidPrice','offerPrice']].fillna(method='ffill')

# here two seperate all_prc variables are crea,'ted - one with missing values populated as 0, and the other with the 
#missing values estimated from surrounding 4 seconds- the accuracies of both of these production systems should
#be checked. The missing value nans are not to be confused with the nans from log calculations - these are the byproduct
#of merging .
all_prc1=all_prc
all_prc1['d_prc_t'] = replace_nan(pd.DataFrame(all_prc['d_prc_t']))
all_prc2['d_prc_t'] = replace_nan(pd.DataFrame(all_prc2['d_prc_t']))


cols2=['d_prc_off','d_prc_bid','d_prc_off_log','d_prc_bid_log','d_off_log','d_bid_log','off_log','bid_log']
#Cubic Spline Interpolation is selected as data is expected to have some seasonal behaviour through day

for col in cols2:
    all_prc1[col] = all_prc1[col].interpolate(method='spline', order = 2)
    all_prc2[col] = all_prc2[col].interpolate(method='spline', order = 2)
    
#%%
# is worth reminding that the same interpolation methodology was not applied to the trade data, as in that case
#0 is either a. absoloute zero in time difference or b. actual lack of trade, not of trade data    
wtf=[]
wtf = pd.DataFrame(all_prc1.merge(trd4, on='tym', how='left', sort = True))
wtf = replace_nan(wtf)


wtf2=[]
wtf2 = pd.DataFrame(all_prc2.merge(trd5, on='tym', how='left', sort = True))
wtf2 = replace_nan(wtf2)
#%%
#The last step is to calculate the actual returns to define target variable:


def wrnglr(wtf):
    wtf['AR'] = ( (wtf['off_q']*wtf['offerPrice'])-(wtf['bidPrice']*wtf['bid_q']) )*wtf['TR']
    wth=[]
    #wtf_f = el_time(wtf, 'tym')
    wtf_f = replace_nan(wtf)
    pd.set_option("display.precision", 8)
    #diffence between bid and offer spread is considered to be of high impact on our target variable, the more the better.
    wtf_f['del'] = wtf['offerPrice']-wtf['bidPrice']
    wtf_f['t_AR'] = wtf_f['AR'].cumsum()
    
    #defining Y target as a function of change in spread, and increase/decrease od prices
    
    wtf_f['rol_off_p'] = wtf_f['offerPrice'].pct_change()
    wtf_f['rol_bid_p'] = wtf_f['bidPrice'].pct_change()
    wtf_f['rol_del'] = wtf_f['del'].pct_change()
                              
    # The Second Target Variable is defined as change in bid price minus change in offer price, plus change in spread change      
    #y=pd.Series()
    wtf_f['y'] = wtf_f['rol_del']  + wtf_f['rol_bid_p'] + wtf_f['rol_off_p']
    wth = wtf_f
    return wth

wtf2 = wrnglr(wtf2)
wtf_f['y'].plot()
wtf2['y'].plot()
#%%
#removing noise effect by introducing rolling means and stdv
def movbol(data,col1, window, no_of_std):  

    rolling_mean = data[col1].rolling(int(window)).mean()
    rolling_std = data[col1].rolling(int(window)).std()

    #create two new DataFrame columns to hold values of upper and lower Bollinger bands
    data[str(col1)+'_' +'rol_mean'+'_'+ str(window)] = rolling_mean
    data[str(col1)+'_' +'bol_h'+'_'+ str(window)] = rolling_mean + (rolling_std * no_of_std)
    data[str(col1)+'_' +'bol_l'+'_'+ str(window)] = rolling_mean - (rolling_std * no_of_std)
    return data

# Caldulate and plot results
cols = ['bidPrice','offerPrice']
windows = [ 10, 100 , 1000 ]
g1=[]
for window in windows:
    for col in cols:
        #g1 = movbol(wtf_f, col, window, 3)
        g1 = movbol(wtf2, col, window, 3)
        #wtf_f[[col ,str(col)+'_' +'bol_h'+'_'+ str(window), str(col)+'_' +'bol_l'+'_'+ str(window)]].plot()
        wtf2[[col ,str(col)+'_' +'bol_h'+'_'+ str(window), str(col)+'_' +'bol_l'+'_'+ str(window)]].plot()

#wtf_f = replace_nan(wtf_f)
wtf2 = replace_nan(wtf2)
def process_data(data):
    
    data['DCH']=ta.volatility.donchian_channel_hband(data['offerPrice']) #Donchian Channel High Band
    data['DCL']=ta.volatility.donchian_channel_lband(data['offerPrice']) #Donchian Channel Low Band
    data['DPO']=ta.trend.dpo(data['offerPrice']) #Detrend Price Oscilator
    data['EMAf']=ta.trend.ema_indicator(data['offerPrice']) #Expornential Moving Average fast
    data['FI']=ta.volume.force_index(data['offerPrice'], data['tradeQuantity']) # Force Index(reveals the value of a trend)

    data['KST']=pd.Series(ta.trend.kst(data['offerPrice'])) #KST Oscillator (KST) identify major stock market cycle junctures
    data['MACD']=ta.trend.macd(data['offerPrice']) # Moving Average convergence divergence
    data['OBV']=ta.volume.on_balance_volume(data['offerPrice'], data['tradeQuantity']) # on_balance_volume_mean
    data['RSI']=ta.momentum.rsi(data['offerPrice']) # Relative Strength Index (RSI)
    data['TRIX']=ta.trend.trix(data['offerPrice']) #Shows the percent rate of change of a triple exponentially smoothed moving average
    data['TSI']=ta.momentum.tsi(data['offerPrice']) #True strength index (TSI)
    data['ROC1']=(data['offerPrice']-data['bidPrice'])/data['bidPrice']
    data['RET']=data['offerPrice'].pct_change()
    #data['y'] = np.where(data['bidPrice'] <= data['offerPrice'],1,-1)
    data=data.dropna()
    return data

#process_data(wtf_f)
process_data(wtf2)

#counter = wtf_f.groupby('tym').count()
#l1 = counter.max()
#counter['bidPrice'].argmax()


#%% Feature Importance Testing

def feature_imp(data):
    corrmat=data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sb.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    plt.title('Correlation between different fearures and target')
    plt.show()
    return
#%%
def modeling(data):
    Xi = data.drop(['y'], axis=1)
    scaler=StandardScaler().fit(Xi) # Use the standard scaler function from scikit learn
    Xs = scaler.transform(Xi) 
    #pca = PCA(n_components=3)
    #pca.fit(Xi)
    #X = pca.transform(Xi)
    X=Xs
    Y=data['y']
    global xTrain
    global xTest 
    global yTrain 
    global yTest
    xTrain, xTest, yTrain, yTest = train_test_split(X,Y, test_size = 0.2, random_state = 0)
    models = []
    models.append(('LR' , LogisticRegression()))
    models.append(('LDA' , LinearDiscriminantAnalysis()))
    models.append(('KNN' , KNeighborsClassifier()))
    models.append(('CART' , DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=6)))
    models.append(('NB' , GaussianNB()))
    models.append(('SVM' , SVC()))
    models.append(('RF' , RandomForestClassifier(n_estimators=60)))
    models.append(('XGBoost', XGBClassifier(gamma=0.0, n_estimators=60,base_score=0.7, max_depth=3, objective = "binary:logistic", colsample_bytree=1,learning_rate=0.01)))
    
    results = []
    names = []
    '''
    for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=42)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg) '''
    for name, model in models:
        clf = model
        clf.fit(xTrain, yTrain)
        y_pred = clf.predict(xTest)
        accu_score = accuracy_score(yTest, y_pred)
        results.append([name, accu_score])
        #print(name + ": " + str(accu_score))
    re=pd.DataFrame(results, columns=['Model', 'Acuracy_Score'])
    re.set_index(['Model'])
    return re

#%%
def predstockmvt(xTrain, yTrain, xTest):
    clf1 = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=6)
    clf1=clf1.fit(xTrain, yTrain)
    yPreddt = clf1.predict(xTest)
    report=classification_report(yTest,yPreddt)
    print(report)
    return
#%% Back Testing
    

def backtest(data):
    trade_data=data.iloc[len(xTrain):]
    trade_data['signal']=0
    trade_data.loc[trade_data['y']>=1,'signal']=1
    trade_data.loc[trade_data['y']<=-1,'signal']=0
    trade_data['Strategy_return']=trade_data['signal']*trade_data['ROC1']
    trade_data['Market_return']=trade_data['ROC1']
    global perf
    perf=trade_data[['Market_return', 'Strategy_return']].cumsum()
    #trade_data[['Market_return'], ['Strategy_return']]
    plt.figure(figsize=(10,10))
    perf.plot()
    plt.title('Evolution of Cumulative Returns')
    plt.show()
    return
#%%
# at this point when trying to draw the corr heat map for the system, the huge amount of one hot encoded 
#trade counterpartyId data made the correlations unrealstic for viewing/impossible to train on systems at hand,
#as such although ideally it would be one hot encoded, in this example it is kept as one column variable .
feature_imp(wtf2)
sb.heatmap(wtf2.corr())
wtf2.plot(x="tym", y=["bidPrice", "offerPrice"])
plt.savefig('price.png', bbox_inches='tight')

wtf2.plot(x="tym", y=["bid_q", "off_q"])
plt.savefig('quantity.png', bbox_inches='tight')

#%% Here we try to extract the relevant features based on our y 

from tsfresh import extract_relevant_features

features_filtered_direct = extract_relevant_features(wtf2, y,
                                                     column_id='counterPartyId', column_sort='tym')

                                      
wtf2.plot(x="tym", y=["del"])
plt.savefig('spread.png', bbox_inches='tight')

wtf2.plot(x="tym", y=["t_AR"])
plt.savefig('evolution.png', bbox_inches='tight')

#%% splitting into train and test
def split_data(X, y, tr_percent):
    length_tr = int(np.floor(tr_percent*(X.shape[0])))
    X_tr = np.zeros((length_tr,X.shape[1]))
    y_tr = np.zeros((length_tr,1))
    for i in range(0,length_tr):
        for j in range(0,X.shape[1]):
            X_tr[i,j] = X[i][j]
            y_tr[i] = y[i]
            length_test = int(np.floor((1-tr_percent)*(X.shape[0])))
            X_test = np.zeros((length_test,X.shape[1]))
            y_test = np.zeros((length_test,1))
    for i in range(0,length_test):
        for j in range(0,X.shape[1]):
            X_test[i,j] = X[i+length_tr][j]
            y_test[i] = y[i+length_tr]
            y_tr = y_tr.ravel()
            y_test = y_test.ravel()
    return X_tr, y_tr, X_test, y_test, length_test


split_data()
sb.heatmap(wtf2)

#%% FInal ML runs
se=[]
se = wtf2
feature_imp(se)
se = se.drop(['tym','tym_shft'], 1)
se = replace_nan(se)
#initially for simplicity a basic classification problem is defined - while our target is continous and a measure of movement
#of prices, if positive is kept at 1 and if negative is kept at 0 and indicator for trade or dont trade 
se['y'][se['y']>0]=1
se['y'][se['y']==0]=0
se['y'][se['y']<0]=0
print(modeling(se))
Y=predstockmvt(xTrain, yTrain, xTest)
print(yTest.value_counts())
print('This justifies the prediction above.')

backtest(se)
#%%
#%% From gere onwards is only applicable if One Hot encoding has been applied to CounterParty ID
#looking for columns with "CounterPartyID" related naming
f1 =wtf2.pivot("counterPartyId", "tradeQuantity", "AR")

wtf_hvy = [col for col in wtf.columns if 'counterPartyId' in col]
wtf.drop(wtf_hvy, axis=1, inplace=True)
# redrawing the corr plot 

colz = wtf.columns
 
wtf ['h'] = [x.hour for x in wtf ['tym']]
wtf ['m'] = [x.minute for x in wtf ['tym']]
wtf ['s'] = [x.second for x in wtf ['tym']]
feature_imp(wtf)
#upon inspection it is clear we have fell into the dummy data trap - by imcluding both side__bid and side__offer
#into the features


#%%

#%%

#pd.concat([all_tym, prc2], sort=True)
       # timp ==  pd.DataFrasideme(prc.ti/me[[i - 1]]).timeated
        #(prc.tym).truncate(before=trd.tym[5]) #nearest(trd.time, prc.time[i])
    #timp.append[i]   
    #trd.clstime[i]==timp
    
    #(prc.time).iloc[(prc.time).get_loc(datetime.datetime(trd.time[i]),method='nearest')]
    


#trd.truncate(before='23')
#timp == pd.DataFrame datetime.datetime(int(str(prc.tym[5]))) .