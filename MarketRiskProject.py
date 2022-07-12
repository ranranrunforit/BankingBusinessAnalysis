# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:33:58 2022

@author: Chaoran
"""

import pandas as pd
import numpy as np
from sklearn import *
import datetime
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import matplotlib as mlp
import arch
from arch.unitroot import ADF
from arch import arch_model
import arch.data.sp500

def data_reading(path):
    newStrat1=pd.read_csv(path)
    newStrat1.index=newStrat1['Date'].map(lambda x : datetime.datetime.strptime(x,"%m/%d/%Y"))
    del newStrat1['Date']
    return newStrat1

def data_reading_clipboard():
    newStrat1=pd.read_clipboard()
    newStrat1.index=newStrat1['Date'].map(lambda x : datetime.datetime.strptime(x,"%m/%d/%Y"))
    del newStrat1['Date']
    return newStrat1

newStrat = data_reading(r'H:\Desktop\Market_Risk_Data.csv')
ret=np.log((newStrat['SPX']/newStrat['SPX'].shift(1)).dropna())

data = arch.data.sp500.load()
data['Adj Close'].plot()
market = data['Adj Close']
ret = 100 * market.pct_change().dropna()

def tsplot(y,lags=None,figsize=(10,8),style='bmh'):
    if not isinstance(y,pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family']='Ubuntu Mono'
        layout = (3,2)
        ts_ax = plt.subplot2grid(layout,(0,0),colspan=2)
        acf_ax = plt.subplot2grid(layout,(1,0))
        pacf_ax = plt.subplot2grid(layout,(1,1))
        qq_ax = plt.subplot2grid(layout,(2,0))
        pp_ax = plt.subplot2grid(layout,(2,1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y,sparams=(y.mean(), y.std()), plot=pp_ax)
        
        plt.tight_layout()
        plt.show()
    return

adf =ADF(market) #union root test
print(adf.summary().as_text())

tsplot(ret, lags=30)
tsplot(ret*ret, lags=30)

adf = ADF(ret)
print(adf.summary().as_text())

am = arch_model(ret,mean='ARX',lags=1,vol='Garch',p=1,o=0,q=1,dist='Noraml')
res = am.fit(update_freq=5)

tsplot(res.resid.dropna(),lags=30)
tsplot(res.resid.dropna()*res.resid.dropna(),lags=30)

new_resid = (res.resid/res.conditional_volatility).dropna()
tsplot(new_resid,lags=30)
tsplot(new_resid*new_resid,lags=30)

print(res.summary())
print(res.params)
print(res.forecast().mean.tail(1).value[0][0])
print(res.forecast().variance.tail(1).value[0][0])
print(res.forecast().residual_variance.tail(1).value[0][0])
const,beta,omega,alpha,beta=res.params


np.random.seed(2)
horizon = 10 # 10 days
simulationPath = 10000 # more smooth is     better
ret_list=[]

for j in range(1,simulationPath):
    w=np.random.normal(size=horizon)
    #residual
    eps = np.ones(horizon,dtype=np.float64)*(res.forecast().mean.tail(1).value[0][0])
    sigsq = np.ones(horizon,dtype=np.float64)*(res.forecast().variance.tail(1).value[0][0])
    for i in range(1,horizon):
        sigsq[i] = omega + alpha*(eps[i-1]**2) + beta*sigsq[i-1] # GARCH MODEL 
        eps[i] = const + beta * eps[i-1] + w[i] * np.sqrt(sigsq[i])
    ret_list.append(eps[-1])#update eps time series

plt.hist(ret_list,bins=20)
plt.show()
a=np.sort(ret_list)
print(a)
print("95% 10 days VaR" + str(np.percentile(ret_list,5)))



am = arch_model(ret, mean='ARX',lags=1,vol='GARCH',power=2.0,p=1,o=1,q=1,dist='Normal')
res = am.fit(update_freq=5)
print(res.summery())

am = arch_model(ret,mean='ARX',lags=1,vol='Garch',p=1,o=0,q=1,dist='skewt')
res = am.fit(update_freq=5)
print(res.summery())

am = arch_model(ret,mean='ARX',lags=1,vol='GARCH',power=2.0,p=1,o=1,q=1,dist='skewt')
res = am.fit(update_freq=5)
print(res.summery())