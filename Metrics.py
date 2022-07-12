# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:11:06 2022

@author: Chaoran
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlit.dates as mdates
import datetime
import os
import seaborn as sns
sns.set()
#import Bloomberg_data as bbg


class AnalysisReport():

    def ___init___(self,df,weightings):
        self.df = df
        self.weightings = weightings
        
    def general_analysis(self,period='daily'):
        annualized_factor = 252
        df = self.df
        if period == 'weekly':
            annualized_factor = 52
            df = self.df.resample('W-FRI').last()
        elif period == 'monthly':
            df = self.df.resample('M').last()
        elif period == 'yearly':
            annualized_factor = 1
            df = self.df.resample('Y').last()
        
        res = {}
        
        for i in range(df.shape[1]):
            irr = (df.iloc[-1,i]/df.iloc[0,i])**(365.0/(df.index[-1] - df.index[0]).days)-1
            realized_volatility = np.log(df.iloc[:,i],df.pct_change(1)+1).std()*np.sqrt(annualized_factor)
            sharpe = irr/realized_volatility
            res[df.columns[i]] = (irr,realized_volatility,sharpe)
        
        return res
    
    @staticmethod
    def rebase_serial(df):
        res = np.ones(df.shape)*100
        for j in range(0, df.shape[1]):
            for i in range(1, df.shape[0]):
                res[i,j]=res[i-1,j]*df.iloc[i,j]/df[i-1,j]
        return pd.DataFrame(res,index=df.index,columns=df.columns)
    
    def return_series(self, period='monthly'):
        df = self.df
        if period == 'weekly':
            df = self.df.resample('W-FRI').last()
        elif period == 'monthly':
            df = self.df.resample('M').last()
        elif period == 'yearly':
            df = self.df.resample('Y').last()
        
        returns = df.pct_change(1)
        returns = returns.dropna(axis=0)
        return returns
    
    def return_analysis(self,period='daily'):
        returns = self.retuurn_series(period)
        res = {}
        
        for i in range(self.df.shape[1]):
            percent_positive = len(returns[returns.iloc[:,i] >= 0])/len(returns)
            avg_returns = returns.iloc[:,i].mean()
            avg_positive_returns = returns[returns.iloc[:,i] >= 0].iloc[:,i].mean()
            avg_negative_returns = returns[returns.iloc[:,i] < 0].iloc[:,i].mean()
            std_returns = returns.iloc[:,i].std()
            std_positive_returns = returns[returns.iloc[:,i] >= 0].iloc[:,i].std()
            std_negative_returns = returns[returns.iloc[:,i] < 0].iloc[:,i].std()
            res[self.df.columns[i]] = [percent_positive,avg_returns,avg_positive_returns,avg_negative_returns,std_returns,std_positive_returns,std_negative_returns]

        return res
    
    def max_drawdown(self):
        res = np.ones(self.df.shape)
        for j in range(self.df.shape[1]):
            peak = -np.inf
            for i in range(self.df.shape[0]):
                if self.df.iloc[i,j] >= peak:
                    peak = self.df.iloc[i,j]
                    
                res[i,j] = self.df.iloc[i,j]/peak - 1.0
                
        return pd.Dataframe(res,index=self.df.index)
    
    def drawdown_window(self, window=252):
        roll_max = self.df.rolling(window).max()
        daily_drawdown = self.df/roll_max - 1.0
        return daily_drawdown
    
    def correlation_analysis(self, window=252,period='daily'):
        df = self.df
        if period == 'weekly':
            df = self.df.resample('W-FRI').last()
        elif period == 'monthly':
            df = self.df.resample('M').last()
        elif period == 'yearly':
            df = self.df.resample('Y').last()

        df = np.log(df.pct_change(1) + 1)         
        df = df.dropna(axis = 0)
        corr_matrix = df.iloc[-window:,:].corr()
        histo_corr = {}
        for i in range(1,df.shape[1]):
            rolling_corr = df.rolling(window).corr(other=df.iloc[:,i]).iloc[:,0]
            rolling_corr = rolling_corr.dropna(axis=0)
            histo_corr[str(df.columns[0])+'/'+str(df.columns[i])] = rolling_corr
        
        return corr_matrix,histo_corr
    
    def yearly_plot(self):
        years = list(self.df.index.year.unique())
        fig,axs = plt.subplots(1+len(years)//5,min(5,len(years)),figsize=(15,3*(1+len(years)//5)))
        fig.subplots_adjust(hspace=.75)
        
        last_ax = None
        for i,ax in enumerate(axs,flat):
            if i >= len(years):
                ax.set_visible(False)
            else:
                last_ax = ax
                year_series = AnalysisReport.rebase_serial(self.df[str(years[i])])
                for j in range(self.df.shape[1]):
                    ax.plot(year_serie.iloc[:,j],label=self.df.columns[j])
                #ax.legend(loc)
                ax.set_title('Performance in '+str(years[i]))
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
        last_ax.legend(bbox_to_anchor = (1.05,0),loc='lower left',borderaxespade)
        plt.show()
                    
    
    def monthly_returns_distribution_plot(self):
        monthly_returns = self.return_series['monthly']
        if df.shape[1] > 1:
            fig,axs = plt.subplots(1+len(years)//3,min(3,len(years)),figsize=(15,3*(1+len(years)//5)))
            fig.subplots_adjust(hspace=.35)
            for i,ax in enumerate(axs,flat):
                if i >= self.df.shape[1]:
                    ax.set_visible(False)
                else:
                    ax.hist(monthly_returns.iloc[:,i],bins=np.linspace(-0.05,0.05,30))
                    ax.set_title('Monthly returns distribution - '+str(self.df.columns[i]))
            plt.show()
        else:
            fig,axs = plt.subplots()
            ax.hist(monthly_returns.iloc[:,0],bins=np.linspace(-0.05,0.05,30))
            ax.set_title('Monthly returns distribution - '+str(self.df.columns[0]))
            plt.show()
            
    
    def monthly_returns_plot(self):
        monthly_returns = self.return_series['monthly']
        if df.shape[1] > 1:
            fig,axs = plt.subplots(self.df.shape[1] , 1,figsize=(15,3*self.df.shape[1]))
            fig.subplots_adjust(hspace=.75)
            for i,ax in enumerate(axs,flat):
                if i >= self.df.shape[1]:
                    ax.set_visible(False)
                else:
                    ax.bar(range(monthly_returns.shape[0],monthly_returns.iloc[:,i]))
                    ax.set_title('Monthly returns '+str(self.df.columns[i]))
                    ax.set_xticks(range(0,len(monthly_returns.index),12))
                    ax.set_xticklabels([monthly_returns.index[i].strftime('%m-%d-%Y') for i in range(0,len(monthly_returns.index))])
            plt.show()
        else:
            fig,axs = plt.subplots()
            ax.bar(range(monthly_returns.shape[0],monthly_returns.iloc[:,0]))
            ax.set_title('Monthly returns '+str(self.df.columns[0]))
            ax.set_xticks(range(0,len(monthly_returns.index),12))
            ax.set_xticklabels([monthly_returns.index[i].strftime('%m-%d-%Y') for i in range(0,len(monthly_returns.index))])
            plt.show()


    def perf_plot(self):
        mdd = self.max_drawdown()
        fig,axs = plt.subplot(1,1,figsize=(15,6))
        fig.subplots_adjust(hspace=0.35)
        for i in range(self.df.shape[1]):
            axs.plot(AnalysisReport.rebase_serial(self.df).iloc[:,i],label=str(self.df.columns))
        axs.legend(loc='lower right')
        ax2 = axs.twinx()
        axs.fill_between(mdd.index,mdd.iloc[:,0],0,color='orange',alpha=0.5,label='drawdown')
        ax2.set_ylim(top=0)
        axs.set_title("Cumulative performance and drawdown")
        plt.show()
    
    def correlation_plot(self, window=252,period='daily'):
        corr_matrix,histo_corr = self.correlation_analysis(window,period)
        
        fig,ax = plt.subplot(1,1,figsize=(15,4))
        for pair in histo_corr:
            ax.plot(histo_corr[pair],label=str(pair))
        ax.legend(loc='lower right')
        ax.set_title('Correlation on'+str(period)+'-log returns')
        plt.show()
        
    def general_analysis_output(self):
        ga_d = self.general_analysis('daily')
        ga_w = self.general_analysis('weekly')
        ga_m = self.general_analysis('monthly')
        for elem in ga_d:
            print('\n', elem)
            print('irr,', ga_d[elem][0],',',ga_w[elem][0],',',ga_m[elem][0])
            print('vol,', ga_d[elem][1],',',ga_w[elem][1],',',ga_m[elem][1])
            print('irr/vol,', ga_d[elem][2],',',ga_w[elem][2],',',ga_m[elem][2])

    
    def return_analysis_output(self):
        ra_d = self.return_analysis('daily')
        ra_w = self.return_analysis('weekly')
        ra_m = self.return_analysis('monthly')
        
        for elem in ra_d:
            print('\n',elem)
            print('% positive, ', ra_d[elem][0],',',ra_w[elem][0],',',ra_m[elem][0])
            print('avg returns, ', ra_d[elem][1],',',ra_w[elem][1],',',ra_m[elem][1])
            print('avg pos returns, ', ra_d[elem][2],',',ra_w[elem][2],',',ra_m[elem][2])
            print('avg neg returns, ', ra_d[elem][3],',',ra_w[elem][3],',',ra_m[elem][3])
            print('std returns, ', ra_d[elem][4],',',ra_w[elem][4],',',ra_m[elem][4])
            print('std pos returns, ', ra_d[elem][5],',',ra_w[elem][5],',',ra_m[elem][5])
            print('std neg returns, ', ra_d[elem][6],',',ra_w[elem][6],',',ra_m[elem][6])
            
    def output(self):
        self.perf_plot()
        self.yearly_plot()
        self.monthly_returns_plot()
        self.monthly_returns_distribution_plot()
        self.correlation_plot()
        
        print('\n\nGENERAL ANALYSIS')
        self.general_analysis_output()
        print('\n\nRETURN ANALYSIS')
        self.return_analysis_output()
        
        #print(AnalysisReport.print_dico('General analysis',self.general_analysis('daily')))
        
df2=pd.read_csv(r'/Users//Downloads/data_set.csv')
df2.index=df2['Date'].map(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y"))
del df2['Date']
df2.columns = [df2.columns[i].upper() for i in range(df2.shae[1])]
df2 = df2.dropna(axis=0)

df = df2
ar = AnalysisReport(df, None)
ar.output()