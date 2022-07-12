# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:43:45 2022

@author: Chaoran
"""


import panadas as pd
import numpy as np
import datetime

from class_CalendarUS import *

class Strategy:
    
    """ special methods """
    def ___repr___(self):
        """ method that displays the strategy index """
        print(self.newStrat)
        
    def ___init___(self,newStrat_dates,newStrat_strat):
        newStrat=pd.DataFrame({'Dates':newStrat_dates,'strat':newStrat_strat})
        newStrat.set_index('Dates',inplace = True) 
        self.newStrat=newStrat
        
    """ class procedures """
    
    """ Statistical ratios """
    
    def IRR(self):
        """ returns the IRR """
        newStrat=self.newStrat
        dfin=newStrat.index[len(newStrat)-1]
        debut=newStrat.index(0)
        days=(dfin-debut).days
        Irr=(newStrat.loc[newStrat.index[len(newStrat)-1],'strat']/newStrat.loc[newStrat.index[0],'strat'])**(365.0/days)-1
        return Irr
        
    def Daily_Volatility(self):
        """ return the volatility """
        newStrat=self.newStrat
        vol=np.log(newStrat['strat'].pct_change()+1).std()
        return vol
    
    def Annualized_Volatility(self):
        """ returns the volatility """
        newStrat=self.newStrat
        vol=np.log(newStrat['strat'].pct_change()+1).std()*np.sqrt(252)
        return vol
    
    def Annualized_Vol_Down(self):
        """ returns the volatility of downwards moves """
        newStrat=self.newStrat
        vol=np.log(newStrat['strat'].pct_change()+1).map(lambda x: 0 if x>0 else x)
        vol=vol.std()*np.sqrt(252)
        return vol
    
    def Sharpe(self):
        """ returns the Sharpe Ratio """
        vol=self.Annualized_Volaility()
        Irr=self.IRR()
        return Irr/vol
    
    def Sharpe_Cleared(self):
        """ returns the Sharpe Ratio excluding the upward volatility """
        vol=self.Annualized_Vol_Down()
        Irr=self.IRR()
        return Irr/vol
    
    def Max_DrawDown(self):
        """ return the Max DD """
        newStrat = self.newStrat
        MaxDD = 1.0
        for k in range(0,len(newStrat)-1):
            dMax = max(newStrat['strat'][0:k+1])
            dMin = min(newStrat['strat'][k+1:len(newStrat)+1])
            if dMin/dMax<MaxDD:
                MaxDD=dMin/dMax
        return MaxDD-1
    
    
    def Annual_Perf(self):
        """ return the Annualized return """
        newStrat = self.newStrat
        newStrat['Year']=newStrat.index.map(lambda x: x.year)
        years=newStrat['Year'].drop_duplicates().to_list()
        perf=[]
        for i in range(0,len(years)):
            temp = newStrat[newStrat['Year']==years[i]]
            begin = temp.index[0]
            position = (newStrat.index.tolist()).index(begin)
            if i>0:
                begin=newStrat.index[position-1]
            end=temp.index(len(temp)-1)
            perf.append(newStrat['strat'][end]/newStrat['strat'][begin]-1)
        del newStrat['Year']
        dfAnnualPerf = pd.DataFrame({'Year':years,'Perf':perf})
        dfAnnualPerf = dfAnnualPerf.sort_index(axis=1,ascending=False)
        return dfAnnualPerf
    
    
    def Monthly_Perf(self):
        """ return the Monthly return """
        dfAnnualPerf = self.Annual_Perf()
        dfAnnualPerf.set_index('Year',inplace=True)
        newStrat = self.newStrat
        calendar=Calendar(newStrat.index[0],newStrat.index[len(newStrat)-1])
        newStrat['Year']=newStrat.index.map(lambda x: x.year)
        newStrat['Month']=newStrat.index.map(lambda x: x.month)
        
        years=newStrat['Year'].drop_duplicates().to_list()
        performance=pd.DataFrame({'Year':years})
        month = ['January','February','March','April','May','June','July','August','September','October','November','December']
        monthEquiv={}
        for k in range(1,13):
            monthEquiv[k,'Month']=month[k-1]
            
        for months in month:
            performance[months]=np.nan
            
        performance.set_index('Year',inplace=True)
        performance['YTD']=0.0
        firstDate = newStrat.index[0]
        lastDate = newStrat.index[len(newStrat)-1]
        
        bfirstdone = False
            
        for year in sorted(years,reverse=False):
            
            performance.loc[year,'YTD'] = dfAnnualPerf.loc[year,'Perf']
            for month in range(1,13):
                if bfirstdone==False: #first year
                    if firstDate.month == month:
                        MonthLastday = calendar.MonthLastBusinessDay(datetime.datetime(month))
                        endPeriodPerf = newStrat[newStrat.index<=MonthLastday].tail(1)
                        lastendperiod=endPeriodPerf
                        perf = newStrat.loc[endPeriodPerf,'strat']/newStrat.loc[firstDate,'strat']
                        performance.loc[year,monthEquiv[month,'Month']]=perf
                        bfirstdone = True
                    else: #no perf
                        performance.loc[year,monthEquiv[month,'Month']]=""
                else:
                    MonthLastday = calendar.MonthLastBusinessDay(datetime.datetime(year))
                    if year == lastDate.year and lastDate.month<month:
                        performance.loc[year,monthEquiv[month,'Month']]=""
                    else:
                        endPeriodPerf = newStrat[newStrat.index<=MonthLastday].tail(1)
                        perf = newStrat.loc[endPeriodPerf,'strat']/newStrat.loc[lastDate,'strat']
                        lastendperiod=endPeriodPerf
                        performance.loc[year,monthEquiv[month,'Month']]=perf
        return performance
    
    def Weekly_perf(self,weekday):
        newStrat=self.newStrat
        # search for the first evaluation date       
        bWeekdayFound = False
        icount = 0
        while bWeekdayFound == False:
            if calendar.isBueinessDay(newStrat.index[icount]):
                day = newStrat.index[icount].weekday()
                if day == weekday:
                    bWeekdayFound = True
                else:
                    icount+=1
            else:
                icount+=1
        Period_begin = newStrat.index[icount]
        newStrat = newStrat[newStrat.index>=newStrat.index[icount]]
        
        calculate_date = []
        perf = []
        
        for k in range(1,len(newStrat)):
            if newStrat.index[k].weekday()==day :
                perf.append(newStrat.loc[newStrat.index[k],'strat']/newStrat.loc[Period_begin,'strat'])
                calculate_date.append(newStrat.index[k])
                Period_begin = newStrat.index[k]
            else:
                if newStrat.index[k]>=Period_begin+datetime.timedelta(+7):
                    Period_begin=newStrat.index[k]+datetime.timedelta(-1)
            dfperf = pd.DateFrame({'Date':calculate_date,'Perf':perf})
            dfperf.set_index('Date',inplace=True)
        return dfperf               
                
    def Describe(self):
        """Display Function"""
        IRR = self.IRR()
        Vol = self.Annualized_Volatility()
        Sharpe = self.Sharpe()
        Sharpe_Cleared = self.Sharpe_Cleared()
        MaxDD = self.Max_DrawDown()
        Daily_Std = self.Daily_Vol()
        dfAnnualPerf = self.Annual_Perf()
        self.newStrat['strat'].plot()
        chaine = 'IRR de  '+str(np.round(100*IRR,4))+"%\nVolatility  "
        chaine+= str(np.round(100*Vol,4))+"%"+"\n"
        chaine+="Daily Volatility  "+ str(np.round(100*Daily_Std,4))+"%"+"\n"
        chaine+="Sharpe Ratio  "+ str(np.round(Sharpe,4))+"\n"
        chaine+="Sortino  "+ str(np.round(Sharpe_Cleared,4))+"\n"
        chaine+="Max Drawdown  "+ str(np.round(100*MaxDD,4))+"%"+"\n"
        print(chaine)
        print("Annual Performance")
        print(dfAnnualPerf)
    
    
    def Save(self,path):
        newStrat = self.newStrat
        irr = self.IRR()
        """stat"""
        vol = self.Annualized_Vol()
        sharpe=self.Sharpe()
        sortino=self.Sharpe_Cleared()
        maxDD = self.Max_DrawDown()
        """monthly perf"""
        dfmonthly_perf = self.Monthly_Perf()
        """reindex"""
        newStrat['Dates_saved']=newStrat.index
        new_index = range(0,len(newStrat))
        
        newStrat.reset_index(new_index,inplace=True)
        new_index= range(0,len(dfmonthly_perf))
        dfmonthly_perf.reset_index(new_index,inplace=True)
        """add statistics""" 
        newStrat["Blanck"]=""
        newStrat["Statistics"]=""
        newStrat["Statistics"][1]="IRR"
        newStrat["Statistics"][2]="Volatility"
        newStrat["Statistics"][3]="Sharpe"
        newStrat["Statistics"][4]="Sortino"
        newStrat["Statistics"][5]="Max Drawdown"
        newStrat["Values"]=""
        newStrat["Values"][1]=irr
        newStrat["Values"][2]=vol
        newStrat["Values"][3]=sharpe
        newStrat["Values"][4]=sortino
        newStrat["Values"][5]=maxDD
        
        """add Monthly perf"""
        newStrat["Blanck2"]=""
        frames = [newStrat,dfmonthly_perf]
        newStrat=pd.concat(frames,axes=1)
        newStrat.set_index('Dates_saved',inplace=True)
        newStrat.to_csv(path)        
        print("Saved.")
        
    """Index transforms"""
    def Leverage(self,Leverage):
        """returns the leveraged index"""
        newStrat = self.newStrat
        newStrat['strat']=Leverage*(newStrat['strat'].pct_change())
        newStrat.loc[newStrat.index[0],'start']=0.0
        newStrat['strat']=newStrat['strat']-1
        newStrat['strat']=100*(newStrat['strat'].cumprod())
        return newStrat
    
    
    def FeesIndex(self,dFees):
        """returns the index including fees"""
        No_fees_index=self.newStrat
        fees_Index=[]
        fees_Index.append(100)
        perf_index=[]
        perf_index.append(0)
        fees_index=[]
        fees_index.append(0)
        for i in range(1,len(No_fees_index)):
            fees = dFees *(No_fees_index[i]-No_fees_index[i-1])
            perf_Index=No_fees_index['strat'][i]/No_fees_index['strat'][i-1]
            fees_Index.append(fees_Index[i-1]*(perf_Index-fees))
        fees_Index=pd.DataFrame({'Dates':No_fees_index.index,'strat':fees_Index})
        fees_Index.set_index('Dates',inplace=True)
        return fees_Index
    
    
    def WithoutFeesIndex(self,dFees):
        """returns the index excluding fees"""
        No_fees_index=self.newStrat
        fees_Index=[]
        fees_Index.append(100)
        perf_index=[]
        perf_index.append(0)
        fees_index=[]
        fees_index.append(0)
        for i in range(1,len(No_fees_index)):
            days = 1#calendar.nbBusinessDaysBetweenTwoDates_exact(No_fees_index.index[i-1],No_fees_index.index[i])
            fees = dFees *days/252
            perf_Index=No_fees_index['strat'][i]/No_fees_index['strat'][i-1]
            fees_Index.append(fees_Index[i-1]*(perf_Index+fees))
        fees_Index=pd.DataFrame({'Dates':No_fees_index.index,'strat':fees_Index})
        fees_Index.set_index('Dates',inplace=True)
        return fees_Index
        
    def RebasedIndex(self,datetime_BeginDate,Initial_Value):
        """returns the index rebased on Begin Date"""
        newStrat = self.newStrat
        newStrat = newStrat[newStrat.index>datetime_BeginDate]
        newStrat['strat']=(newStrat['strat'].pct_change()).fillna(0)+1
        newStrat['strat']=Initial_Value(newStrat['strat'].cumprod())
        return newStrat


newStrat=pd.read_csv(r'/Users//Downloads/data_set.csv')
newStrat.index=newStrat["Date"].map(lambda x : datetime.datetime.strptime(x,"%m/%d/%Y"))
del newStrat['Date']
Strat = Strategy(newStrat.index,newStrat['SPX'])
Strat.Describe()
Strat.IRR()
