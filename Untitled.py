import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.stats.diagnostic as diag
import statsmodels.api as sm


st.set_page_config(page_title = "Compunnel digital")
st.image("compunnel.png",width=100)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Demand Forecast App')

def Category_19():
    df = pd.read_csv(r"Historical Product Demand.csv",parse_dates=['Date'])
    index = df[ df['Order_Demand'] <1000 ].index
    df.drop(index,inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    index1 = df[df['Year'] == 2011 ].index
    df.drop(index1,inplace=True)
    index2 = df[df['Year'] == 2017].index
    df.drop(index2,inplace=True)
    df.drop(['Year','Month'],axis=1,inplace=True)
    df.dropna(axis=0, inplace=True)
    q1=df['Order_Demand'].quantile(0.25)
    q2=df['Order_Demand'].quantile(0.50)
    q3=df['Order_Demand'].quantile(0.75)
    iqr=q3-q1
    upper_limit=q3+1.5*iqr
    lower_limit=q1-1.5*iqr
    upper_limit,lower_limit
    def limit_imputer(value):
        if value > upper_limit:
            return upper_limit
        if value < lower_limit:
            a=a+1
            return lower_limit
        else:
            return value
    df['Order_Demand']=df['Order_Demand'].apply(limit_imputer)

    li = ['Category_019','Category_006','Category_028','Category_005','Category_007']
    df19 = df[df.Product_Category==li[0]]
    df19= df19.groupby('Date')['Order_Demand'].count().reset_index()
    df19 = df19.set_index(['Date'])
    df19= df19['Order_Demand'].resample('MS').mean()
    df19 = df19.fillna(df19.bfill())
    df_19=df19.to_frame()
    decomposition = sm.tsa.seasonal_decompose(df_19, model='multiplicative')
    model=sm.tsa.statespace.SARIMAX(df19,order=(1,1,1),seasonal_order=(1,1,0,12))
    results=model.fit()
    pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=True)
    pred_ci = pred.conf_int()
    pred_uc = results.get_forecast(steps = N_Month)
    pred_ci = pred_uc.conf_int()
    ax = df19.plot(label='observed', figsize=(16, 8))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Order_Demand')
    plt.show()
    plt.legend()
    st.pyplot()
    FORECAST_19 = results.forecast(steps = N_Month)
    FORECAST_19=FORECAST_19.to_frame()
    inventory_management_list_19 = FORECAST_19['predicted_mean'].tolist()
    stock=0
    refill_list=[]
    balanced_stock=[]
    order_placed=[]
    extra_order_for_refill=[]
    for x in inventory_management_list_19:
        if stock<=(x*1.2):
            Extra_order=(x*1.2)-stock
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x#balancedStock
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
        else:
            Extra_order=0
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
    df = pd.DataFrame(list(zip(inventory_management_list_19,extra_order_for_refill,refill_list,order_placed,balanced_stock)), columns =['order_demand','Refill_0rder','refill_list','order','balanced'],index=FORECAST_19.index)
    st.write(df)
    mae = np.mean(np.abs(results.resid))
    st.write('MAE: %.3f' % mae)
    y_forecasted = pred.predicted_mean
    y_truth = df19['2016-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    st.write('MSE {}'.format(round(mse, 2)))
    st.write('RMSE: {}'.format(round(np.sqrt(mse), 2)))
    return(df,ax,FORECAST_19)

def Category_06():
    df = pd.read_csv(r"Historical Product Demand.csv",parse_dates=['Date'])
    index = df[ df['Order_Demand'] <1000 ].index
    df.drop(index,inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    index1 = df[df['Year'] == 2011 ].index
    df.drop(index1,inplace=True)
    index2 = df[df['Year'] == 2017].index
    df.drop(index2,inplace=True)
    df.drop(['Year','Month'],axis=1,inplace=True)
    df.dropna(axis=0, inplace=True)
    q1=df['Order_Demand'].quantile(0.25)
    q2=df['Order_Demand'].quantile(0.50)
    q3=df['Order_Demand'].quantile(0.75)
    iqr=q3-q1
    upper_limit=q3+1.5*iqr
    lower_limit=q1-1.5*iqr
    upper_limit,lower_limit
    def limit_imputer(value):
        if value > upper_limit:
            return upper_limit
        if value < lower_limit:
            a=a+1
            return lower_limit
        else:
            return value
    df['Order_Demand']=df['Order_Demand'].apply(limit_imputer)

    li = ['Category_019','Category_006','Category_028','Category_005','Category_007']
    df06 = df[df.Product_Category==li[1]]
    df06= df06.groupby('Date')['Order_Demand'].count().reset_index()
    df06 = df06.set_index(['Date'])
    df06= df06['Order_Demand'].resample('MS').mean()
    df06 = df06.fillna(df06.bfill())
    df_06=df06.to_frame()
    decomposition = sm.tsa.seasonal_decompose(df_06, model='multiplicative')
    model=sm.tsa.statespace.SARIMAX(df06,order=(1,1,1),seasonal_order=(1,1,0,12))
    results=model.fit()
    
    pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=True)
    pred_ci = pred.conf_int()
    pred_uc = results.get_forecast(steps = N_Month)
    pred_ci = pred_uc.conf_int()
    ax = df06.plot(label='observed', figsize=(16, 8))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Order_Demand')
    plt.show()
    plt.legend()
    st.pyplot()
    FORECAST_06 = results.forecast(steps = N_Month)
    FORECAST_06=FORECAST_06.to_frame()
    inventory_management_list_06 = FORECAST_06['predicted_mean'].tolist()
    stock=0
    refill_list=[]
    balanced_stock=[]
    order_placed=[]
    extra_order_for_refill=[]
    for x in inventory_management_list_06:
        if stock<=(x*1.2):
            Extra_order=(x*1.2)-stock
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x#balancedStock
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
        else:
            Extra_order=0
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
    df = pd.DataFrame(list(zip(inventory_management_list_06,extra_order_for_refill,refill_list,order_placed,balanced_stock)), columns =['order_demand','Refill_0rder','refill_list','order','balanced'],index=FORECAST_06.index)
    st.write(df)
    mae = np.mean(np.abs(results.resid))
    st.write('MAE: %.3f' % mae)
    y_forecasted = pred.predicted_mean
    y_truth = df06['2016-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    st.write('MSE {}'.format(round(mse, 2)))
    st.write('RMSE: {}'.format(round(np.sqrt(mse), 2)))
    return(df,ax,FORECAST_06)

def Category_28():
    df = pd.read_csv(r"Historical Product Demand.csv",parse_dates=['Date'])
    index = df[ df['Order_Demand'] <1000 ].index
    df.drop(index,inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    index1 = df[df['Year'] == 2011 ].index
    df.drop(index1,inplace=True)
    index2 = df[df['Year'] == 2017].index
    df.drop(index2,inplace=True)
    df.drop(['Year','Month'],axis=1,inplace=True)
    df.dropna(axis=0, inplace=True)
    q1=df['Order_Demand'].quantile(0.25)
    q2=df['Order_Demand'].quantile(0.50)
    q3=df['Order_Demand'].quantile(0.75)
    iqr=q3-q1
    upper_limit=q3+1.5*iqr
    lower_limit=q1-1.5*iqr
    upper_limit,lower_limit
    def limit_imputer(value):
        if value > upper_limit:
            return upper_limit
        if value < lower_limit:
            a=a+1
            return lower_limit
        else:
            return value
    df['Order_Demand']=df['Order_Demand'].apply(limit_imputer)


    li = ['Category_019','Category_006','Category_028','Category_005','Category_007']
    df28 = df[df.Product_Category==li[2]]
    df28= df28.groupby('Date')['Order_Demand'].count().reset_index()
    df28 = df28.set_index(['Date'])
    df28= df28['Order_Demand'].resample('MS').mean()
    df28 = df28.fillna(df28.bfill())
    df_28=df28.to_frame()
    decomposition = sm.tsa.seasonal_decompose(df_28, model='multiplicative')
    model=sm.tsa.statespace.SARIMAX(df28,order=(1,1,1),seasonal_order=(1,1,0,12))
    results=model.fit()
    
    pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=True)
    pred_ci = pred.conf_int()
    pred_uc = results.get_forecast(steps = N_Month)
    pred_ci = pred_uc.conf_int()
    ax = df28.plot(label='observed', figsize=(16, 8))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Order_Demand')
    plt.show()
    plt.legend()
    st.pyplot()
    FORECAST_28 = results.forecast(steps = N_Month)
    FORECAST_28=FORECAST_28.to_frame()
    inventory_management_list_28 = FORECAST_28['predicted_mean'].tolist()
    stock=0
    refill_list=[]
    balanced_stock=[]
    order_placed=[]
    extra_order_for_refill=[]
    for x in inventory_management_list_28:
        if stock<=(x*1.2):
            Extra_order=(x*1.2)-stock
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x#balancedStock
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
        else:
            Extra_order=0
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
    df = pd.DataFrame(list(zip(inventory_management_list_28,extra_order_for_refill,refill_list,order_placed,balanced_stock)), columns =['order_demand','Refill_0rder','refill_list','order','balanced'],index=FORECAST_28.index)
    st.write(df)
    mae = np.mean(np.abs(results.resid))
    st.write('MAE: %.3f' % mae)
    y_forecasted = pred.predicted_mean
    y_truth = df28['2016-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    st.write('MSE {}'.format(round(mse, 2)))
    st.write('RMSE: {}'.format(round(np.sqrt(mse), 2)))
    return(df,ax,FORECAST_28)
  
def Category_05():
    df = pd.read_csv(r"Historical Product Demand.csv",parse_dates=['Date'])
    index = df[ df['Order_Demand'] <1000 ].index
    df.drop(index,inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    index1 = df[df['Year'] == 2011 ].index
    df.drop(index1,inplace=True)
    index2 = df[df['Year'] == 2017].index
    df.drop(index2,inplace=True)
    df.drop(['Year','Month'],axis=1,inplace=True)
    df.dropna(axis=0, inplace=True)
    q1=df['Order_Demand'].quantile(0.25)
    q2=df['Order_Demand'].quantile(0.50)
    q3=df['Order_Demand'].quantile(0.75)
    iqr=q3-q1
    upper_limit=q3+1.5*iqr
    lower_limit=q1-1.5*iqr
    upper_limit,lower_limit
    def limit_imputer(value):
        if value > upper_limit:
            return upper_limit
        if value < lower_limit:
            a=a+1
            return lower_limit
        else:
            return value
    df['Order_Demand']=df['Order_Demand'].apply(limit_imputer)


    li = ['Category_019','Category_006','Category_028','Category_005','Category_007']
    df05 = df[df.Product_Category==li[3]]
    df05= df05.groupby('Date')['Order_Demand'].count().reset_index()
    df05 = df05.set_index(['Date'])
    df05= df05['Order_Demand'].resample('MS').mean()
    df05 = df05.fillna(df05.bfill())
    df_05=df05.to_frame()
    decomposition = sm.tsa.seasonal_decompose(df_05, model='multiplicative')
    model=sm.tsa.statespace.SARIMAX(df05,order=(1,1,1),seasonal_order=(1,1,0,12))
    results=model.fit()
    
    pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=True)
    pred_ci = pred.conf_int()
    pred_uc = results.get_forecast(steps = N_Month)
    pred_ci = pred_uc.conf_int()
    ax = df05.plot(label='observed', figsize=(16, 8))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Order_Demand')
    plt.show()
    plt.legend()
    st.pyplot()
    FORECAST_05 = results.forecast(steps = N_Month)
    FORECAST_05=FORECAST_05.to_frame()
    inventory_management_list_05 = FORECAST_05['predicted_mean'].tolist()
    stock=0
    refill_list=[]
    balanced_stock=[]
    order_placed=[]
    extra_order_for_refill=[]
    for x in inventory_management_list_05:
        if stock<=(x*1.2):
            Extra_order=(x*1.2)-stock
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x#balancedStock
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
        else:
            Extra_order=0
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
    df = pd.DataFrame(list(zip(inventory_management_list_05,extra_order_for_refill,refill_list,order_placed,balanced_stock)), columns =['order_demand','Refill_0rder','refill_list','order','balanced'],index=FORECAST_05.index)
    st.write(df)
    mae = np.mean(np.abs(results.resid))
    st.write('MAE: %.3f' % mae)
    y_forecasted = pred.predicted_mean
    y_truth = df05['2016-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    st.write('MSE {}'.format(round(mse, 2)))
    st.write('RMSE: {}'.format(round(np.sqrt(mse), 2)))
    return(df,ax,FORECAST_05)
  

def Category_07():
    df = pd.read_csv(r"Historical Product Demand.csv",parse_dates=['Date'])
    index = df[ df['Order_Demand'] <1000 ].index
    df.drop(index,inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    index1 = df[df['Year'] == 2011 ].index
    df.drop(index1,inplace=True)
    index2 = df[df['Year'] == 2017].index
    df.drop(index2,inplace=True)
    df.drop(['Year','Month'],axis=1,inplace=True)
    df.dropna(axis=0, inplace=True)
    q1=df['Order_Demand'].quantile(0.25)
    q2=df['Order_Demand'].quantile(0.50)
    q3=df['Order_Demand'].quantile(0.75)
    iqr=q3-q1
    upper_limit=q3+1.5*iqr
    lower_limit=q1-1.5*iqr
    upper_limit,lower_limit
    def limit_imputer(value):
        if value > upper_limit:
            return upper_limit
        if value < lower_limit:
            a=a+1
            return lower_limit
        else:
            return value
    df['Order_Demand']=df['Order_Demand'].apply(limit_imputer)


    li = ['Category_019','Category_006','Category_028','Category_005','Category_007']
    df07 = df[df.Product_Category==li[4]]
    df07= df07.groupby('Date')['Order_Demand'].count().reset_index()
    df07 = df07.set_index(['Date'])
    df07= df07['Order_Demand'].resample('MS').mean()
    df07 = df07.fillna(df07.bfill())
    df_07=df07.to_frame()
    decomposition = sm.tsa.seasonal_decompose(df_07, model='multiplicative')
    model=sm.tsa.statespace.SARIMAX(df07,order=(1,1,1),seasonal_order=(1,1,0,12))
    results=model.fit()
    
    pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=True)
    pred_ci = pred.conf_int()
    pred_uc = results.get_forecast(steps = N_Month)
    pred_ci = pred_uc.conf_int()
    ax = df07.plot(label='observed', figsize=(16, 8))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Order_Demand')
    plt.show()
    plt.legend()
    st.pyplot()
    FORECAST_07 = results.forecast(steps = N_Month)
    FORECAST_07=FORECAST_07.to_frame()
    inventory_management_list_07 = FORECAST_07['predicted_mean'].tolist()
    stock=0
    refill_list=[]
    balanced_stock=[]
    order_placed=[]
    extra_order_for_refill=[]
    for x in inventory_management_list_07:
        if stock<=(x*1.2):
            Extra_order=(x*1.2)-stock
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x#balancedStock
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
        else:
            Extra_order=0
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
    df = pd.DataFrame(list(zip(inventory_management_list_07,extra_order_for_refill,refill_list,order_placed,balanced_stock)), columns =['order_demand','Refill_0rder','refill_list','order','balanced'],index=FORECAST_07.index)
    st.write(df)
    mae = np.mean(np.abs(results.resid))
    st.write('MAE: %.3f' % mae)
    y_forecasted = pred.predicted_mean
    y_truth = df07['2016-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    st.write('MSE {}'.format(round(mse, 2)))
    st.write('RMSE: {}'.format(round(np.sqrt(mse), 2)))
    return(df,ax,FORECAST_07)
  

def All_Category():
    df = pd.read_csv(r"Historical Product Demand.csv",parse_dates=['Date'])
    index = df[ df['Order_Demand'] <1000 ].index
    df.drop(index,inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    index1 = df[df['Year'] == 2011 ].index
    df.drop(index1,inplace=True)
    index2 = df[df['Year'] == 2017].index
    df.drop(index2,inplace=True)
    df.drop(['Year','Month'],axis=1,inplace=True)
    df.dropna(axis=0, inplace=True)
    q1=df['Order_Demand'].quantile(0.25)
    q2=df['Order_Demand'].quantile(0.50)
    q3=df['Order_Demand'].quantile(0.75)
    iqr=q3-q1
    upper_limit=q3+1.5*iqr
    lower_limit=q1-1.5*iqr
    upper_limit,lower_limit
    def limit_imputer(value):
        if value > upper_limit:
            return upper_limit
        if value < lower_limit:
            a=a+1
            return lower_limit
        else:
            return value
    df['Order_Demand']=df['Order_Demand'].apply(limit_imputer)    
    li = ['Category_019','Category_006','Category_028','Category_005','Category_007']
    alldf=df[df.Product_Category.isin(li)]
    alldf = alldf.set_index('Date')
    alldf= alldf.groupby('Date')['Order_Demand'].count().reset_index()
    alldf = alldf.set_index(['Date'])
    alldf= alldf['Order_Demand'].resample('MS').mean()
    alldf = alldf.fillna(alldf.bfill())
    df_alldf =alldf.to_frame()
    decomposition = sm.tsa.seasonal_decompose(df_alldf, model='multiplicative')
    model=sm.tsa.statespace.SARIMAX(alldf,order=(1,1,1),seasonal_order=(1,1,0,12))
    results=model.fit()
    
    pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=True)
    pred_ci = pred.conf_int()
    pred_uc = results.get_forecast(steps = N_Month)
    pred_ci = pred_uc.conf_int()
    ax = alldf.plot(label='observed', figsize=(16, 8))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Order_Demand')
    plt.show()
    plt.legend()
    st.pyplot()
    FORECAST_alldf = results.forecast(steps = N_Month)
    FORECAST_alldf=FORECAST_alldf.to_frame()
    inventory_management_list_alldf = FORECAST_alldf['predicted_mean'].tolist()
    stock=0
    refill_list=[]
    balanced_stock=[]
    order_placed=[]
    extra_order_for_refill=[]
    for x in inventory_management_list_alldf:
        if stock<=(x*1.2):
            Extra_order=(x*1.2)-stock
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x#balancedStock
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
        else:
            Extra_order=0
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
    df = pd.DataFrame(list(zip(inventory_management_list_alldf,extra_order_for_refill,refill_list,order_placed,balanced_stock)), columns =['order_demand','Refill_0rder','refill_list','order','balanced'],index=FORECAST_alldf.index)
    st.write(df)
    mae = np.mean(np.abs(results.resid))
    st.write('MAE: %.3f' % mae)
    y_forecasted = pred.predicted_mean
    y_truth = alldf['2016-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    st.write('MSE {}'.format(round(mse, 2)))
    st.write('RMSE: {}'.format(round(np.sqrt(mse), 2)))
    return(df,ax,FORECAST_alldf)
    
    
if st.button('Top5_Category'):
    N_Month = int(st.text_input(" Input Forecast Months ", 24))
    All_Category()

if st.button('Category_07'):
    N_Month = int(st.text_input(" Input Forecast Months ", 24))
    Category_07()
if st.button('Category_05'):
    N_Month = int(st.text_input(" Input Forecast Months ", 24))
    Category_05()
if st.button('Category_28'):
    N_Month = int(st.text_input(" Input Forecast Months ", 24))
    Category_28()
if st.button('Category_19'):
    N_Month = int(st.text_input(" Input Forecast Months ", 24))
    Category_19()
if st.button('Category_06'):
    N_Month = int(st.text_input(" Input Forecast Months ", 24))
    Category_06()
