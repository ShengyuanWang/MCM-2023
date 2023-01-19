# 利用BIC最小的模型作为识别的依据：
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


sales=pd.read_sas('sales_monthly.sas7bdat')
sales.head(12)
#设置一下时间索引，这一步很有必要，否则后面程序会报错
sales.index=pd.Index(pd.date_range('1/2001','9/2008',freq='1M'))

#判断需要进行几阶差分
sales['Sales'].plot()
sales['Sales'].diff(1).plot() #可以看出1阶差分后是一个零均值化的平稳序列,第一个数是NaN
order_p,order_q,bic=[],[],[]
model_order=pd.DataFrame()
for p in range(4):
    for q in range(4):
        arma_model=sm.tsa.ARMA(sales['Sales'].diff(1).iloc[1:92].dropna(),
                               (p,q)).fit()
        order_p.append(p)
        order_q.append(q)
        bic.append(arma_model.bic)
        print('The BIC of ARMA(%s,%s) is %s'%(p,q,arma_model.bic))

model_order['p']=order_p
model_order['q']=order_q
model_order['BIC']=bic
P=list(model_order['p'][model_order['BIC']==model_order['BIC'].min()])
Q=list(model_order['q'][model_order['BIC']==model_order['BIC'].min()])
print('\n最好的模型是ARMA(%s,%s)' %(P[0],Q[0]))


# 单位根检验(原假设H0：时间序列具有单位根（不平稳）)：
from statsmodels.tsa.stattools import adfuller
def DFTest(sales,regression, maxlag,autolag='AIC'):
    print("ADF-Test Result:")
    dftest=adfuller(sales,regression=regression,maxlag=maxlag, autolag=autolag)

    dfoutput=pd.Series(dftest[0:4],
                   index=['Test Statistic','P-value',
                          'Lags Used','nobs'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value at %s'%key]=value
    print(dfoutput)

# regression可以根据模型形式指定为：
# ‘c'：仅常数，默认；’ct‘：常数和长期趋势；’ctt’：常数、线性和二次曲线趋势；‘nc’：无常数趋势。

DFTest(sales['Sales'],regression='nc', maxlag=6,autolag='AIC')  #对原始数据进行单位根检验

print(37*'-')
DFTest(sales['Sales'].diff(1).iloc[1:92], regression='nc', maxlag=5,autolag='AIC')
# 结论：一阶差分后平稳。
