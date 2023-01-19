#加载数据
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

#ARIMA模型分析：
#先把ACF图和PACF图画出来看看：
fig=plt.figure(figsize=(10,6))
ax1=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(sales['Sales'].diff(1).iloc[1:92].dropna(),lags=24,ax=ax1) # 注意：要去掉第1个空值                             
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(sales['Sales'].diff(1).iloc[1:92].dropna(),  lags=24,ax=ax2)# 注意：要去掉第1个空值
#判断：ACF图在2之后截尾，而PACF拖尾。模型可以由MA(2)→ARIMA(0,1,2).

#建立ARIMA模型
model=sm.tsa.ARMA(sales['Sales'].diff(1).iloc[1:92].dropna(),(0,2)).fit(method='css') #使用最小二乘，‘mle’是极大似然估计
#画图比较一下预测值和真实观测值之间的关系
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)
ax.plot(sales['Sales'].diff(1).iloc[1:92],color='blue',label='Sales')
ax.plot(model.fittedvalues,color='red',label='Predicted Sales')
plt.legend(loc='lower right')

#最后，把预测值还原为原始数据的形式，预测值是差分数值，需要转化
def forecast(step,var,modelname):
    diff=list(modelname.predict(len(var)-1,len(var)-1+step,dynamic=True))
    prediction=[]
    prediction.append(var[len(var)-1])
    seq=[]
    seq.append(var[len(var)-1])
    seq.extend(diff)
    for i in range(step):
        v=prediction[i]+seq[i+1]
        prediction.append(v)

    prediction=pd.DataFrame({'Predicted Sales':prediction})
    return prediction[1:]  #第一个值是原序列最后一个值，故第二个值是预测值。
forecast(4,sales['Sales'],model)
