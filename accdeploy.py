import streamlit as st
import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima import model_selection
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from math import sqrt

header = st.beta_container()
dataset = st.beta_container()
modelling = st.beta_container()

@st.cache
def get_data(filepath):
	df = pd.read_excel(filepath)
	return df

with header:
	st.title('Accessories Forecasting')

	code = pd.read_excel('dataset/code.xlsx', index_col=0, engine='openpyxl')
	st.write(code)

with dataset:
	df = get_data('dataset/Car_Decors_Data_Rev2_13062021.xlsx', engine='openpyxl')
	#st.write(df.head(24))

with modelling:
	sel_col, dis_col = st.beta_columns(2)

	access = sel_col.selectbox('Choose Accessories Code', options=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], index=0)
	period = sel_col.selectbox('Choose Forecast Period (days)', options=[7,14,21,28,30], index=0)

	df2 = df.iloc[:,[access]]
	df2.columns = ['y']

	model = pm.auto_arima(df2.y, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, n=1, d=None, seasonal=True,
                      error_action='ignore', suppress_warning=True, stepwise=True)
	model.fit(df2.y)
	forecast = model.predict(n_periods=len(df2))
	forecast = pd.DataFrame(forecast, columns=['yhat'])
	forecast['yhat'] = forecast['yhat'] + 0.15
	forecast = forecast.round({'yhat':1}).round({'yhat':0})
	forecastunit = int(model.predict(n_periods=period).sum())

	y_actual = df2.y
	y_predicted = forecast.yhat
	rmse = round(sqrt(mean_squared_error(y_actual, y_predicted)),2)

	st.title('Sales Forecast')

	st.write('Forecasted sales of', df.columns[access], 'for the next', period, 'days are', forecastunit, 'units')
	st.write('Model Accuracy:', round(accuracy_score(y_actual, y_predicted)*100,2), '%')
	#st.write('RMSE          :', rmse)
	






