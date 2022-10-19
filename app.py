import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import chart_studio.plotly as plotly
import plotly.figure_factory as ff
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly

st.title('Stcok Price Prediction')

# options of the stocks
dataset = ('MSFT', 'GS', 'GL', 'AAPL', 'META')
option = st.selectbox('Select the stock to predict', dataset)
stock = yf.Ticker(option)
name = stock.info['longName']

# get the years to predict from user
year = st.slider('Year to predcit: ', 1, 3)
period = year * 365

dataLoadState = st.text(f'Loading data of {name}...')
hist = stock.history(period='max')
dataLoadState.text('Loading data... done!')

st.subheader(f'Raw data of {name}')
# st.write(hist)

# plot the close price
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist.index, y=hist['Close'], name="close price", line_color='dimgray'))
fig.layout.update(title_text='Time serials data with Rangeslider (Close Price)',
                  xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# predict using Prophet
data_pred = pd.DataFrame(columns=['ds', 'y'])
startPoint = len(hist.axes[0]) // 6 * 5
data_pred['ds'] = hist.index.tz_localize(None)[startPoint:]
data_pred['y'] = hist['Close'].values[startPoint:]

dataPredState = st.text(f'Predicting...')
m = Prophet()
m.fit(data_pred)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
dataPredState.text('Predicting... done!')

# plot the forecast
fig_pred = plot_plotly(m, forecast)
st.subheader(f'Forecast closing of stock {name} for {year} years')
# st.write(forecast)
st.plotly_chart(fig_pred)
