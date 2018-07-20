import numpy as np
import pandas as pd
import seaborn as sb
import datetime as dt
import plotly.plotly as py
import plotly.graph_objs as go
from datetime import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.offline as offline


class ModelType:

    # Returns and Ordinary Least Squares regression analysis of 2 variables
    class LinReg:
        def linear_regression(self, df):
            frame = df.iloc[::-1]
            # frame = df.reindex(index=df.index[::-1])
            frame['timestamp'] = frame['timestamp'].astype('datetime64[D]')
            x, y = frame['volume'], frame['close']

            if type(frame['timestamp'].iloc[0]) != pd.Timestamp:
                raise ValueError("Please check the dtype of timestamp field")

            model = sm.OLS(y, x).fit()
            predict = model.predict(x)
            # print(predict)
            print(model.summary())
            # print(isinstance(x, datetime.date))
            # print(frame.columns.values)
            sb.set(color_codes=True)
            sb.distplot(x)
            plt.show()
            sb.distplot(y)
            plt.show()


    # Plots a simple 2D graph of the specified columns
    #TODO: replace seaborn dependencies with plotly
    #TODO: consolidate dataframe processing steps in single method
    #TODO: resolve historical slice error (only displaying '00-'02)
    #TODO: add OHLC graph
    class Graph:
        def simple_graph(self, df):
            frame = df.iloc[::-1]
            ts_frame = pd.DataFrame(data=frame)
            ts_frame['timestamp'] = ts_frame['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
            ts_frame.index = range(len(ts_frame))
            # frame['timestamp'] = pd.to_datetime(frame['timestamp'], format='%Y-%m-%d')

            # print(type(ts_frame['timestamp']))
            # print(ts_frame['timestamp'])
            # print(type(ts_frame['timestamp'].iloc[0]))

            start_date = ts_frame['timestamp'].iloc[0]
            end_date = ts_frame['timestamp'].iloc[-1]
            date_slice = pd.date_range(start=start_date, end=end_date)
            ts_frame.set_index('timestamp', inplace=True)

            y = ts_frame['open']
            ax = y.plot()
            ticklabels = ts_frame.index
            ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            plt.show()


        # Creates a simple candlestick plot for a given security
        # Currently the only method from avAPI.py that needs to be called with fine-grain is intraday requests
        def candlestick(self, df, grain):
            frame = df.iloc[::-1]
            ts_frame = pd.DataFrame(data=frame)

            if grain == "low-grain":
                ts_frame['timestamp'] = ts_frame['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
            if grain == "fine-grain":
                ts_frame['timestamp'] = ts_frame['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            else:
                raise KeyError("Timestamp format could not be converted to datetime object, check grain parameter")

            ts_frame.index = range(len(ts_frame))
            start_date = ts_frame['timestamp'].iloc[0]
            end_date = ts_frame['timestamp'].iloc[-1]
            date_slice = pd.date_range(start=start_date, end=end_date, freq='H')
            ts_frame.set_index('timestamp', inplace=True)

            trace = go.Candlestick(x=date_slice,
                                   open=ts_frame['open'],
                                   high=ts_frame['high'],
                                   low=ts_frame['low'],
                                   close=ts_frame['close'])
            layout = go.Layout(
                title='Candlestick Graph',
                xaxis=go.layout.XAxis(
                    title='Time',
                    tickmode='array',
                    tickvals=list(range(len(ts_frame.index))),
                    ticktext=ts_frame.index),
                yaxis=go.layout.YAxis(
                    title='Price'))

            data = [trace]
            fig = go.Figure(data=data, layout=layout)
            offline.plot(fig, filename='graph.html')
            #print(ts_frame)


        def multi_scatter(self, df, grain):
            frame = df.iloc[::-1]
            ts_frame = pd.DataFrame(data=frame)

            if grain == "low-grain":
                ts_frame['timestamp'] = ts_frame['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
            if grain == "fine-grain":
                ts_frame['timestamp'] = ts_frame['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            else:
                raise KeyError("Timestamp format could not be converted to datetime object, check grain parameter")

            ts_frame.index = range(len(ts_frame))
            start_date = ts_frame['timestamp'].iloc[0]
            end_date = ts_frame['timestamp'].iloc[-1]
            date_slice = pd.date_range(start=start_date, end=end_date)
            ts_frame.set_index('timestamp', inplace=True)

            trace1 = go.Scatter(
                x=date_slice,
                y=ts_frame['open'],
                name='open',
            )

            trace2 = go.Scatter(
                x=date_slice,
                y=ts_frame['volume'],
                name='volume',
                yaxis='y2'
            )
            layout = go.Layout(
                title='Multi-Scatter Plot',
                xaxis=dict(
                    rangeslider=dict(
                        visible=False
                    )
                ),
                yaxis=dict(
                    title='Price ($)'
                ),
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    type='log',
                    side='right'
                )
            )

            plot_data = [trace1, trace2]
            fig = go.Figure(data=plot_data, layout=layout)
            offline.plot(fig, filename='multiscatter.html')
