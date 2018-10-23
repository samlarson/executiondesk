import os
import io
import sys
import csv
import json
import requests
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from urllib.request import urlopen
from pandas.io.json import json_normalize


class AVCall:
    def __init__(self):
        self.api_key = "FFK3WVR52378LU0Q"
        self.url = "https://www.alphavantage.co/query"

    # Returns intraday time series (timestamp, open, high, low, close, volume) of
    # the equity specified, updated real-time.
    # symbol: the name of the equity of your choice. Example, symbol = MSFT
    # interval: Time interval between two consecutive data points in the time series.
    #           Supports: 1min, 5min, 15min, 30min, 60min
    # datatype: Strings json and csv are accepted with the following specifications:
    #           json returns the time series in JSON format; csv returns the time
    #           series as a CSV (comma separated value) file.
    # api_key:  AlphaVantage API Key
    def getDataIntraDay(self, symbol, interval, datatype):
        data = { "function": "TIME_SERIES_INTRADAY",
                 "interval": interval,
                 "symbol": symbol,
                 "outputsize": "full",
                 "datatype": datatype,
                 "apikey": self.api_key}
        url_response = requests.get(self.url, params=data).content
        df_response = pd.read_csv(io.StringIO(url_response.decode('utf-8')))
        return df_response

    # Returns daily time series (date, daily open, daily high, daily low, daily close,
    # daily volume) of the equity specified, covering up to 20 years of historical data.
    # symbol: the name of the equity of your choice. Example, symbol = MSFT
    # datatype: Strings json and csv are accepted with the following specifications:
    #           json returns the time series in JSON format; csv returns the time
    #           series as a CSV (comma separated value) file.
    # api_key:  AlphaVantage API Key
    def getDataDaily(self, symbol, datatype):
        data = { "function": "TIME_SERIES_DAILY",
                 "symbol": symbol,
                 "datatype": datatype,
                 "apikey": self.api_key}
        url_response =  requests.get(self.url, params = data).content
        df_response = pd.read_csv(io.StringIO(url_response.decode('utf-8')))
        return df_response

    # Returns daily time series (date, daily open, daily high, daily low, daily close,
    # daily volume, daily adjusted close, and split/dividend events) of the equity specified,
    # covering up to 20 years of historical data.
    def getDataDailyAdjusted (self, symbol, datatype):
        data = { "function": "TIME_SERIES_DAILY_ADJUSTED",
                 "symbol": symbol,
                 "datatype": datatype,
                 "apikey": self.api_key}
        url_response = requests.get(self.url, params=data).content
        df_response = pd.read_csv(io.StringIO(url_response.decode('utf-8')))
        return df_response

    # Returns weekly time series (last trading day of each week, weekly open, weekly high,
    # weekly low, weekly close, weekly volume) of the equity specified, covering up to 20
    # years of historical data.
    def getDataWeekly (self, symbol, datatype):
        data = { "function": "TIME_SERIES_WEEKLY",
                 "symbol": symbol,
                 "datatype": datatype,
                 "apikey": self.api_key}
        url_response = requests.get(self.url, params=data).content
        df_response = pd.read_csv(io.StringIO(url_response.decode('utf-8')))
        return df_response

    # Returns weekly adjusted time series (last trading day of each week, weekly open,
    # weekly high, weekly low, weekly close, weekly adjusted close, weekly volume,
    # weekly dividend) of the equity specified, covering up to 20 years of historical data.
    def getDataWeeklyAdjusted (self, symbol, datatype):
        data = { "function": "TIME_SERIES_WEEKLY_ADJUSTED",
                 "symbol": symbol,
                 "datatype": datatype,
                 "apikey": self.api_key}
        url_response = requests.get(self.url, params=data).content
        df_response = pd.read_csv(io.StringIO(url_response.decode('utf-8')))
        return df_response

    # Returns monthly time series (last trading day of each month, monthly open,
    # monthly high, monthly low, monthly close, monthly volume) of the equity
    # specified, covering up to 20 years of historical data.
    def getDataMonthly (self, symbol, datatype):
        data = { "function": "TIME_SERIES_MONTHLY",
                 "symbol": symbol,
                 "datatype": datatype,
                 "apikey": self.api_key}
        url_response = requests.get(self.url, params=data).content
        df_response = pd.read_csv(io.StringIO(url_response.decode('utf-8')))
        return df_response

    # Returns monthly adjusted time series (last trading day of each month, monthly
    # open, monthly high, monthly low, monthly close, monthly adjusted close,
    # monthly volume, monthly dividend) of the equity specified, covering up to
    # 20 years of historical data.
    def getDataMonthlyAdjusted (self, symbol, datatype):
        data = { "function": "TIME_SERIES_MONTHLY_ADJUSTED",
                 "symbol": symbol,
                 "datatype": datatype,
                 "apikey": self.api_key}
        url_response = requests.get(self.url, params=data).content
        df_response = pd.read_csv(io.StringIO(url_response.decode('utf-8')))
        return df_response

    def getMACD (self, symbol, interval, series_type, datatype):
        data = { "function": "MACD",
                 "symbol": symbol,
                 "interval": interval,
                 "series_type": series_type,
                 "datatype": datatype,
                 "apikey": self.api_key}
        url_response = requests.get(self.url, params=data).content
        df_response = pd.read_csv(io.StringIO(url_response.decode('utf-8')))
        return df_response

    def getBBands (self, symbol, interval, time_period, series_type, matype, datatype):
        data = { "function": "BBANDS",
                 "symbol": symbol,
                 "interval": interval,
                 "time_period": time_period,
                 "series_type": series_type,
                 "matype": matype,
                 "datatype": datatype,
                 "apikey": self.api_key}
        url_response = requests.get(self.url, params=data).content
        df_response = pd.read_csv(io.StringIO(url_response.decode('utf-8')))
        return df_response

    def getSMA (self, symbol, interval, time_period, series_type, datatype):
        data = { "function": "SMA",
                 "symbol": symbol,
                 "interval": interval,
                 "time_period": time_period,
                 "series_type": series_type,
                 "datatype": datatype,
                 "apikey": self.api_key}
        url_response = requests.get(self.url, params=data).content
        df_response = pd.read_csv(io.StringIO(url_response.decode('utf-8')))
        return df_response


class Preprocess:

    def concat_df(self, df1, df2):
        if 'time' in df1.columns:
            df1.rename(columns={'time': 'timestamp'}, inplace=True)
        if 'time' in df2.columns:
            df2.rename(columns={'time': 'timestamp'}, inplace=True)

        frame1 = df1.iloc[::-1]
        ts_frame1 = pd.DataFrame(data=frame1)
        ts_frame1['timestamp'] = ts_frame1['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M'))
        frame2 = df2.iloc[::-1]
        ts_frame2 = pd.DataFrame(data=frame2)
        ts_frame2['timestamp'] = ts_frame2['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

        df1['timestamp'] = pd.to_datetime(df1['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        df2['timestamp'] = pd.to_datetime(df2['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

        df = pd.concat(objs=[df1, df2],  axis=1, join='inner', ignore_index=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
        return df

    # Prints contents of csv file onto terminal / IDE.
    # data: csv file returned from any one of the methods above.
    # TODO: this can probably be deleted because we can print from createFrame or any above functions
    def print_data(self, data):
        with requests.Session() as s:
            data_csv = data
            decoded_content = data_csv.content.decode('utf-8')
            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            my_list = list(cr)
            for row in my_list:
                print(row)

    # TODO: handle potential csv/json calls from main method; possibly remove "json_normalize" method on dataframe call
    def create_frame(self, data):
        # json_data = data.text
        # print(json_data)
        # x = json.loads(json_data)
        # # df = pd.DataFrame(pd.io.json.json_normalize(x))
        # df = pd.DataFrame(x)
        # df.drop(df.index[:2], inplace=True)
        # df.drop(df.index[-3:], inplace=True)
        # #print(df["Time Series (Daily)"])
        # #df.plot(x='Time Series (Daily)', y='low', style='o')

        csv_data = data.text
        Preprocess.writeCSV(data, "csv_data.csv")
        df = pd.read_csv("csv_data.csv")
        print(df)

    # TODO: refactor "fieldname" input for dynamic input_data; separate bottom statement into a readCSV function
    # We don't need this to transform API requests to pandas dataframes but we can use it later when we need to write
    # results to csv files
    def writeCSV(self, input_data, out_file):
        if os.path.isfile("csv_data.csv"):
            os.remove("csv_data.csv")

        with open(out_file, "w") as file:
            writer = csv.DictWriter(file, fieldnames=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
            writer.writeheader()
            with open(input_data, "r") as f:
                reader = csv.reader(f)
                writer.writerows({'Date': row[0], 'Open': row[1]} for row in reader)
