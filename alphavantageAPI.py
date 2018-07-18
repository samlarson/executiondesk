import requests
import csv

api_key = "FFK3WVR52378LU0Q"

# Returns intraday time series (timestamp, open, high, low, close, volume) of
# the equity specified, updated realtime.

# symbol: the name of the equity of your choice. Example, symbol = MSFT
# interval: Time interval between two consecutive data points in the time series.
#           Supports: 1min, 5min, 15min, 30min, 60min
# datatype: Strings json and csv are accepted with the following specifications:
#           json returns the time series in JSON format; csv returns the time
#           series as a CSV (comma seperated value) file.
# api_key:  AlphaVantage API Key

def getDataIntraDay (symbol, interval, datatype, api_key):
    data = { "function": "TIME_SERIES_INTRADAY",
             "interval":interval,
             "symbol": symbol,
             "datatype": datatype,
             "apikey": api_key}
    return requests.get("https://www.alphavantage.co/query", params = data)

# Returns daily time series (date, daily open, daily high, daily low, daily close,
# daily volume) of the equity specified, covering up to 20 years of historical data.

# symbol: the name of the equity of your choice. Example, symbol = MSFT
# datatype: Strings json and csv are accepted with the following specifications:
#           json returns the time series in JSON format; csv returns the time
#           series as a CSV (comma seperated value) file.
# api_key:  AlphaVantage API Key

def getDataDaily (symbol, datatype, api_key):
    data = { "function": "TIME_SERIES_DAILY",
             "symbol": symbol,
             "datatype": datatype,
             "apikey": api_key}
    return requests.get("https://www.alphavantage.co/query", params = data)

# Returns daily time series (date, daily open, daily high, daily low, daily close,
# daily volume, daily adjusted close, and split/dividend events) of the equity specified,
# covering up to 20 years of historical data.

# symbol: the name of the equity of your choice. Example, symbol = MSFT
# datatype: Strings json and csv are accepted with the following specifications:
#           json returns the time series in JSON format; csv returns the time
#           series as a CSV (comma seperated value) file.
# api_key:  AlphaVantage API Key

def getDataDailyAdjusted (symbol, datatype, api_key):
    data = { "function": "TIME_SERIES_DAILY_ADJUSTED",
             "symbol": symbol,
             "datatype": datatype,
             "apikey": api_key}
    return requests.get("https://www.alphavantage.co/query", params = data)

# Returns weekly time series (last trading day of each week, weekly open, weekly high,
# weekly low, weekly close, weekly volume) of the equity specified, covering up to 20
# years of historical data.

# symbol: the name of the equity of your choice. Example, symbol = MSFT
# datatype: Strings json and csv are accepted with the following specifications:
#           json returns the time series in JSON format; csv returns the time
#           series as a CSV (comma seperated value) file.
# api_key:  AlphaVantage API Key

def getDataWeekly (symbol, datatype, api_key):
    data = { "function": "TIME_SERIES_WEEKLY",
             "symbol": symbol,
             "datatype": datatype,
             "apikey": api_key}
    return requests.get("https://www.alphavantage.co/query", params = data)

# Returns weekly adjusted time series (last trading day of each week, weekly open,
# weekly high, weekly low, weekly close, weekly adjusted close, weekly volume,
# weekly dividend) of the equity specified, covering up to 20 years of historical data.

# symbol: the name of the equity of your choice. Example, symbol = MSFT
# datatype: Strings json and csv are accepted with the following specifications:
#           json returns the time series in JSON format; csv returns the time
#           series as a CSV (comma seperated value) file.
# api_key:  AlphaVantage API Key

def getDataWeeklyAdjusted (symbol, datatype, api_key):
    data = { "function": "TIME_SERIES_WEEKLY_ADJUSTED",
             "symbol": symbol,
             "datatype": datatype,
             "apikey": api_key}
    return requests.get("https://www.alphavantage.co/query", params = data)

# Returns monthly time series (last trading day of each month, monthly open,
# monthly high, monthly low, monthly close, monthly volume) of the equity
# specified, covering up to 20 years of historical data.

# symbol: the name of the equity of your choice. Example, symbol = MSFT
# datatype: Strings json and csv are accepted with the following specifications:
#           json returns the time series in JSON format; csv returns the time
#           series as a CSV (comma seperated value) file.
# api_key:  AlphaVantage API Key

def getDataMonthly (symbol, datatype, api_key):
    data = { "function": "TIME_SERIES_MONTHLY",
             "symbol": symbol,
             "datatype": datatype,
             "apikey": api_key}
    return requests.get("https://www.alphavantage.co/query", params = data)

# Returns monthly adjusted time series (last trading day of each month, monthly
# open, monthly high, monthly low, monthly close, monthly adjusted close,
# monthly volume, monthly dividend) of the equity specified, covering up to
# 20 years of historical data.

# symbol: the name of the equity of your choice. Example, symbol = MSFT
# datatype: Strings json and csv are accepted with the following specifications:
#           json returns the time series in JSON format; csv returns the time
#           series as a CSV (comma seperated value) file.
# api_key:  AlphaVantage API Key

def getDataMonthlyAdjusted (symbol, datatype, api_key):
    data = { "function": "TIME_SERIES_MONTHLY_ADJUSTED",
             "symbol": symbol,
             "datatype": datatype,
             "apikey": api_key}
    return requests.get("https://www.alphavantage.co/query", params = data)

# Prints contents of csv file onto terminal / IDE.

# data: csv file returned from any one of the methods above.

def printData (data):
    with requests.Session() as s:
        data_csv = data

        decoded_content = data_csv.content.decode('utf-8')

        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
        for row in my_list:
            print(row)

#---TEST METHOD---#

dailyAAPL = getDataDaily("AAPL", "csv", api_key)
printData(dailyAAPL)
