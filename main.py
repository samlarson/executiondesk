import pandas as pd
from executiondesk.alphavantageAPI import AVCall, Preprocess
from executiondesk.model import ModelType
from executiondesk.strategy import Strategy
from executiondesk.ml_test import ML



new = AVCall()
sma_tsla = new.getSMA(symbol="TSLA", interval="30min", time_period=200, series_type="open", datatype="csv")
intra_tsla = new.getDataIntraDay(symbol="TSLA", interval="30min", datatype="csv")
# bband_tsla = new.getBBands(symbol="TSLA", interval="30min", time_period=300,
#                            series_type="open", matype=2, datatype="csv")
# daily_aapl = new.getDataDaily("AAPL", "csv")
# weekly_aapl = new.getDataWeeklyAdjusted("AAPL", "csv")
# intraday_aapl = new.getDataIntraDay(symbol='AAPL', interval='15min', datatype='csv')
# macd_aapl = new.getMACD(symbol='AAPL', interval='30min', series_type='open', datatype='csv')
#
# new_type = ModelType.LinReg()
# new_type.linear_regression(daily_aapl)
#
# new_type = ModelType.Graph()
# new_type.simple_graph(weekly_aapl)
#
# new_type = ModelType.Graph()
# new_type.candlestick(df=intraday_aapl, grain='fine-grain')

info = pd.read_csv('tsla_v3_close_WMA.csv', encoding='utf-8')
# new_type = ML()
# new_type.bayes_ridge(info)

new_type = ModelType.Graph()
new_type.candle_bbands(info)

# intra_tsla.to_csv("tesla_intra.csv", encoding='utf-8')
# bband_tsla.to_csv("tsla_bbands_open_V3.csv", encoding='utf-8')


# new_type = ModelType.Graph()
# new_type.macd_graph(macd_aapl)


# new_bands = Strategy.Execution
# new_bands.bollinger(df=intraday_aapl)


# new_select = Strategy.Execution()
# new_select.sec_selector('simple_NYSE')

# new_type = Preprocess()
# new_type.concat_df(sma_tsla, intra_tsla)

