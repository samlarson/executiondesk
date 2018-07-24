from alphavantageAPI import AVCall, Preprocess
from model import ModelType
from strategy import Strategy



#new = AVCall()
# daily_aapl = new.getDataDaily("AAPL", "csv")
# weekly_aapl = new.getDataWeeklyAdjusted("AAPL", "csv")
#intraday_aapl = new.getDataIntraDay(symbol='ATHN', interval='15min', datatype='csv')

# new_type = ModelType.LinReg()
# new_type.linear_regression(daily_aapl)

# new_type = ModelType.Graph()
# new_type.simple_graph(weekly_aapl)

# new_type = ModelType.Graph()
# new_type.candlestick(df=intraday_aapl, grain='fine-grain')

new_select = Strategy.Execution()
new_select.sec_selector('simple_NYSE')


