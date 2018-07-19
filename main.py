from alphavantageAPI import AVCall

new = AVCall()
daily_aapl = new.getDataDaily("AAPL", "csv")
print(daily_aapl)