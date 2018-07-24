import numpy as np
import pandas as pd
import sklearn


#TODO: add function to iterate through list/object of securities and implement strategy and recommendation
class Strategy:

    class Execution:
        # TODO: check market_type, set sec_list, loop to make api request for each security, make rec based on strategy
        def sec_selector(self, market_type):
            if market_type == 'simple_NYSE':
                sec_list = ['AAPL', 'MSFT']
                print(type(sec_list))
            if market_type == 'NSE':
                sec_list = []
            if market_type == 'ETF':
                sec_list = []