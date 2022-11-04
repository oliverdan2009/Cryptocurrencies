import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

class ticker_data:
    """
    This class works well for all data except hourly data, since yfinance returns equity hourly data on the half hour. 
    Use av_ticker_data script for hourly equity data with the caution that Alpha Vantage only takes five data calls per minute.
    """
    
    def __init__(self, ticker_x: list):
        """Tickers are passed as lists of strings."""
        self.ticker_x = ticker_x
        
        
    def get_data(self, ticker, period = "1mo", interval = "1d", actions = False):
        """Takes ticker as required parameter and returns historical adjusted price data. Uses parameters from yfinance. 
        Actions = False means that dividends and stock split data is not returned. All columns are adjusted prices. WARNING:
        DOESN'T WORK WITH HOURLY Intervals between cryptos and equities due to equities returning on the half hour.
        
        Parameters:
            ticker: str
                Asset tickers that are accepted by yfinance.
            
            period : str
                Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            
            interval : str
                Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                Intraday data cannot extend past 60 days
            
            actions : bool
                Download dividend + stock splits data. Default is False
                
        Returns:
            x_hist : Pandas DataFrame
                Returns OHLC adjusted prices for the the given ticker.
        
        """
        x_tick = yf.Ticker(str(ticker))
        x_hist = x_tick.history(period = period, interval = interval, actions = actions)

        try:
            if x_hist.index.tzinfo is not None:
                x_hist.index = pd.to_datetime(x_hist.index).tz_convert("UCT")
                return x_hist
            else:
                return x_hist
        except:
            print("X_hist.index does not have time zone information (.tzinfo) ")
            return x_hist
        
    
    def get_prices(self, period = "1mo", interval = "1d", actions = False, OHLC = "Close"):
        """
        Returns dataframe of prices.
        
        Parameters:
            period : str
                Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            
            interval : str
                Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                Intraday data cannot extend past 60 days
            
            actions : bool
                Download dividend + stock splits data. Default is False
            
            OHLC : str
                Valid OHLC: Open, High, Low, Close
                
        Returns:
            price_df : Pandas DataFrame
                DataFrame of asset prices indexed by time
                
        """
        
        indicator = None
        
        for ticker in self.ticker_x:
            x_hist = pd.DataFrame(self.get_data(ticker, period = period, interval = interval, actions = actions)[OHLC])
            x_hist.columns = [ticker + "_" + OHLC]
            
            if indicator == None:
                price_df = pd.DataFrame(x_hist)
                indicator = 1
            else:
                price_df = price_df.merge(x_hist, how="inner", left_index=True, right_index=True)
        
        return price_df
    
    
    def get_returns(self, period = "1mo", interval = "1d", actions = False, OHLC = "Close", standardize = False):
        """
        Takes historical data as required parameter. Option to standardize the data.
        
        Parameters:
            period : str
                Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                Default is "1mo"
            
            interval : str
                Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                Intraday data cannot extend last 60 days
                Default is "1d"
            
            actions : bool
                Download dividend + stock splits data. Default is False
            
            OHLC : str
                Valid OHLC: Open, High, Low, Close.
                Default is "Close"
                
            standardize : bool
                Standardize the returns by subtracting the returns by their mean and dividing by their standard deviation.
                Default is False.
                
        Returns:
            x_ret : Pandas DataFrame
                DataFrame of asset returns indexed by time
                
        """
        
        x_hist = self.get_prices(period = period, interval = interval, actions = actions, OHLC = OHLC)
        x_ret = x_hist.pct_change().replace(np.inf, np.nan).dropna()
        
        if standardize == True:
            pipeline = ColumnTransformer([ #ColumnTransformer works on pandas DF, while StandardScaler().fit() takes numpy arrays
                ("pandas num", StandardScaler(), list(x_ret.columns))
            ])
            
            x_stand = pipeline.fit_transform(x_ret)
            return pd.DataFrame(x_stand, index = x_ret.index, columns = list(x_ret.columns))
        
        else:
            return x_ret