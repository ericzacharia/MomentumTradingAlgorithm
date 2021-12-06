import numpy as np
from collections import deque
from datetime import date, datetime
from AlgorithmImports import *

class SlidingAverage:
    def __init__(self, window_size=365, com=0.005):
        self.window_size = window_size
        self.volume_lst = np.zeros((self.window_size))
        self.price_lst = np.zeros((self.window_size))
        self.queue = deque([])
        self.weights = np.array(
            [(1 + com) ** i for i in range(self.window_size)])

    def insert_data(self, bar):
        if len(self.queue) >= self.window_size:
            self.volume_lst = np.roll(self.volume_lst, -1)
            self.price_lst = np.roll(self.price_lst, -1)
            self.queue.popleft()
        self.queue.append(bar)
        self.volume_lst[len(self.queue) - 1] = bar.Volume
        self.price_lst[len(self.queue) - 1] = abs(bar.Close - bar.Open)

    def get_sma(self):
        if self.has_enough_data():
            return np.average(self.volume_lst), np.average(self.price_lst)
        return None

    def get_ema(self):
        if self.has_enough_data():
            return np.average(self.volume_lst, weights=self.weights), np.average(self.price_lst, weights=self.weights)
        return None

    def get_last(self):
        if self.queue:
            bar = self.queue[-1]
            return (bar.Volume, bar.Close - bar.Open)
        return None

    def has_enough_data(self):
        return len(self.queue) >= self.window_size

    def length(self):
        return len(self.queue)


class Volatility:
    def __init__(self):
        self.prices = []
        self.vol = None

    def add_price(self, p):
        self.prices.append(p)

    def fft_vol(self):
        if self.is_ready():
            window = np.array(self.prices)
            wmax, wmin = np.max(window), np.min(window)
            norm_window = (window - wmin) / (wmax - wmin)
            freq = np.absolute(np.fft.rfft(norm_window))
            vol = np.matmul(freq, np.arange(1, len(freq) + 1))
            self.vol = np.log(vol)
        return None

    def std_vol(self):
        if self.is_ready():
            vol = np.std(np.array(self.prices))
            self.vol = vol
        return None

    def linear_vol(self):
        if self.is_ready():
            window = np.array(self.prices)
            intervals = len(window) - 1
            p1 = window[0]
            p2 = window[-1]
            slope = (p2 - p1) / intervals
            linear = np.array([p1 + slope * i for i in range(intervals + 1)])
            vol = np.std(np.absolute(window - linear))
            self.vol = vol
        return None

    def cond_vol(self):
        if self.is_ready():
            window = np.absolute(np.diff(np.array(self.prices)))
            vol = np.std(window)
            self.vol = vol
        return None

    def is_ready(self):
        if len(self.prices) > 1:
            return True
        return False

    def clear_prices(self):
        self.prices = []

    def get_vol(self):
        return self.vol

    def get_length(self):
        return len(self.prices)

    def trim_prices(self):
        self.prices = [x for x in self.prices if x != 0]


class FatApricotMule(QCAlgorithm):

    def Initialize(self):
        # Date and Cash
        self.SetStartDate(2017, 11, 1)
        self.SetEndDate(2017, 12, 1)
        self.SetCash(10000000)

        # window size and thresholds
        self.rolling_window = 200
        self.buy_threshold = -1
        self.sell_threshold = 1

        # averages and volatility methods
        self.ema = True
        self.VolType = 'fft'  # fft, std, linear, cond

        # Trading Strategy
        # If false, buy when momentum > threshold; if true, buy when momentum < threshold
        self.reverse = self.buy_threshold < self.sell_threshold

        # Resolutions
        self.volatility_precision = Resolution.Minute
        self.momentum_period = Resolution.Daily

        # Parameters to adjust
        self.Tickers = ['WYNN', 'LVS', 'HLT', 'GME', 'NCLH', 'XPEV', 'LULU', 'EBAY', 'W', 'PENN', 'FD', 'TJX', 'CMG', 'ETSY', 'TSLA', 'AMZN', 'BABA', 'NKE', 'HD', 'F', 'GM', 'CCL', 'PCLN', 'LOW', 'PTON', 'MELI', 'MCD', 'SBUX', 'DKNG', 'JD', 'K', 'ADM', 'CHGG', 'HSY', 'SYY', 'BTI', 'DPS', 'HANS', 'DLTR', 'KHC', 'SAM', 'CBRNA', 'GIS', 'BUD', 'KMB', 'CLX', 'DH', 'WMT', 'MO', 'KO', 'PG', 'COST', 'PEP', 'PM', 'EL', 'DG', 'KFT', 'CL', 'KR', 'BYND', 'REXR', 'LAP', 'AMH', 'VC', 'COLD', 'CONE', 'ESS', 'EXR', 'PEAK', 'BYA', 'ARE', 'AGNC', 'EQR', 'RIGX', 'SUI', 'CBRE',
                        'HCN', 'INVH', 'WY', 'PSA', 'VICI', 'SBAC', 'DLR', 'AMT', 'SPG', 'EQIXD', 'AMB', 'TWRS', 'O', 'OPEN', 'ACN', 'PANW', 'TXN', 'NOW', 'SNOW', 'IBM', 'CRWD', 'ZM', 'ORCL', 'AVGO', 'NET', 'CSCO', 'ASMLF', 'TSM', 'QCOM', 'AMAT', 'PLTR', 'COIN', 'LRCX', 'AAPL', 'MSFT', 'AMD', 'NVDA', 'AFRM', 'ADBE', 'SQ', 'CRM', 'MU', 'INTC', 'UBER', 'DISCA', 'ATUS', 'LYV', 'TTWO', 'BILI', 'FB', 'GOOG', 'NFLX', 'DIS', 'AMC', 'SNAP', 'ROKU', 'SBC', 'CMCSA', 'BEL', 'ERTS', 'MTCH', 'TWLO', 'CHTR', 'RBLX', 'ATVI', 'TWTR', 'ABNB', 'NTES', 'PCS', 'Z', 'BIDU', 'PINS', 'DASH', 'SPOT']
        # If false, buy when momentum > threshold; if true, buy when momentum < threshold
        self.reverse = self.buy_threshold < self.sell_threshold
        self.purchase_date = {}
        # Maybe also consider adding short selling threshold

        # Initialization
        self.stock_avgs = {ticker: SlidingAverage(
            self.rolling_window) for ticker in self.Tickers}
        self.volatility = {ticker: Volatility() for ticker in self.Tickers}
        self.symbol_to_ticker = {}
        for ticker in self.Tickers:
            new_symbol = self.AddEquity(
                ticker, self.volatility_precision).Symbol
            self.symbol_to_ticker[new_symbol] = ticker
            self.Consolidate(ticker, self.momentum_period,
                             self.OnDataConsolidated)
            self.Schedule.On(self.DateRules.EveryDay(ticker),
                             self.TimeRules.At(16, 0), self.GetVolatility)
        self.nextTradeTime = self.Time.date()

        self.SetWarmup(timedelta(self.rolling_window + 1))

    def OnDataConsolidated(self, bar):
        if bar.Symbol in self.symbol_to_ticker:
            ticker = self.symbol_to_ticker[bar.Symbol]
            self.stock_avgs[ticker].insert_data(bar)
            #self.Debug(f"OnDataConsolidated {bar.Time.hour} {bar.Time.minute} {bar.Symbol} {bar.Open} {bar.Close}")

    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''
        current_date = self.Time.date()
        if current_date >= self.nextTradeTime:
            self.nextTradeTime = current_date + timedelta(1)
            holding_momentum = {}
            buys, sells = [], []
            for symbol, ticker in self.symbol_to_ticker.items():
                avgs = self.stock_avgs[ticker].get_ema(
                ) if self.ema else self.stock_avgs[ticker].get_sma()
                if avgs is not None:
                    volume_avg, price_avg = avgs[0], avgs[1]
                    volume, price = self.stock_avgs[ticker].get_last()
                    volatility = self.volatility[ticker].get_vol()
                    index = (volume / volume_avg) * \
                        (price / price_avg) * (1/volatility)
                    # self.Debug(f"{self.Time.hour} {self.Time.minute} {ticker}, {index}, {self.volatility[ticker]}")
                    to_buy = index < self.buy_threshold if self.reverse else index > self.buy_threshold
                    to_sell = index > self.sell_threshold if self.reverse else index < self.sell_threshold
                    if to_buy:
                        buys.append((ticker, symbol, index))
                    elif to_sell:
                        sells.append((ticker, symbol, index))
                    holding_momentum[ticker] = index
                self.volatility[ticker].clear_prices()

            #self.Debug(f"{[self.volatility[ticker].get_vol() for ticker in self.Tickers]}")

            buys = sorted(buys, key=lambda x: x[2], reverse=(not self.reverse))
            buy_set, buy_stocks = set(), []
            sum_buy_index = 0
            for buy_stock in buys:
                buy_set.add(buy_stock[1])
                buy_stocks.append(buy_stock)
                sum_buy_index += buy_stock[2]
            if buy_stocks:
                for symbol in self.Portfolio.keys():
                    timeDelta = 0
                    if symbol.Value in self.purchase_date:
                        timeDelta = datetime.now() - \
                            self.purchase_date[symbol.Value]
                        timeDelta = timeDelta.total_seconds()
                    if (symbol not in buy_set and self.Portfolio[symbol].Invested) or (self.Portfolio[symbol].Invested and timeDelta > 432000):
                        self.Liquidate(symbol)
                for ticker, symbol, index in buy_stocks:
                    self.SetHoldings(ticker, min(index/sum_buy_index, 1))
                    self.purchase_date[ticker] = datetime.now()
            else:
                for ticker, symbol, index in sells:
                    self.Liquidate(symbol)

        for ticker in self.Tickers:
            if ticker in data.Bars:
                self.volatility[ticker].add_price(data.Bars[ticker].Close)
                #self.Debug(f"{self.price_by_minute[ticker]}")

    def GetVolatility(self):
        """
        calculate price volatility with custom function every 10 minutes
        """
        for ticker in self.Tickers:
            if self.VolType == "fft":
                self.volatility[ticker].fft_vol()
            elif self.VolType == "std":
                self.volatility[ticker].std_vol()
            elif self.VolType == "linear":
                self.volatility[ticker].linear_vol()
            elif self.VolType == "cond":
                self.volatility[ticker].cond_vol()
            else:
                self.Debug(f"invalid volatility calculation method")
        pass
