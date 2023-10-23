import math
from environment.constants import TIME_LOOKUP
import ccxt

milliseconds = 1000


class Binance:
    def __init__(self, APIKey='', Secret='', config={'enableRateLimit': True, 'rateLimit': 0}):
        self.APIKey = APIKey.encode()
        self.Secret = Secret.encode()
        self.exchange = ccxt.binance(config)
        self.market = self.exchange.load_markets()
        self.timeframes = {float(interval[:-1]) * TIME_LOOKUP[interval[-1]]: interval for interval in
                           self.exchange.timeframes}


    def returnChartData(self, currencyPair, period, start, end):
        '''
        return chart from start - int(start%period) to end - int(end%period)
        '''
        retToRes = lambda dataset: [
            {'date': data[0] / milliseconds, 'open': data[1], 'high': data[2], 'low': data[3], 'close': data[4],
             'volume': data[5], 'quoteVolume': data[4] * data[5]} for data in dataset]
        adjustStart, adjustEnd = start - int(start % period), end - int(end % period)
        remain = int((adjustEnd - adjustStart) / period + 1)
        res = retToRes(
            self.exchange.fetchOHLCV(currencyPair, self.timeframes[period], adjustStart * milliseconds, remain))
        count = len(res)
        if count > 0:
            for i in range(1, math.ceil(remain / count)):
                ret = self.exchange.fetchOHLCV(currencyPair, self.timeframes[period], (adjustStart + i * count * period) * milliseconds, min((i + 1) * count, remain) - i * count)
                res += retToRes(ret)
            """
            # TODO: fix the problem that we call api too often
            with futures.ThreadPoolExecutor(max_workers=16) as executor:
                jobs = [executor.submit(self.exchange.fetchOHLCV, currencyPair, self.timeframes[period],
                                        (adjustStart + i * count * period) * milliseconds,
                                        min((i + 1) * count, remain) - i * count) for i in
                        range(1, math.ceil(remain / count))]
                for ret in futures.as_completed(jobs):
                    res += retToRes(ret.result())
            """
        return res


if __name__ == '__main__':
    '''
    date_norm             
1.600000e+09  10347.99
1.600000e+09  10333.41
1.600001e+09  10307.01
1.600001e+09  10339.03
1.600001e+09  10332.41
    '''

    exchange = ccxt.binance()
    print(exchange.fetchOHLCV('BTC/USDT', '5m', 1483246800000, 3))
