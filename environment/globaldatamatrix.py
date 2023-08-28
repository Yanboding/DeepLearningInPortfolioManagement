from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
from tqdm import tqdm

from constants import *
import sqlite3
from datetime import datetime
import logging
import xarray as xr

from constants import TIME_LOOKUP

from marketAPIWrapper.binance import Binance


class HistoryManager:
    # if offline ,the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples, each tuple is a row
    def __init__(self, baseAsset='USDT', online=True, market=Binance):
        self.initialize_db()
        self.__storage_period = FIVE_MINUTES  # keep this as 300
        self._online = online
        self.baseAsset = baseAsset
        if self._online:
            self.exchange = market()
            # we can simplify input
            self.active_coins = self.allActiveCoins()
        self.__coins = None

    @property
    def coins(self):
        return self.__coins

    def initialize_db(self):
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS History (date FLOAT,'
                           ' coin varchar(20), high FLOAT, low FLOAT,'
                           ' open FLOAT, close FLOAT, volume FLOAT, quoteVolume FLOAT,'
                           'PRIMARY KEY (date, coin));')
            connection.commit()

    def allActiveCoins(self):
        # connect the internet to access marketTicker
        pairs = []
        coins = []
        # I did not see when price are used.
        for k in self.exchange.market:
            if not k.startswith(self.baseAsset + "/") and not k.endswith("/" + self.baseAsset):
                continue
            pairs.append(k)
            coin1, coin2 = k.split('/')
            if coin2 == self.baseAsset:
                coins.append(coin1)
            elif coin1 == self.baseAsset:
                coins.append(coin2)
        activeCoins = pd.DataFrame({'coin': coins, 'pair': pairs})
        activeCoins.set_index('coin', inplace=True)
        return activeCoins

    def get_global_panel_by_volume(self, start, end, period=300, features=('close',), coin_number=3,
                                   volume_average_days=30):
        self.__checkperiod(period)
        # select coins
        start = int(start - (start % period))
        end = int(end - (end % period))
        # update all coins from start to end if online is true
        # use data from database.
        if self._online:
            self.update_data(self.active_coins.index, start, end)
        # select best coins by volume
        # check how volume are calculated if offline
        coins = self.select_coins(start=end - volume_average_days * DAY, end=end, coin_number=coin_number)
        self.__coins = coins

        logging.info("feature type list is %s" % str(features))
        return self._get_global_panel(coins, start, end, period, features)

    def get_global_panel_by_coins(self, coins, start, end, period=300, features=('close',)):
        self.__checkperiod(period)
        # select coins
        start = int(start - (start % period))
        end = int(end - (end % period))
        # update all coins from start to end if online is true
        # use data from database.
        if self._online:
            self.update_data(coins, start, end)
        self.__coins = coins

        logging.info("feature type list is %s" % str(features))
        return self._get_global_panel(coins, start, end, period, features)

    def _get_global_panel(self, coins, start, end, period=300, features=('close',)):
        start = int(start - (start % period))
        end = int(end - (end % period))
        time_index = pd.to_datetime(list(range(start, end + 1, period)), unit='s')
        panel = xr.DataArray(coords=[time_index, [self.baseAsset]+coins, list(features)], dims=['time', 'coin', 'feature'])
        panel.loc[time_index, self.baseAsset, features] = 1
        with sqlite3.connect(DATABASE_DIR) as connection:
            for row_number, coin in enumerate(coins):
                for feature in features:
                    # NOTE: transform the start date to end date
                    if feature == "close":
                        sql = 'SELECT date + :period AS date_norm, close ' \
                              'FROM History ' \
                              'WHERE date_norm BETWEEN :start AND :end AND date_norm % :period = 0 AND coin = :coin '
                    elif feature == "high":
                        sql = 'SELECT date + :period - (date % :period) AS date_norm, MAX(high) ' \
                              'FROM History ' \
                              'WHERE date_norm BETWEEN :start AND :end AND coin = :coin ' \
                              'GROUP BY date_norm'
                    elif feature == "low":
                        sql = 'SELECT date + :period - (date % :period) AS date_norm, MIN(low) ' \
                              'FROM History ' \
                              'WHERE date_norm BETWEEN :start AND :end AND coin = :coin ' \
                              'GROUP BY date_norm'
                    elif feature == "open":
                        sql = 'SELECT date + :period AS date_norm, open ' \
                              'FROM History ' \
                              'WHERE date_norm BETWEEN :start AND :end AND date_norm % :period = 0 AND coin = :coin '
                    elif feature == "volume":
                        sql = 'SELECT date + :period - (date % :period) AS date_norm, SUM(quoteVolume) ' \
                              'FROM History ' \
                              'WHERE date_norm BETWEEN :start AND :end AND coin = :coin ' \
                              'GROUP BY date_norm'
                    else:
                        msg = ("The feature %s is not supported" % feature)
                        logging.error(msg)
                        raise ValueError(msg)
                    serial_data = pd.read_sql_query(sql, con=connection,
                                                    parse_dates=["date_norm"],
                                                    index_col="date_norm",
                                                    params={'start': start, 'end': end, 'period': period, 'coin': coin})
                    panel.loc[serial_data.index, coin, feature] = serial_data.squeeze()
            panel = panel.bfill(dim='time', limit=None).ffill(dim='time', limit=None)
        return panel

    # select top coin_number of coins by volume from start to end from database
    def select_coins(self, start, end, coin_number):
        logging.info("select coins offline from %s to %s" % (datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
                                                             datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')))
        with sqlite3.connect(DATABASE_DIR) as connection:
            sql = 'SELECT coin,SUM(quoteVolume)' \
                  ' AS total_volume From History' \
                  ' WHERE History.date BETWEEN :start AND :end' \
                  ' GROUP BY coin ORDER BY total_volume DESC LIMIT :coinNumber'
            coins = pd.read_sql_query(sql, con=connection,
                                      params={'start': start, 'end': end, 'coinNumber': coin_number})['coin']
            if len(coins) != coin_number:
                logging.error("the sqlite error happend")
        return coins

    def __checkperiod(self, period):
        valid = {FIVE_MINUTES, FIFTEEN_MINUTES, HALF_HOUR, TWO_HOUR, FOUR_HOUR, DAY}
        if period not in valid:
            raise ValueError('peroid has to be 5min, 15min, 30min, 2hr, 4hr, or a day')

    # add new history data into the database
    def update_data(self, coins, start, end):
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            # get all ticker, for each ticker, get all return data finnally store them in database
            # consider use multithread in the future
            for coin in tqdm(coins):
                pair = self.active_coins.at[coin, 'pair']
                chart = self.exchange.returnChartData(
                    pair,
                    period=self.__storage_period,
                    start=start,
                    end=end
                )
                data = []
                for c in chart:
                    c['coin'] = coin
                    # this coin is quote coin
                    if pair.endswith("/" + coin):
                        c['high'] = 1 / c['high']
                        c['low'] = 1 / c['low']
                        c['open'] = 1 / c['open']
                        c['close'] = 1 / c['close']
                        c['volume'], c['quoteVolume'] = c['quoteVolume'], c['volume']
                    data.append(c)
                # add coin to each data
                # insert many into
                sql = 'INSERT OR REPLACE INTO History (date, coin, high, low, open, close, volume, quoteVolume) ' \
                      'VALUES (:date, :coin, :high, :low, :open, :close, :volume, :quoteVolume)'
                cursor.executemany(sql, data)
                connection.commit()


def get_history_data_by_coins(start, end, period, coins, online, features, baseAsset='USDT'):
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").timestamp()
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").timestamp()
    if isinstance(period, str):
        period = TIME_LOOKUP[period[-1]] * int(period[:-1])
    historyManager = HistoryManager(baseAsset, online)
    history_data = historyManager.get_global_panel_by_coins(coins, start, end, period, features)
    return history_data


if __name__ == '__main__':
    import pickle
    data_params = {
        'start': '2018-06-01',
        'end': '2022-12-31',
        'period': '30m',
        'coins': ['BTC', 'ETH', 'XRP', 'BNB', 'ADA'],
        'online': False,
        'features': ['close', 'high', 'low'],
        'baseAsset': 'USDT'
    }
    data = get_history_data_by_coins(**data_params)
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(len(data))
