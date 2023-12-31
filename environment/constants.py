#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

# always use os.path.realpath(__file__) to determine the location of the current file, constants.py.
DATABASE_DIR = os.path.realpath(__file__).replace(os.path.join('environment', 'constants.py'), 'database')
DATABASE = os.path.join(DATABASE_DIR, 'Data.db')
# About time
NOW = 0
FIVE_MINUTES = 60 * 5
FIFTEEN_MINUTES = FIVE_MINUTES * 3
HALF_HOUR = FIFTEEN_MINUTES * 2
HOUR = HALF_HOUR * 2
TWO_HOUR = HOUR * 2
FOUR_HOUR = HOUR * 4
DAY = HOUR * 24
YEAR = DAY * 365
TIME_LOOKUP = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400 , 'w': 604800, 'M': 2592000}
# trading table name
TABLE_NAME = 'test'

if __name__ == '__main__':
    print(DATABASE, DATABASE_DIR)
    print('DAY:', DAY)

