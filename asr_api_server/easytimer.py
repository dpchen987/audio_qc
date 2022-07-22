#!/usr/bin/env python
# coding:utf-8


import time


class Timer:
    ''' print time cost '''

    def __init__(self, logger=None):
        assert type(logger) != str
        self.name = 'timer'
        self.logger = logger
        self.b_time = time.time()

    def begin(self, name=''):
        if name:
            self.name = name
        self.b_time = time.time()
        msg = f'=== Timer-{self.name} begin @ {time.strftime("%H:%M:%S")}'
        if self.logger:
            self.logger.debug(msg)
        else:
            print(msg)

    def end(self, name=''):
        if name:
            self.name = name
        time_cost = round(time.time() - self.b_time, 4)
        msg = (f'=== Timer-{self.name} passed [{time_cost}], end @ '
               f'{time.strftime("%H:%M:%S")}')
        if self.logger:
            self.logger.debug(msg)
        else:
            print(msg)
