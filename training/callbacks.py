"""
Created on Tue Sep 11 20:52:46 2018
@author: pablosanchez
"""
import numpy as np
import logging
from utils.reporting.logging import log_message

class EarlyStopping(object):
    def __init__(self, name='', patience=5, min_delta=1e-6):
        self.name = name
        self.patience = patience
        self.min_delta = min_delta
        self.patience_cnt = 0
        self.prev_loss_val = np.Infinity

    def stop(self, loss_val):
        if self.prev_loss_val != np.Infinity:
            if np.abs(np.abs(self.prev_loss_val) - np.abs(loss_val)) >= self.min_delta:
                self.patience_cnt = 0
                self.prev_loss_val = loss_val
            else:
                self.patience_cnt += 1

                msg = '\nPervious {}: {}'.format(self.name, self.prev_loss_val)
                msg += '\nCurrent {}: {}'.format(self.name, loss_val)
                msg += '\n{} Patience count-out: {}'.format(self.name, self.patience_cnt)
                log_message(msg, logging.DEBUG)


        if (self.patience_cnt > self.patience):
            log_message('{} is out of patience'.format(self.name), logging.CRITICAL)
            return True
        else:
            return False
