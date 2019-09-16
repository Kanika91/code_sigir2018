#coding: utf8

#define constant
from TemporalRecData import TemporalRecData
from TemporalSocData import TemporalSocData
from TemporalValData import TemporalValData
from TemporalTestData import TemporalTestData
from EvaData import EvaData
from Logging import Logging
from evaluate import *

from time import time

def prepare_training_data(filename, num_users, time_steps, num_items, num_negatives, training_batch_size):
    t0 = time()
    train_filename = "%strain.rating" % filename
    val_filename = "%sval.rating" % filename
    test_filename = "%stest.rating" % filename
    soc_filename = "%strain.links" % filename
    train = TemporalRecData(train_filename, training_batch_size, time_steps, num_items, num_users, num_negatives)
    val = TemporalValData(val_filename, num_negatives, num_users, num_items, time_steps)
    test = TemporalTestData(test_filename, num_negatives, num_items)
    soc = TemporalSocData(soc_filename, time_steps)
    t1 = time()
    print("prepare data cost time:%.4fs, train:%d, val:%d, test:%d" %\
        ((t1-t0), train.num_records, val.num_records, test.num_records))
    return train, val, test, soc

def prepare_evaluate_data(filename, num_items, num_evaluate, evaluate_batch_size):
    test_filename = "%stest.rating" % filename
    eva = EvaData(test_filename, num_items, num_evaluate, evaluate_batch_size)
    print("evaluate records:%d" % (eva.num_records))
    return eva