#coding:utf8

from collections import defaultdict
import numpy as np
from time import time

class RecData():
    def __init__(self, filename, num_users, num_items, num_negatives, batch_size=0):
        self.filename = filename
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.hash_data, self.total_user_list = self.read_hash_data()
        self.positive_data, self.negative_data, self.num_total_data = self.generate_training_data() 
        #self.training_data, self.training_list = self.generate_training_data_v2()   
        self.index = 0
        self.batch_size = batch_size

    def generate_negative_data(self):
        num_items = self.num_items
        num_negatives = self.num_negatives
        negative_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            for t in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                negative_data[u].add(j)
                total_data.add((u, j))
        self.negative_data = negative_data
        self.index = 0

    def read_hash_data(self):
        f = open(self.filename)
        total_user_list = set()
        hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split("\t")
            hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_list.add(int(arr[0]))
        return hash_data, list(total_user_list)
    
    def generate_training_data(self):
        num_items = self.num_items
        num_negatives = self.num_negatives
        negative_data = defaultdict(set)
        positive_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            positive_data[u].add(i)
            for t in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                negative_data[u].add(j)
                total_data.add((u, j))
        return positive_data, negative_data, len(total_data)

    def generate_training_data_v2(self):
        hash_data = self.hash_data
        training_data = defaultdict(int)
        for (u, i) in hash_data:
            training_data[(u, i)] = 1
            for t in range(self.num_negatives):
                j = np.random.randint(self.num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(self.num_items)
                training_data[(u, j)] = 0
        return training_data, list(training_data.keys())

    def generate_one_batch(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        user_list = []
        item_list = []
        labels_list = []
        for u in total_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(positive_data[u])
            labels_list.extend([1] * len(positive_data[u]))
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(negative_data[u])
            labels_list.extend([0] * len(negative_data[u]))
        return np.reshape(user_list, [-1, 1]), np.reshape(item_list, [-1, 1]), np.reshape(labels_list, [-1, 1])
    
    '''
    def generate_one_batch_v2(self):
        training_data = self.training_data
        user_list = []
        item_list = []
        labels_list = []
        for (u, i) in training_data:
            user_list.append(u)
            item_list.append(i)
            labels_list.append(training_data[(u, i)])
        return np.reshape(user_list, [-1, 1]), np.reshape(item_list, [-1, 1]), np.reshape(labels_list, [-1, 1])
    '''

    def get_batch(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        index = self.index
        batch_size = self.batch_size
        terminal_flag = 1
        user_list = []
        item_list = []
        labels_list = []
        if index + batch_size < len(total_user_list):
            target_user_list = total_user_list[index:index+batch_size]
        else:
            terminal_flag = 0
            target_user_list = total_user_list[index:len(total_user_list)]
            self.index = 0
        for u in target_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(list(positive_data[u]))
            labels_list.extend([1] * len(positive_data[u]))
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(list(negative_data[u]))
            labels_list.extend([0] * len(negative_data[u]))
        self.index = index + batch_size
        return np.reshape(user_list, [-1, 1]), np.reshape(item_list, [-1, 1]), np.reshape(labels_list, [-1, 1]), terminal_flag
    
    '''
    def get_batch_v2(self):
        training_data = self.training_data
        training_list = self.training_list
        index = self.index
        batch_size = self.batch_size
        terminal_flag = 1
        user_list = []
        item_list = []
        labels_list = []
        if index + batch_size < len(training_list):
            for e in training_list[index:index+batch_size]:
                user_list.append(e[0])
                item_list.append(e[1])
                labels_list.append(training_data[e])
        else:
            terminal_flag = 0
            for e in training_list[index:len(training_list)]:
                user_list.append(e[0])
                item_list.append(e[1])
                labels_list.append(training_data[e])
        self.index = index + batch_size
        return np.reshape(user_list, [-1, 1]), np.reshape(item_list, [-1, 1]), np.reshape(labels_list, [-1, 1]), terminal_flag
    '''  
