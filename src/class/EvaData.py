#coding: utf-8

from collections import defaultdict
import numpy as np

class EvaData():
    def __init__(self, filename, num_items, num_evaluate, batch_size):
        self.filename = filename
        self.num_items = num_items
        self.num_evaluate = num_evaluate
        self.batch_size = batch_size
        self.index = 0
        self.hash_data, self.total_user_list, self.num_records = self.read_hash_data()
        self.generate_negative_data()
    
    def read_hash_data(self):
        f = open(self.filename)
        hash_data = defaultdict(int)
        total_user_list = set()
        num_records = 0
        for _, line in enumerate(f):
            arr = line.split("\t")
            hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_list.add(int(arr[0]))
            num_records += 1
        return hash_data, list(total_user_list), num_records

    def arrange_positive_data(self):
        hash_data = self.hash_data
        user_list = []
        item_list = []
        index_dict = defaultdict(list)
        index = 0
        for (u, i) in hash_data:
            user_list.append(u)
            item_list.append(i)
            index_dict[u].append(index)
            index = index + 1
        return np.reshape(user_list, [-1, 1]), np.reshape(item_list, [-1, 1]), index_dict

    def generate_negative_data(self):
        hash_data = self.hash_data
        total_user_list = self.total_user_list
        num_evaluate = self.num_evaluate
        num_items = self.num_items
        negative_data = defaultdict(list)
        for u in total_user_list:
            for _ in range(num_evaluate):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                negative_data[u].append(j)
        self.negative_data = negative_data
    
    def generate_batch(self):
        batch_size = self.batch_size
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        num_evaluate = self.num_evaluate
        index = self.index
        terminal_flag = 1
        total_users = len(total_user_list)
        user_list = []
        item_list = []
        if index + batch_size < total_users:
            batch_user_list = total_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            terminal_flag = 0
            batch_user_list = total_user_list[index:total_users]
            self.index = 0
        for u in batch_user_list:
            user_list.extend([u]*num_evaluate)
            item_list.extend(negative_data[u])
        
        return batch_user_list, np.reshape(user_list, [-1, 1]), np.reshape(item_list, [-1, 1]), terminal_flag