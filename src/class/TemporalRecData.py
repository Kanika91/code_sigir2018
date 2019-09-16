#coding:utf8

from collections import defaultdict
import numpy as np

class TemporalRecData():
    def __init__(self, filename, batch_size, time_steps, num_items, num_users, num_negatives):
        self.filename = filename
        self.index = 0
        self.total_user_list = range(num_users)
        self.batch_size = batch_size
        self.num_items = num_items
        self.time_steps = time_steps
        self.num_negatives = num_negatives
        self.hash_data, self.num_records, self.total_user_list = self.generate_hash_data()
        self.positive_history_dict = self.generate_history_dict()
    
    def generate_hash_data(self):
        num_records = 0
        # In order to help generating negative ratings
        hash_data = defaultdict(int)
        f = open(self.filename)
        total_user_list = set()
        for _, line in enumerate(f):
            tmp = line.split('\t')
            u, i, r = int(tmp[0]), int(tmp[1]), float(tmp[2])
            if r > 0:
                num_records += 1
                hash_data[(u, i)] = 1
                total_user_list.add(u)
        return hash_data, num_records, list(total_user_list)
    
    def generate_history_dict(self):
        positive_history_dict = dict()
        f = open(self.filename)
        for _, line in enumerate(f):
            tmp = line.split('\t')
            u, i, r, t = int(tmp[0]), int(tmp[1]), float(tmp[2]), int(tmp[3])-1
            if r > 0:
                if u not in positive_history_dict:
                    positive_history_list = defaultdict(list)
                else:
                    positive_history_list = positive_history_dict[u]
                positive_history_list[t].append(i)
                positive_history_dict[u] = positive_history_list
        return positive_history_dict

    def generate_negative_history(self):
        positive_history_dict = self.positive_history_dict
        num_negatives = self.num_negatives
        num_items = self.num_items
        hash_data = self.hash_data
        negative_history_dict = dict()
        self.index = 0
        for u in positive_history_dict:
            positive_history_list = positive_history_dict[u]
            if u not in negative_history_dict:
                negative_history_list = defaultdict(list)
            else:
                negative_history_list = negative_history_dict[u]
            for t in positive_history_list:
                for _ in positive_history_list[t]:
                    for _ in range(num_negatives):
                        j = np.random.randint(num_items)
                        while (u, j) in hash_data:
                            j = np.random.randint(num_items)
                        negative_history_list[t].append(j)
            negative_history_dict[u] = negative_history_list
        self.negative_history_dict = negative_history_dict
    
    def get_pooling_input(self, user_list):
        time_steps = self.time_steps
        positive_history_dict = self.positive_history_dict
        pooling_item_list = []
        pooling_index_list = []
        for index, u in enumerate(user_list):
            if u in positive_history_dict:
                history_list = positive_history_dict[u]
                for t in range(time_steps):
                    if t in history_list:
                        pooling_item_list.extend(history_list[t])
                        pooling_index_list.extend([index * time_steps + t] * len(history_list[t]))
        return np.reshape(pooling_item_list, [-1, 1]), pooling_index_list
    
    def get_loss_input(self, user_list):
        time_steps = self.time_steps
        positive_history_dict = self.positive_history_dict
        negative_history_dict = self.negative_history_dict
        # item list, labels list, dynamic index list, static index list
        loss_item_list = []
        loss_dynamic_index_list = []
        loss_static_index_list = []
        loss_labels_list = []
        for index, u in enumerate(user_list):
            positive_history_list = positive_history_dict[u]
            negative_history_list = negative_history_dict[u]
            for t in list(positive_history_list.keys()):
                loss_item_list.extend(positive_history_list[t])
                loss_dynamic_index_list.extend([index * time_steps + t] * len(positive_history_list[t]))
                loss_labels_list.extend([1] * len(positive_history_list[t]))
                loss_static_index_list.extend([index] * len(positive_history_list[t]))
                loss_item_list.extend(negative_history_list[t])
                loss_dynamic_index_list.extend([index * time_steps + t] * len(negative_history_list[t]))
                loss_labels_list.extend([0] * len(negative_history_list[t]))
                loss_static_index_list.extend([index] * len(negative_history_list[t]))
        return np.reshape(loss_item_list, [-1, 1]), np.reshape(loss_dynamic_index_list, [-1, 1]),\
            np.reshape(loss_static_index_list, [-1, 1]), np.reshape(loss_labels_list, [-1, 1])
    
    def generate_batch_user_list(self):
        index = self.index
        batch_size = self.batch_size
        terminal_flag = 1
        total_user_list = self.total_user_list
        if index + batch_size < len(total_user_list):
            user_list = total_user_list[index:index+batch_size]
        else:
            terminal_flag = 0
            user_list = total_user_list[index:len(total_user_list)]
        self.index = index + batch_size
        return user_list, terminal_flag