#coding:utf8

from collections import defaultdict
import numpy as np

class TemporalValData():
    def __init__(self, filename, num_negatives, num_users, num_items, time_steps):
        self.filename = filename
        self.hash_data = self.generate_hash_data()
        self.num_negatives = num_negatives
        self.num_items = num_items
        self.num_users = num_users
        self.time_steps = time_steps
        self.hash_data, self.num_records, self.total_user_list = self.generate_hash_data()
        self.positive_history_dict = self.generate_history_dict()
    
    def generate_hash_data(self):
        num_records = 0
        # In order to help generating negative ratings
        hash_data = defaultdict(int)
        total_user_list = set()
        f = open(self.filename)
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
        return negative_history_dict
    
    def verify_val(self):
        positive_count = 0
        negative_count = 0
        positive_history_dict = self.positive_history_dict
        negative_history_dict = self.generate_negative_history()
        for u in positive_history_dict:
            positive_history_list = positive_history_dict[u]
            for t in positive_history_list:
                positive_count += len(positive_history_list[t])
        for u in negative_history_dict:
            negative_history_list = negative_history_dict[u]
            for t in negative_history_list:
                negative_count += len(negative_history_list[t])
        print('positive records:%d, negative records:%d' % (positive_count, negative_count))
    
    def get_val_input(self):
        time_steps = self.time_steps
        total_user_list = self.total_user_list
        positive_history_dict = self.positive_history_dict
        negative_history_dict = self.generate_negative_history()
        # item list, value list, dynamic index list, static index list
        val_item_list = []
        val_dynamic_index_list = []
        val_static_index_list = []
        val_labels_list = []
        for u in total_user_list:
            positive_history_list = positive_history_dict[u]
            negative_history_list = negative_history_dict[u]
            for t in range(time_steps):
                if t in positive_history_list:
                    val_item_list.extend(positive_history_list[t])
                    #### if t ranges from 0 to T-1, then user with idx at time tt can be described as idx*time_steps + tt ####
                    #### else it should be described as idx*time_steps + (tt - 1) ####
                    ### ! important replace index with u, because it use total users ###
                    val_dynamic_index_list.extend([u * time_steps + t] * len(positive_history_list[t]))
                    val_static_index_list.extend([u] * len(positive_history_list[t]))
                    val_labels_list.extend([1] * len(positive_history_list[t]))               
                    val_item_list.extend(negative_history_list[t])
                    val_dynamic_index_list.extend([u * time_steps + t] * len(negative_history_list[t]))
                    val_static_index_list.extend([u] * len(negative_history_list[t]))
                    val_labels_list.extend([0] * len(negative_history_list[t]))
        return np.reshape(val_item_list, [-1, 1]), np.reshape(val_dynamic_index_list, [-1, 1]), \
            np.reshape(val_static_index_list, [-1, 1]), np.reshape(val_labels_list, [-1, 1])