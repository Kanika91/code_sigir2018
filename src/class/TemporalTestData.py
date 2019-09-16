#coding:utf8

from collections import defaultdict
import numpy as np

class TemporalTestData():
    def __init__(self, filename, num_negatives, num_items):
        self.filename = filename
        self.hash_data, self.num_records, self.total_user_list = self.generate_hash_data()
        self.test_positive_dict = self.get_test_positives()
        self.num_negatives = num_negatives
        self.num_items = num_items

    def generate_hash_data(self):
        hash_data = defaultdict(int)
        num_records = 0
        total_user_list = set()
        f = open(self.filename)
        for _, line in enumerate(f):
            tmp = line.split("\t")
            u, i, r = int(tmp[0]), int(tmp[1]), float(tmp[2])
            if r > 0:
                hash_data[(u, i)] = 1
                num_records += 1
                total_user_list.add(u)
        return hash_data, num_records, list(total_user_list)

    def get_test_positives(self):
        test_positive_dict = defaultdict(list)
        hash_data = self.hash_data
        for (u, i) in hash_data:
            test_positive_dict[u].append(i)
        return test_positive_dict

    def generate_test_negatives(self):
        num_items = self.num_items
        hash_data = self.hash_data
        num_negatives = self.num_negatives
        test_negative_dict = defaultdict(list)
        for (u, _) in hash_data:
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                test_negative_dict[u].append(j)
        return test_negative_dict
    
    def get_test_input(self):
        test_user_list = []
        test_item_list = []
        test_labels_list = []
        total_user_list = self.total_user_list
        test_positive_dict = self.test_positive_dict
        test_negative_dict = self.generate_test_negatives()
        for u in total_user_list:
            test_user_list.extend([u] * len(test_positive_dict[u]))
            test_item_list.extend(test_positive_dict[u])
            test_labels_list.extend([1] * len(test_positive_dict[u]))
            test_user_list.extend([u] * len(test_negative_dict[u]))
            test_item_list.extend(test_negative_dict[u])
            test_labels_list.extend([0] * len(test_negative_dict[u]))
        return np.reshape(test_user_list, [-1, 1]), np.reshape(test_item_list, [-1, 1]), \
            np.reshape(test_labels_list, [-1, 1])