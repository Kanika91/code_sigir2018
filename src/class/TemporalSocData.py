#coding:utf8

from collections import defaultdict
import numpy as np

class TemporalSocData():
    def __init__(self, filename, time_steps):
        self.filename = filename
        self.time_steps = time_steps
        self.neighbors_dict = self.generate_neighbors_dict()

    def generate_neighbors_dict(self):
        neighbors_dict = defaultdict(list)
        f = open(self.filename)
        for _, line in enumerate(f):
            tmp = line.split("\t")
            u1, u2 = int(tmp[0]), int(tmp[1])
            neighbors_dict[u1].append(u2)
        return neighbors_dict

    def get_dynamic_social(self, user_list):
        temporal_neighbors_dict = self.temporal_neighbors_dict
        neighbors_dict = self.neighbors_dict
        time_steps = self.time_steps
        dynamic_neigh_index_list = []
        index = 0
        for u in user_list:
            neighbors_list = neighbors_dict[u]
            for n in neighbors_list:
                dynamic_neigh_index_list.append(temporal_neighbors_dict[n] * time_steps)
            index += 1
        return np.reshape(dynamic_neigh_index_list, [-1, 1])
    
    def get_static_social(self, user_list):
        neighbors_dict = self.neighbors_dict
        static_neigh_real_list = []
        social_user_real_list = []
        social_user_index_list = []
        for index, u in enumerate(user_list):
            static_neigh_real_list.extend(neighbors_dict[u])
            social_user_real_list.extend([u] * len(neighbors_dict[u]))
            social_user_index_list.extend([index] * len(neighbors_dict[u]))
        temporal_neighbors_list = list(set(static_neigh_real_list))
        temporal_neighbors_dict = defaultdict(int)
        for idx, n in enumerate(temporal_neighbors_list):
            temporal_neighbors_dict[n] = idx
        self.temporal_neighbors_dict = temporal_neighbors_dict
        return np.reshape(static_neigh_real_list, [-1, 1]), np.reshape(social_user_real_list, [-1, 1]), \
            social_user_index_list, temporal_neighbors_list