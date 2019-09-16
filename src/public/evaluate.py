import math
import numpy as np
import ipdb

def get_mean(dict):
    count = 0
    num_keys = 0
    for user_id in dict.keys():
        num_keys = num_keys + 1
        value = dict[user_id]
        count = count + value
    return count * 1.0 / num_keys

def get_idcg(length):
    idcg = 0.0
    for i in range(length):
        idcg = idcg + math.log(2) / math.log(i + 2)
    return idcg

def get_dcg(value):
    dcg = math.log(2) / math.log(value + 2)
    return dcg

def get_hr(value):
    hit = 1.0
    return hit

def get_precision(hits_num, topK):
    return hits_num * 1.0 / topK

def get_recall(hits_num, positive_length):
    return hits_num * 1.0 / positive_length

def get_f1_measure(precision, recall):
    return 2.0 * precision * recall / (precision + recall)

### precision@n ###
def get_p_at_n(hits_num, topK):
    return hits_num * 1.0 / topK

### recall@n ###
def get_r_at_n(hits_num, positive_length):
    return hits_num * 1.0 / positive_length

### mAP ###
### refer to 
### https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision
### ###
def get_mAP(p_list, r_list):
    tmp_mAP = p_list[0] * r_list[0]
    for idx, _ in enumerate(r_list):
        if idx != 0:
            tmp_mAP += p_list[idx] * (r_list[idx] - r_list[idx-1])
    return tmp_mAP

def evaluate_fun_proc(evaluate_index_dict, evaluate_real_rating_matrix, \
    evaluate_predict_rating_matrix, topK, num_procs, exp_flag=0, sp_name=None, result_file=None):
    user_list = list(evaluate_index_dict.keys())
    batch_size = len(user_list) / num_procs

    hr_list = []
    ndcg_list = []
    mAP_list = []
    index = 0
    for _ in xrange(num_procs):
        if index + batch_size < len(user_list):
            batch_user_list = user_list[index:index+batch_size]
            index = index + batch_size
        else:
            batch_user_list = user_list[index:len(user_list)]
        tmp_hr_list, tmp_ndcg_list, tmp_mAP_list = get_hr_ndcg_proc(evaluate_index_dict, evaluate_real_rating_matrix, \
            evaluate_predict_rating_matrix, topK, batch_user_list)
        hr_list.extend(tmp_hr_list)
        ndcg_list.extend(tmp_ndcg_list)
        mAP_list.extend(tmp_mAP_list)
    return np.mean(hr_list), np.mean(ndcg_list), np.mean(mAP_list)

def get_hr_ndcg_proc(evaluate_index_dict, evaluate_real_rating_matrix, \
    evaluate_predict_rating_matrix, topK, user_list):
    tmp_hr_list = []
    tmp_ndcg_list = []
    tmp_precision_list = []
    tmp_recall_list = []
    tmp_f1_list = []
    tmp_mAP_list = []
    
    for u in user_list:
        real_item_index_list = evaluate_index_dict[u]
        real_item_rating_list = list(np.concatenate(evaluate_real_rating_matrix[real_item_index_list]))
        positive_length = len(real_item_rating_list)
        target_length = min(positive_length, topK)

        #ipdb.set_trace()
        #print('topK:%d, positive:%d' % (topK, len(real_item_rating_list)))
        
        predict_rating_list = evaluate_predict_rating_matrix[u]
        real_item_rating_list.extend(predict_rating_list)
        sort_index = np.argsort(real_item_rating_list)
        sort_index = sort_index[::-1]

        user_p_at_n_list = []
        user_r_at_n_list = []

        user_hr_list = []
        user_ndcg_list = []
        user_map_list = []
        hits_num = 0
        for idx in range(topK):
            ranking = sort_index[idx]
            if ranking < positive_length:
                hits_num += 1
                user_hr_list.append(get_hr(idx))
                user_ndcg_list.append(get_dcg(idx))
                #user_map_list.append(get_map(idx, len(user_hr_list)))
            user_p_at_n_list.append(get_p_at_n(hits_num, topK))
            user_r_at_n_list.append(get_r_at_n(hits_num, positive_length))
        
        idcg = get_idcg(target_length)

        tmp_hr = np.sum(user_hr_list) / target_length
        tmp_ndcg = np.sum(user_ndcg_list) / idcg
        tmp_hr_list.append(tmp_hr)
        tmp_ndcg_list.append(tmp_ndcg)

        '''
        ### compute precision for each user ###
        tmp_precision = get_precision(hits_num, topK)
        tmp_precision_list.append(tmp_precision)
        
        ### compute recall for each user ###
        tmp_recall = get_recall(hits_num, positive_length)
        tmp_recall_list.append(tmp_recall)

        ### compute f1 score ###
        tmp_f1 = get_f1_measure(tmp_precision, tmp_recall)
        tmp_f1_list.append(tmp_f1)
        '''

        ### compute mAP ###
        tmp_mAP = get_mAP(user_p_at_n_list, user_r_at_n_list)
        tmp_mAP_list.append(tmp_mAP)

        #if u == 17150:
        #    print('user:%.4f, hr:%.4f, ndcg:%.4f' % (u, tmp_hr, tmp_ndcg))
            
    return tmp_hr_list, tmp_ndcg_list, tmp_mAP_list