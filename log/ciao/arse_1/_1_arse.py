import os, sys, shutil

import tensorflow as tf
import ConfigParser as cp
from tensorflow.contrib import rnn
from arse_dataset import *
from clean_tmp_model import *
from time import time

def start_model(conf, record_id):
    # read configuration from bpr.ini
    config = cp.ConfigParser()
    config.read(conf)

    root_dir = config.get('Basic', 'root_dir')
    model_name = config.get('Model', 'name')
    path = "%s%s_%s" % (root_dir, model_name, record_id)
    if not os.path.exists(path):
            os.makedirs(path)
            print('create directory %s success.' % path)
    # copy configure to dst directory
    dst_conf = '%s/_%s_%s.ini' % (path, record_id, model_name)
    shutil.copyfile(conf, dst_conf)
    # copy model code to dst directory
    src_dir = config.get('Model', 'src_dir')
    src_code = '%s%s.py' % (src_dir, model_name)
    dst_code = '%s/_%s_%s.py' % (path, record_id, model_name)
    shutil.copyfile(src_code, dst_code)
    # define log name 
    log_filename = '%s/_%s_%s.log' % (path, record_id, model_name)
    # define tmp train model
    train_model = '%s/tmp_model/%s' % (path, model_name)
    # create tmp_model store tmp models
    train_model_dir = '%s/tmp_model' % path
    if not os.path.exists(train_model_dir):
        os.makedirs(train_model_dir)

    os.environ['CUDA_VISIBLE_DEVICES']= config.get('Basic', 'gpu_device')
    num_users = int(config.get('Basic', 'num_users'))
    num_items = int(config.get('Basic', 'num_items'))
    dimension = int(config.get('Model', 'dimension'))
    learning_rate = float(config.get('Model', 'learning_rate'))
    epochs = int(config.get('Model', 'epochs'))
    num_negatives = int(config.get('Model', 'num_negatives'))
    num_evaluate = int(config.get('Model', 'num_evaluate'))
    data_dir = config.get('Recommendation', 'data_dir')
    num_procs = int(config.get('Model', 'num_procs'))
    topK = int(config.get('Model', 'topK'))
    evaluate_batch_size = int(config.get('Model', 'evaluate_batch_size'))
    training_batch_size = int(config.get('Model', 'training_batch_size'))
    pretrain_flag = int(config.get('Pretrain', 'pretrain_flag'))
    pre_model = config.get('Pretrain', 'pre_model_filename')
    time_steps = int(config.get('Model', 'time_steps'))
    #[_, _, social_embedding] = np.load(config.get('Pretrain', 'social_embedding_filename'))
    social_embedding = np.zeros((num_users, dimension))


    batch_user_input = tf.placeholder("int32", [None])
    pooling_item_input = tf.placeholder("int32", [None, 1])
    pooling_index_input = tf.placeholder("int32", [None])
    neighbors_origin_input = tf.placeholder("int32", [None])
    dynamic_neigh_index_input = tf.placeholder("int32", [None, 1])
    static_neigh_real_input = tf.placeholder("int32", [None, 1])

    social_user_index_input = tf.placeholder("int32", [None])
    social_user_real_input = tf.placeholder("int32", [None, 1])

    loss_item_input = tf.placeholder("int32", [None, 1])
    loss_dynamic_index_input = tf.placeholder("int32", [None, 1])
    loss_static_index_input = tf.placeholder("int32", [None, 1])
    loss_labels_input = tf.placeholder("float32", [None, 1])

    social_embedding_input = tf.placeholder("float32", [num_users, dimension])
    hidden_embedding_input = tf.placeholder("float32", [None, dimension])

    static_attention_weights = tf.Variable(tf.random_normal([dimension*6, 1], stddev=0.01), name='static_attention_weights')
    second_static_attention_weights = tf.Variable(tf.nn.relu(tf.random_normal([num_users, 1], stddev=0.01)), name='second_static_attention_weights')
    dynamic_attention_weights = tf.Variable(tf.random_normal([dimension*6, 1], stddev=0.01), name='dynamic_attention_weights')    
    second_dynamic_attention_weights = tf.Variable(tf.random_normal([num_users, 1], stddev=0.01), name='second_dynamic_attention_weights')
    convert_predict_weights = tf.Variable(tf.random_normal([2*dimension, 1], stddev=0.01), name='convert_predict_weights')
    dynamic_item_embedding = pooling_item_embedding = tf.Variable(tf.random_normal([num_items, dimension], stddev=0.01), name='pooling_item_embedding')
    #dynamic_item_embedding = tf.Variable(tf.random_normal([num_items, dimension], stddev=0.01), name='dynamic_item_embedding')
    static_item_embedding = tf.Variable(tf.random_normal([num_items, dimension], stddev=0.01), name='static_item_embedding')
    lstm_init_user_embedding = tf.Variable(tf.random_normal([num_users, dimension], stddev=0.01), name='lstm_init_user_embedding')
    static_basic_user_embedding = tf.Variable(tf.random_normal([num_users, dimension], stddev=0.01), name='static_basic_user_embedding')
    static_soc_cf_user_embedding = tf.Variable(tf.random_normal([num_users, dimension], stddev=0.01), name='static_soc_cf_user_embedding')
    lstm_layer = rnn.BasicLSTMCell(dimension, state_is_tuple=True, name='lstm_layer')

    #[static_basic_user_embedding, static_item_embedding] = np.load('/home/sunpeijie/files/task/pyrec/log/gowalla/bpr_3/tmp_model/bpr_hr_0.6703_ndcg_0.4772.npy')

    def get_pooling_item_latent(pooling_item_embedding, 
        pooling_item_input, 
        pooling_index_input,
        dimension, 
        time_steps):
        '''
        Description: Input Pooling (need to be verified)
            time_steps: T
            dimension: D
            num_users: B
            input itemid: list by user(0 ~ B-1) by time (0 ~ T-1)
            input index: 0~T-1, T~2T-1, 2T~3T-1 ... (N-2)T~(N-1)T-1
            output: T's B * D
        '''
        pooling_item_latent = tf.gather_nd(pooling_item_embedding, pooling_item_input)
        pooling_item_latent = tf.segment_max(pooling_item_latent, pooling_index_input)
        pooling_item_latent = tf.concat([pooling_item_latent, \
            tf.zeros([time_steps - tf.mod(pooling_index_input[-1], time_steps) - 1, dimension])], 0)
        pooling_item_latent = tf.reshape(pooling_item_latent, [-1, time_steps, dimension])
        pooling_item_latent = tf.unstack(pooling_item_latent, time_steps, 1)
        return pooling_item_latent

    def generate_dynamic_neighbor_embedding(lstm_layer, 
        lstm_init_user_embedding, 
        neigh_pooling_item_latent, 
        neighbors_origin_input,
        dimension,
        time_steps):
        '''
        Description: Generate Assist Hidden Embedding
            lstm_layer = rnn.BasicLSTMCell(dimension, state_is_tuple=True) # define public LSTM cell
            num_users: B (about static_user_input C's neighbors)
            neighbors_origin_input: B's real userid(C's total neighbors)
            dimension: D
            time_steps: T
            pooling_item_latent: T * B * D
            lstm_init_user_embedding: total_users(N) * D
            output: (B * T) * D ... (u1: T*D, u2: T*D, ..., uB: T*D)
        '''
        neighbors_origin_input = tf.reshape(neighbors_origin_input, [-1, 1])
        h_0 = lstm_init_user_latent = tf.gather_nd(lstm_init_user_embedding, neighbors_origin_input)
        c_0 = tf.zeros([tf.shape(neighbors_origin_input)[0], dimension])
        state_0 = rnn.LSTMStateTuple(c=c_0, h=h_0)  # c alias for field number 0, h alias for field number 1
        outputs_0 = []
        for i in range(time_steps):
            output_0, state_0 = lstm_layer(neigh_pooling_item_latent[i], state_0)
            outputs_0.append(output_0)
        hidden_outputs = tf.concat(outputs_0[0: -1], 1)
        hidden_embedding = tf.reshape(tf.concat([lstm_init_user_latent, hidden_outputs], 1), [-1, dimension])
        return hidden_embedding

    def generate_dynamic_user_embedding(second_dynamic_attention_weights, 
        static_soc_cf_user_embedding,
        lstm_layer, batch_user_input,
        lstm_init_user_embedding,
        dimension,
        time_steps,
        social_user_real_input, 
        social_user_index_input,
        static_neigh_real_input,
        dynamic_neigh_index_input,
        dynamic_attention_weights,
        pooling_item_latent,
        social_embedding_input,
        hidden_embedding_input):
        '''
        Description: Dynamic Attentive and Generate Dynamic User Embedding
            lstm_layer = rnn.BasicLSTMCell(dimension, state_is_tuple=True) # define public LSTM cell
            batch_user_input: for each training batch, we have B users, and we use np.reshape([U_i1, U_i2, ..., U_i|B|]) 
                denotes this formal parameter
            lstm_init_user_embedding: (total_users N) * D
            dimension: each use can be denoted as a D-dimension vector, the input is the number D
            time_steps: each data can be divided into T time-steps, the input is the number T
            social_user_real_input: in order to boost the computation efficiency, we adopt a special strategy which can be found from the following code.
                for example, if the input batch only has two users `a` and `b`, which has |S_a| and |S_b| neighbors separately, 
                we repeat the userid of `a` |S_a| times and `b` |S_b| times, so we can get np.reshape([a, a, ..., b, b, ....]), 
            social_user_index_input: this value one can refer to the usage of tf.segment_mean in tensorflow, https://www.tensorflow.org/api_docs/python/tf/segment_mean
                this one corresponde to the segment_ids
            static_neigh_real_input: the input is connected with `social_user_index_input`, for example, if the input batch 
                only has two users `a` and `b`, for them, we can get their social neighbors ids, 
                we use np.reshape([S_a1, S_a2, ..., S_a_(last social neighbor), S_b1, S_b2, ..., S_a_(last social neighbor)]) denotes 
                the formal parameter
            dynamic_neigh_index_input: Similar to the social_user_real_input, but the userid a must be replaced by the index of a in the batch.
            dynamic_attention_weights: initalize it with tf.random_normal([dimension*4, 1], stddev=0.01)
            pooling_item_latent: T * B * D
            social_embedding_input: the value of this parameter can be get by any embedding representation method with the social network data
            hidden_embedding_input: (B's total neighbors number: C) (C*T) * D
            
            output: 
                dynamic_user_embedding: (T * B) * D, u_1:T*D, u_2:T*D, ..., u_B:T*D
                evaluate_user_embedding: B * D
        '''
        second_dynamic_attention = tf.gather_nd(second_dynamic_attention_weights, tf.reshape(batch_user_input, [-1, 1]))
        sc_user_real_latent = tf.gather_nd(social_embedding_input, social_user_real_input)
        sc_neigh_real_latent = tf.gather_nd(social_embedding_input, static_neigh_real_input)
        soc_cf_user_real_latent = tf.gather_nd(static_soc_cf_user_embedding, social_user_real_input)
        soc_cf_neigh_real_latent = tf.gather_nd(static_soc_cf_user_embedding, static_neigh_real_input)

        h_0 = lstm_init_user_latent = tf.gather_nd(lstm_init_user_embedding, tf.reshape(batch_user_input, [-1, 1]))
        c_0 = tf.zeros([tf.shape(batch_user_input)[0], dimension])
        state_1 = rnn.LSTMStateTuple(c=c_0, h=h_0)  # c alias for field number 0, h alias for field number 1
        outputs_1 = []
        for i in range(time_steps):
            c_1, h_1 = state_1[0], state_1[1]
            hidden_user_latent = tf.gather_nd(h_1, tf.reshape(social_user_index_input, [-1, 1]))
            temporal_neigh_index = dynamic_neigh_index_input + 1
            hidden_neigh_latent = tf.gather_nd(hidden_embedding_input, temporal_neigh_index)
            concat_latent = tf.concat(
                [sc_user_real_latent, sc_neigh_real_latent, hidden_user_latent, hidden_neigh_latent, \
                soc_cf_user_real_latent, soc_cf_neigh_real_latent], axis=1)
            dm_attention_1 = tf.matmul(concat_latent, dynamic_attention_weights) #+ dynamic_attention_bias
            dm_attention_2 = tf.nn.sigmoid(dm_attention_1) + 0.01
            dm_attention_3 = tf.segment_sum(dm_attention_2, social_user_index_input)
            dm_attention_4 = tf.gather_nd(dm_attention_3, tf.reshape(social_user_index_input, [-1, 1]))
            dm_attention_5 = tf.div(dm_attention_2, dm_attention_4)
            extra_user_latent = tf.multiply(hidden_neigh_latent, dm_attention_5)
            extra_user_latent = tf.segment_sum(extra_user_latent, social_user_index_input)

            ### next line realize DARSE AVG ###
            #extra_user_latent = tf.segment_mean(hidden_neigh_latent, social_user_index_input)

            h_latent = h_1 + tf.multiply(extra_user_latent, second_dynamic_attention)

            ### next line realize none attention
            h_latent = h_1

            state_1 = rnn.LSTMStateTuple(c=c_1, h=h_latent)
            output_1, state_1 = lstm_layer(pooling_item_latent[i], state_1)
            outputs_1.append(output_1)
        hidden_outputs = tf.concat(outputs_1[0: -1], 1)
        dynamic_user_embedding = tf.reshape(tf.concat([lstm_init_user_latent, hidden_outputs], 1), [-1, dimension])
        evaluate_user_embedding = outputs_1[-1]
        return dynamic_user_embedding, evaluate_user_embedding, hidden_neigh_latent

    def generate_static_user_embedding(static_attention_weights,
            static_basic_user_embedding,
            static_soc_cf_user_embedding, # test flag, insert id: arse_13
            second_static_attention_weights, #test flag, insert id: arse_15
            social_embedding_input,
            batch_user_input,
            static_neigh_real_input,
            social_user_index_input,
            social_user_real_input):
        '''
        Description: Static Attention Network and Generate Static User Embedding
            static_attention_weights: 4D*1
            tmp_sc_user_latent: C's map userid correspond to C's social neighbors + sc_user_latent
            sc_neigh_latent: C's social neighbors real userid + social_embedding
            tmp_rs_user_latent: C's map userid correspond to C's social neighbors + rs_user_latent
            rs_neigh_latent: C's social neighbors real userid + basic_user_embedding
            user_map_input: C's map userid correspond to C's social neighbors 
            rs_user_latent: C's real userid + static_basic_user_embedding
        '''
        origin_rs_user_latent = tf.gather_nd(static_basic_user_embedding, tf.reshape(batch_user_input, [-1, 1]))
        second_static_attention = tf.gather_nd(second_static_attention_weights, tf.reshape(batch_user_input, [-1, 1]))
        rs_user_latent = tf.gather_nd(static_basic_user_embedding, social_user_real_input)
        sc_user_latent = tf.gather_nd(social_embedding_input, social_user_real_input)
        rs_neigh_latent = tf.gather_nd(static_soc_cf_user_embedding, static_neigh_real_input)
        sc_neigh_latent = tf.gather_nd(social_embedding_input, static_neigh_real_input)
        
        tmp_rs_user_latent = tf.gather_nd(static_soc_cf_user_embedding, social_user_real_input)
        tmp_rs_neigh_latent = tf.gather_nd(static_basic_user_embedding, static_neigh_real_input)
        
        concat_latent = tf.concat(
                    [sc_user_latent, sc_neigh_latent, rs_user_latent, rs_neigh_latent,\
                    tmp_rs_user_latent, tmp_rs_neigh_latent], axis=1)
        sm_attention_1 = tf.matmul(concat_latent, static_attention_weights)
        sm_attention_2 = tf.nn.sigmoid(sm_attention_1) + 0.01 #can be modified#
        sm_attention_3 = tf.segment_sum(sm_attention_2, social_user_index_input)
        sm_attention_4 = tf.gather_nd(sm_attention_3, tf.reshape(social_user_index_input, [-1, 1]))
        sm_attention_5 = tf.div(sm_attention_2, sm_attention_4)
        extra_user_latent_1 = tf.multiply(rs_neigh_latent, sm_attention_5)
        extra_user_latent = tf.segment_sum(extra_user_latent_1, social_user_index_input)  #can be modified#

        ### next line realize AVG SARSE ###
        #extra_user_latent = tf.segment_mean(rs_neigh_latent, social_user_index_input)

        #static_user_embedding = extra_user_latent
        static_user_embedding = origin_rs_user_latent + tf.multiply(extra_user_latent, tf.nn.relu(second_static_attention))
        #static_user_embedding = origin_rs_user_latent + 0.01 * extra_user_latent

        ### none attention srse ###
        #static_user_embedding = origin_rs_user_latent

        return static_user_embedding, static_basic_user_embedding, \
            sm_attention_1, sm_attention_2, sm_attention_3, sm_attention_4, sm_attention_5, rs_neigh_latent

    def compute_dynamic_predct_vector(dynamic_user_embedding,
            dynamic_item_embedding,
            loss_dynamic_index_input,
            loss_item_input):
        '''
        Description: Compute Dynamic Predict Vector
        Parameters:
            dynamic_user_embedding
            dynamic_item_embedding
            loss_dynamic_index_input
            loss_item_input
        '''
        dynamic_user_latent = tf.gather_nd(dynamic_user_embedding, loss_dynamic_index_input) # loss_dynamic_index_input default user index
        dynamic_item_latent = tf.gather_nd(dynamic_item_embedding, loss_item_input)
        dynamic_multiply_vector = tf.multiply(dynamic_user_latent, dynamic_item_latent)
        return dynamic_multiply_vector, dynamic_user_latent, dynamic_item_latent

    def compute_static_predict_vector(static_user_embedding,
            static_item_embedding,
            loss_static_index_input,
            loss_item_input):
        '''
        Description: Compute Static Predict Vector
        Parameters:
            static_user_embedding
            static_item_embedding
            loss_static_index_input
            loss_item_input
        '''
        static_user_latent = tf.gather_nd(static_user_embedding, loss_static_index_input) # loss_static_index_input default user index
        static_item_latent = tf.gather_nd(static_item_embedding, loss_item_input)
        static_multiply_vector = tf.multiply(static_user_latent, static_item_latent)
        return static_multiply_vector, static_user_latent, static_item_latent

    #### Compute Prediction and Loss ####
    pooling_item_latent = \
        get_pooling_item_latent(pooling_item_embedding, pooling_item_input, pooling_index_input, 
            dimension, time_steps)
    hidden_embedding = \
        generate_dynamic_neighbor_embedding(lstm_layer, lstm_init_user_embedding, pooling_item_latent, 
            neighbors_origin_input, dimension, time_steps)
    dynamic_user_embedding, evaluate_user_embedding, concat_latent = \
        generate_dynamic_user_embedding(second_dynamic_attention_weights, static_soc_cf_user_embedding,\
            lstm_layer, batch_user_input,
            lstm_init_user_embedding, dimension, time_steps, social_user_real_input, social_user_index_input, 
            static_neigh_real_input, dynamic_neigh_index_input, dynamic_attention_weights, pooling_item_latent, 
            social_embedding_input, hidden_embedding_input)
    static_user_embedding, static_basic_user_embedding, \
            sm_attention_1, sm_attention_2, sm_attention_3, sm_attention_4, sm_attention_5, rs_neigh_latent = \
        generate_static_user_embedding(static_attention_weights,
            static_basic_user_embedding, static_soc_cf_user_embedding,
            second_static_attention_weights, social_embedding_input, batch_user_input,
            static_neigh_real_input, social_user_index_input, social_user_real_input)
    dynamic_multiply_vector, dynamic_user_latent, dynamic_item_latent = \
        compute_dynamic_predct_vector(dynamic_user_embedding,
            dynamic_item_embedding, loss_dynamic_index_input, loss_item_input)
    static_multiply_vector, static_user_latent, static_item_latent = \
        compute_static_predict_vector(static_user_embedding,
            static_item_embedding, loss_static_index_input, loss_item_input)

    concat_multiply_vector = tf.concat([dynamic_multiply_vector, static_multiply_vector], axis=1)
    predict_vector = tf.matmul(concat_multiply_vector, convert_predict_weights)    
    ### next line realize none DARSE ###
    #predict_vector = static_multiply_vector

    ### next line realize none SARSE
    predict_vector = tf.reduce_sum(dynamic_multiply_vector, 1, keepdims=True)
    
    prediction = tf.nn.sigmoid(predict_vector)
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(loss_labels_input, prediction))
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    ##################################### Save Variables ####################################
    variables_dict = {v.op.name: v for v in lstm_layer.variables}
    variables_dict[static_attention_weights.op.name] = static_attention_weights
    variables_dict[dynamic_attention_weights.op.name] = dynamic_attention_weights
    variables_dict[pooling_item_embedding.op.name] = pooling_item_embedding
    variables_dict[dynamic_item_embedding.op.name] = dynamic_item_embedding
    variables_dict[lstm_init_user_embedding.op.name] = lstm_init_user_embedding
    variables_dict[static_soc_cf_user_embedding.op.name] = static_soc_cf_user_embedding
    variables_dict[second_static_attention_weights.op.name] = second_static_attention_weights
    variables_dict[convert_predict_weights.op.name] = convert_predict_weights
    saver = tf.train.Saver(variables_dict)

    ############################ Validation, Test, Evaluation ################################
    side_static_user_embedding = tf.placeholder("float32", [None, dimension])
    side_dynamic_user_embedding = tf.placeholder("float32", [None, dimension])
    side_item_input = tf.placeholder("int32", [None, 1])
    side_static_user_index_input = tf.placeholder("int32", [None, 1])
    side_dynamic_user_index_input = tf.placeholder("int32", [None, 1])
    side_labels_input = tf.placeholder("float32", [None, 1])

    def side_compute_dynamic_predct_vector(side_dynamic_user_embedding,
            side_dynamic_user_index_input,
            side_item_input,
            dynamic_item_embedding):
        '''
        Description: Dynamic Attentive and Generate Dynamic User Embedding
        Parameters:
            side_dynamic_user_embedding
            side_item_input
            side_user_index_input
            dynamic_item_embedding
        '''
        side_dynamic_user_latent = tf.gather_nd(side_dynamic_user_embedding, side_dynamic_user_index_input)
        side_dynamic_item_latent = tf.gather_nd(dynamic_item_embedding, side_item_input)
        side_dynamic_multiply_vector = tf.multiply(side_dynamic_user_latent, side_dynamic_item_latent)
        return side_dynamic_multiply_vector

    def side_compute_static_predict_vector(side_static_user_embedding,
            side_static_user_index_input,
            side_item_input,
            static_item_embedding):
        '''
        Description: Dynamic Attentive and Generate Dynamic User Embedding
        Parameters:
            side_static_user_embedding
            side_user_index_input
            side_item_input
            static_item_embedding
        '''
        side_static_user_latent = tf.gather_nd(side_static_user_embedding, side_static_user_index_input)
        side_static_item_latent = tf.gather_nd(static_item_embedding, side_item_input)
        side_static_multiply_vector = tf.multiply(side_static_user_latent, side_static_item_latent)
        return side_static_multiply_vector

    side_dynamic_multiply_vector = \
        side_compute_dynamic_predct_vector(side_dynamic_user_embedding, side_dynamic_user_index_input, 
        side_item_input, dynamic_item_embedding)
    side_static_multiply_vector = \
        side_compute_static_predict_vector(side_static_user_embedding, side_static_user_index_input, 
        side_item_input, static_item_embedding)

    side_concat_multiply_vector = tf.concat([side_dynamic_multiply_vector, side_static_multiply_vector], axis=1)
    side_predict_vector = tf.matmul(side_concat_multiply_vector, convert_predict_weights)
    #side_multiply_vector = side_dynamic_multiply_vector + side_static_multiply_vector
    
    ### next line realize None DARSE ###
    #side_multiply_vector = side_static_multiply_vector

    ### next line realize None SARSE ###
    side_predict_vector = tf.reduce_sum(side_dynamic_multiply_vector, 1, keepdims=True)

    side_prediction = tf.nn.sigmoid(side_predict_vector)
    side_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(side_labels_input, side_prediction))

    ############################ Start to Prepare Data and Train ################################
    '''
        Train:
            Basic Parameters: dimension, time_steps
            origin users:
                dynamic history:
                    INPUT: pooling_item_input, pooling_index_input
                dynamic embedding:
                    INPUT: social_user_real_input, social_user_index_input
                        static_neigh_real_input, dynamic_neigh_index_input
                dynamic predict:
                    INPUT: loss_dynamic_index_input, loss_item_input
                dynamic loss:
                    INPUT: loss_value_input
                static embedding:
                    INPUT: social_user_real_input, social_user_index_input, static_neigh_index_input
                static predict:
                    INPUT: loss_static_index_input, loss_item_input
                static loss:
                    INPUT: loss_value_input
            neighbors: 
                pooling:
                    INPUT: neigh_pooling_item_input neigh_pooling_index_input
                dynamic embedding:
                    INPUT: neighbors_origin_input
        Val:
            dynamic loss:
                INPUT: val_loss_user_map_input, loss_item_real_input
            static loss:
                INPUT: val_loss_user_real_input, loss_item_real_input
        Test:
            dynamic loss:
                INPUT: test_loss_user_real_input, loss_item_real_input, loss_value_input
            static loss:
                INPUT: test_loss_user_real_input, loss_item_real_input, loss_value_input
        Evaluation:
            Positive:
                dynamic predict:
                    INPUT: 
                static predict:
                    INPUT: 
            Negative:
    '''

    init = tf.global_variables_initializer()
    # start tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)

    ## load pretrain model
    if pretrain_flag == 1:
        saver.restore(sess, pre_model)

    # set debug_flag=0, doesn't print any results
    log = Logging(log_filename, debug_flag=1)

    # second define constant and prepare evaluate data
    eva = prepare_evaluate_data(data_dir, num_items, num_evaluate, evaluate_batch_size)
    max_hr, max_ndcg, earlystop_flag = 0, 0, 1
    val_loss_dict = {}

    train, val, test, soc = \
        prepare_training_data(data_dir, num_users, time_steps, num_items, num_negatives, training_batch_size) 
    learn_user_eva_embedding = np.zeros([num_users, dimension])
    learn_user_common_embedding = np.zeros([num_users * time_steps, dimension])
    learn_user_static_embedding = np.zeros([num_users, dimension])

    # Start Training !!!
    for epoch in range(epochs):
        # optimize model with training data and compute train loss
        train_count, train_loss, terminal_flag = 0, 0, 1
        train.generate_negative_history()

        train_index = 0
        
        t0 = time()
        while terminal_flag:
            train_index += 1
            user_list, terminal_flag = train.generate_batch_user_list()
            static_neigh_real_list, social_user_real_list, social_user_index_list, temporal_neighbors_list = \
                soc.get_static_social(user_list)
            neigh_pooling_item_list, neigh_pooling_index_list = train.get_pooling_input(temporal_neighbors_list)
            
            [neigh_hidden_embedding, dynamic_item_output] = sess.run([hidden_embedding, dynamic_item_embedding], feed_dict={neighbors_origin_input: temporal_neighbors_list,\
                pooling_item_input: neigh_pooling_item_list, pooling_index_input: neigh_pooling_index_list})
            
            dynamic_neigh_index_list = soc.get_dynamic_social(user_list)
            pooling_item_list, pooling_index_list = train.get_pooling_input(user_list)
            loss_item_list, loss_dynamic_index_list, loss_static_index_list, loss_labels_list = \
                train.get_loss_input(user_list)

            [out_dynamic, out_evaluate, out_static, 
                sub_train_loss, _] = \
                sess.run([dynamic_user_embedding, evaluate_user_embedding, static_user_embedding, 
                loss, opt], \
                feed_dict={batch_user_input: user_list, 
                    pooling_item_input: pooling_item_list, 
                    pooling_index_input: pooling_index_list,
                    hidden_embedding_input: neigh_hidden_embedding,
                    social_embedding_input: social_embedding, 
                    social_user_real_input: social_user_real_list,
                    social_user_index_input: social_user_index_list,
                    static_neigh_real_input: static_neigh_real_list,
                    dynamic_neigh_index_input: dynamic_neigh_index_list,
                    loss_item_input: loss_item_list,
                    loss_dynamic_index_input: loss_dynamic_index_list,
                    loss_static_index_input: loss_static_index_list,
                    loss_labels_input: loss_labels_list})
            sub_train_count = len(user_list)
            train_loss = train_loss + sub_train_count * sub_train_loss
            train_count = train_count + sub_train_count
            
            for idx in range(len(user_list)):
                u = user_list[idx]
                learn_user_static_embedding[u] = out_static[idx]
                learn_user_eva_embedding[u] = out_evaluate[idx]
                learn_user_common_embedding[u*time_steps:((u+1)*time_steps-1)] = \
                    out_dynamic[idx*time_steps:((idx+1)*time_steps-1)]
        train_loss = train_loss / train_count
        t1 = time()
        train.index = 0

        train_index = 0

        # compute val loss and test loss
        item_list, dynamic_index_list, static_index_list, labels_list = val.get_val_input()
        val_loss = sess.run(side_loss, \
            feed_dict={side_static_user_index_input: static_index_list, 
                side_dynamic_user_index_input: dynamic_index_list,
                side_item_input: item_list, 
                side_labels_input: labels_list,
                side_dynamic_user_embedding: learn_user_common_embedding, 
                side_static_user_embedding: learn_user_static_embedding})
        val_loss_dict[epoch] = val_loss

        user_list, item_list, labels_list = test.get_test_input()
        test_loss = sess.run(side_loss, \
            feed_dict={side_static_user_index_input: user_list, 
                side_dynamic_user_index_input: user_list,
                side_item_input: item_list, 
                side_labels_input: labels_list,
                side_dynamic_user_embedding: learn_user_eva_embedding, 
                side_static_user_embedding: learn_user_static_embedding})

        t2 = time()

        # start evaluate model performance, hr and ndcg
        def get_positive_predictions(user_list, item_list):
            positive_predictions = sess.run(side_prediction,
                feed_dict={side_static_user_index_input: user_list, 
                    side_dynamic_user_index_input: user_list,
                    side_item_input: item_list, 
                    side_dynamic_user_embedding: learn_user_eva_embedding, 
                    side_static_user_embedding: learn_user_static_embedding})
            return positive_predictions

        def get_negative_predictions(eva):
            negative_predictions = {}
            terminal_flag = 1
            while terminal_flag:
                batch_user_list, user_list, item_list, terminal_flag = eva.generate_batch()
                index = 0
                tmp_negative_predictions = np.reshape(sess.run(side_prediction,
                    feed_dict={side_static_user_index_input: user_list, 
                        side_dynamic_user_index_input: user_list,
                        side_item_input: item_list, 
                        side_dynamic_user_embedding: learn_user_eva_embedding, 
                        side_static_user_embedding: learn_user_static_embedding}), [-1, num_evaluate])
                for u in batch_user_list:
                    negative_predictions[u] = tmp_negative_predictions[index]
                    index = index + 1
            return negative_predictions

        user_list, item_list, index_dict = eva.arrange_positive_data()
        positive_predictions = get_positive_predictions(user_list, item_list)
        negative_predictions = get_negative_predictions(eva)
        eva.index = 0 # !!!important, prepare for new batch

        hr, ndcg, _ = evaluate_fun_proc(index_dict, positive_predictions, negative_predictions, topK, num_procs)
        if ndcg > max_ndcg or hr > max_hr:
            np.save("%s_hr_#%.4f#_ndcg_#%.4f#" % (train_model, hr, ndcg), \
                [learn_user_common_embedding, learn_user_eva_embedding, learn_user_static_embedding])
            saver.save(sess, "%s_hr_#%.4f#_ndcg_#%.4f#.ckpt" % (train_model, hr, ndcg))
        max_hr = max(hr, max_hr)
        max_ndcg = max(ndcg, max_ndcg)

        t3 = time()

        if epoch > 10 and val_loss_dict[epoch] > val_loss_dict[epoch-1] and val_loss_dict[epoch-1] > val_loss_dict[epoch-2] and earlystop_flag:
            log.record('max hr:%.4f, max ndcg:%.4f' % (max_hr, max_ndcg))
            log.record('-----------------------------********************-------------------------------')
            log.record('-----------------------------****Early Stop******-------------------------------')
            log.record('-----------------------------********************-------------------------------')
            earlystop_flag = 0

        # print log to console and log_file
        log.record('Epoch:%d, compute loss cost:%.4fs, train loss:%.4f, val loss:%.4f, test loss:%.4f' % (epoch, (t2-t0), train_loss, val_loss, test_loss))
        log.record('evaluate cost:%.4fs, hr:%.4f, ndcg:%.4f' % ((t3-t2), hr, ndcg))
    log.record('max hr:%.4f, max ndcg:%.4f' % (max_hr, max_ndcg))
    log.record('--------------------- Train Over -------------------')

    # clean tmp models
    arse_clean_tmp_model_parameters(train_model_dir, max_hr, max_ndcg)
