# coding=utf-8

import numpy as np
import random
import Utility as util
#import Eval
import tensorflow as tf
import math
import time

group_prefix = "group1"
embed_size = 40
fold_total = 5
fold_n = 1
isTrain = 1
isTest = 0
isLoss = 0
isReward = 0

if isTrain:
    conduct = "training"
else:
    conduct = "testing"
print group_prefix, conduct, "fold=%d/%d" % (fold_n, fold_total)




if group_prefix == "group1":
    discount_rate = 0.9
    compress_rate = 0.2130
    beta_reward = 0.7

    learning_rate = 0.06
    total_epoch = 10
    decay_epoch = 10
    roll_iteration = 50

    #一些操作的起始点
    LR_n_epoch = total_epoch * 0.5
    LR_decrease_length = 3
    save_n_epoch = total_epoch * 0.1
    copy_n_epoch = total_epoch * 0.1
    momentum = 0.95

if group_prefix == "group2":
    discount_rate = 0.95
    compress_rate = 0.4115
    beta_reward = 0.9

    learning_rate = 0.06
    total_epoch = 70
    decay_epoch = 10
    roll_iteration = 50

    LR_n_epoch = total_epoch * 0.5
    LR_decrease_length = 3
    save_n_epoch = total_epoch * 0.1
    copy_n_epoch = total_epoch * 0.1
    momentum = 0.95


aligned_source = util.loadObj(group_prefix+"/aligned_input.pkl")
fid_index_map = util.loadObj(group_prefix+"/file_index_map.pkl") #index-fid
fid_index_map = util.getFidToIndexMap(fid_index_map) #fid-index

train_set_fids, test_set_fids = util.split_folds(fold_n, fid_index_map, n_split=fold_total)


file_sentence_size_map = util.loadObj(group_prefix+"/file_sentence_size_map.pkl")
file_size_map = util.loadObj(group_prefix+"/file_size_map.pkl")
file_valid_sentenceCount_map = util.loadObj(group_prefix+"/file_valid_sentenceCount_map.pkl")
file_sentenceVectors_map = util.loadObj(group_prefix+"/filedId_sentenceVectors_map.pkl")
file_sentenceSentiSum_map = util.loadObj(group_prefix+"/fileId_SentiValue_Sum_Map.pkl")
file_sentence_map = util.loadObj(group_prefix+'/fileId_Sentence_Map.pkl')

if isTrain:
    aligned_source = util.extractInstances(train_set_fids, fid_index_map, aligned_source)
    data_set = train_set_fids
if isTest:
    aligned_source = util.extractInstances(test_set_fids, fid_index_map, aligned_source)
    data_set = test_set_fids


doc_vector_map = util.compute_doc_vectors(data_set, file_sentenceVectors_map)
fileAmount, maxSen, mergeDimension = aligned_source.shape
aligned_source = np.reshape(aligned_source, newshape=[fileAmount, maxSen, mergeDimension, 1]) #CPU上面只能NHWC
aligned_summary = np.zeros((fileAmount, maxSen, embed_size)) #整个训练过程就是不断更新这个summary的过程

sentence_amount_mask = util.createValidSentenceMap(data_set, maxSen, file_valid_sentenceCount_map)



SSECIF_initializer = tf.variance_scaling_initializer()
def q_networks(source, summary, name):

    with tf.variable_scope(name) as scope:
        #CNN + pooling
        cnn_output1 = tf.layers.conv2d(inputs=source,
                                      filters=1,
                                      kernel_size=(1, mergeDimension), padding='valid',
                                      activation=tf.nn.tanh,
                                      kernel_initializer=SSECIF_initializer,
                                      bias_initializer=SSECIF_initializer)
        _, unit_dim1, _, _ = cnn_output1.shape
        print cnn_output1.shape

        pooling_output = tf.layers.average_pooling2d(inputs=cnn_output1, pool_size=(3, 1), strides=(3, 1), padding='valid')
        _, pooling_dim, _, _ = pooling_output.shape
        print pooling_output.shape

        cnn_output2 = tf.layers.conv2d(inputs=pooling_output,
                                       filters=1,
                                       kernel_size=(2, 1), padding='valid',
                                       strides=(2, 1),
                                       activation=tf.nn.tanh,
                                       kernel_initializer=SSECIF_initializer,
                                       bias_initializer=SSECIF_initializer)
        print cnn_output2.shape

        a, unit_dim2, _, _ = cnn_output2.shape
        cnn_output2 = tf.reshape(cnn_output2, shape=[a, unit_dim2])

        #FC adaptation
        adapt1 = tf.layers.dense(inputs=cnn_output2, units=unit_dim2, activation=tf.nn.tanh, kernel_initializer=SSECIF_initializer)

        adapt2 = tf.layers.dense(inputs=adapt1, units=pooling_dim, activation=tf.nn.tanh, kernel_initializer=SSECIF_initializer)

        adapt3 = tf.layers.dense(inputs=adapt2, units=unit_dim1, activation=tf.nn.tanh, kernel_initializer=SSECIF_initializer)
        adapt3_dim1, adapt3_dim2 = adapt3.get_shape().as_list() #(5,93)
        print adapt3_dim1, adapt3_dim2

        #None linear estimation
        importance_w = tf.Variable(initial_value=tf.truncated_normal(shape=(adapt3_dim2, adapt3_dim2)),
                                   dtype=tf.float32, name="importance_w", trainable=True)
        redundancy_w = tf.Variable(initial_value=tf.truncated_normal(shape=(embed_size, 1)),
                                    dtype=tf.float32, name="redundancy_w", trainable=True)

        importance_possibility = tf.nn.sigmoid(tf.matmul(adapt3, importance_w))

        dim1, dim2,  dim3 = summary.shape
        summary = tf.reshape(summary, shape=[-1, dim3])
        not_redundancy_possibility = tf.reshape(1-tf.nn.sigmoid(tf.matmul(summary, redundancy_w)), shape=[dim1, dim2])
        summary = tf.reshape(summary, shape=[dim1, dim2, dim3])

        q_values = importance_possibility*not_redundancy_possibility*100


        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = {var.name[len(scope.name):]:var for var in trainable_vars}

        return q_values, trainable_vars_by_name


X_source = tf.placeholder(dtype=tf.float32, shape=aligned_source.shape, name="aligned_source")
X_summary = tf.placeholder(dtype=tf.float32, shape=aligned_summary.shape, name="aligned_summary")
X_mask = tf.placeholder(dtype=tf.float32, shape=sentence_amount_mask.shape, name="sentence_amount_mask")
online_q_values, online_vars = q_networks(X_source, X_summary, "online_model")
target_q_values, target_vars = q_networks(X_source, X_summary, "target_model")
copy_ops = [target_var.assign(online_vars[var_name]) for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)


with tf.variable_scope("train"):
    LR = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
    learning_rate_decay_op = LR.assign(LR * 0.9)

    X_action = tf.placeholder(tf.float32, shape=[fileAmount, maxSen])
    y_target = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(tf.multiply(online_q_values, X_action), axis=-1, keepdims=True)

    error = tf.abs(y_target - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2*(error-clipped_error)
    train_loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=LR, momentum=momentum, use_nesterov=True)
    training_op = optimizer.minimize(train_loss)


init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=total_epoch*2)

def rolling_records(aligned_source, aligned_summary, mask, roll_iteration, train_set,
                    currentStep, sentenceVectors_map, sentenceCount_map, doc_vectors, beta_reward, maxSen, embed_size, compress):
    records = {}

    #预先跑2次计算reward
    for iteration in range(1, 3):
        q_values = online_q_values.eval(feed_dict={X_source: aligned_source, X_summary: aligned_summary}) * mask
        #此时随机选择
        sids = util.epsilon_greedy(q_values, train_set, sentenceCount_map, currentStep, decay_epoch)
        #添加记录
        records = util.init_records(sids, records, train_set)
        aligned_summary = util.updateSummary(aligned_summary, sids, sentenceVectors_map, train_set, maxSen)

    #back_up_sum0
    util.backup(group_prefix, "sum_0.pkl", aligned_summary)

    '''
    records 数据结构: records{fid:[ [(sumi-1, sumi, reward), (sumi-1, sumi, reward)], [Int_snapshot]]}
    '''

    #从O 到 roll_iteration全备份
    #开始真正滚动
    for iteration in range(1, roll_iteration+1):
        q_values = online_q_values.eval(feed_dict={X_source: aligned_source, X_summary: aligned_summary}) * mask
        sids = util.epsilon_greedy(q_values, train_set, sentenceCount_map, currentStep, decay_epoch)
        # 在当前选择的sids基础上，深入summary立方内部修改，但是更新还是在整个数据集上更新
        records, aligned_summary, sids = util.compute_reward_add_records(records, doc_vectors, aligned_summary, iteration,
                                        sids, train_set, file_sentenceVectors_map, file_sentenceSentiSum_map,
                                        file_size_map, file_sentence_size_map, beta_reward, maxSen, embed_size, compress)
        # print sids
        aligned_summary = util.updateSummary(aligned_summary, sids, sentenceVectors_map, train_set, maxSen)
        util.backup(group_prefix, "sum_%d.pkl"%iteration, aligned_summary)
    return records

def train_model(aligned_source, aligned_summary, doc_vectors, save_n_epoch, copy_n_epoch, group):
    saver.restore(sess, tf.train.latest_checkpoint('../resources/%s/model/max_reward'% group))


    #util.del_file('../resources/%s/sum_logs' % group)
    #util.del_file('../resources/%s/model/min_loss' % group)
    #util.del_file('../resources/%s/model/max_reward' % group)
    #util.del_file('../resources/%s/model/final' % group)

    min_loss = 999999
    best_loss_epoch = -1
    max_reward = -9999
    best_reward_epoch = -1

    collected_step_loss = []
    collected_step_reward = []
    loss_step_track = []
    # log_lines = []

    for n_epoch in range(1, total_epoch + 1):


        sum1 = np.sum(aligned_summary, keepdims=False)
        sum2 = np.sum(sum1)
        assert sum2 == 0


        records = rolling_records(aligned_source, aligned_summary, sentence_amount_mask,
                                  roll_iteration, train_set_fids, n_epoch,
                                  file_sentenceVectors_map, file_valid_sentenceCount_map, doc_vectors, beta_reward,
                                  maxSen, embed_size, compress_rate)
        # for index in range(len(train_set_fids)):
        #     fid = train_set_fids[index]
        #     logs, snapshot = records[fid]
        #     print "fid:%d"%fid,
        #     for log in logs:
        #         print log,
        #     print

        instances, file_amount = util.extractInstancesFromRecords(records, train_set_fids, roll_iteration)
        #util.checkInstances(records, instances, roll_iteration, file_amount)
        random.shuffle(instances)
        # for item in instances:
        #     print item
        # print '================================================================'


        all_loss = []
        all_rewards = []

        for index in range(len(instances)):
            instance = instances[index]
            start, actions, next_id, rewards, moves = instance


            valid_rewards = []
            for j in range(len(actions)):
                if actions[j] != -1:
                    valid_rewards.append(rewards[j])
            if len(valid_rewards) != 0:
                all_rewards.append(np.average(valid_rewards))

            summary_path = "%s/sum_logs/sum_%d.pkl" % (group_prefix, next_id)
            next_summary = util.loadObj(summary_path)
            feed_dict = {X_source: aligned_source, X_summary: next_summary}
            next_q_values = target_q_values.eval(feed_dict=feed_dict) * sentence_amount_mask
            max_next_q_values = np.max(next_q_values, axis=-1, keepdims=False)
            #print "\tmax_next_q_values shape:", max_next_q_values.shape
            y_val = np.array(rewards) + np.array(moves) * max_next_q_values * discount_rate
            y_val = np.reshape(y_val, newshape=[-1, 1])
            #print "\ty_val:", y_val.shape

            summary_path = "%s/sum_logs/sum_%d.pkl" % (group_prefix, start)
            start_summary = util.loadObj(summary_path)
            action_mask = util.getActionMask(maxSen, file_amount, actions, train_set_fids)

            feed_dict = {X_source: aligned_source, X_summary: start_summary, X_action: action_mask, y_target: y_val}
            _, instance_loss = sess.run([training_op, train_loss], feed_dict=feed_dict)
            all_loss.append(instance_loss)

        avg_loss = np.average(all_loss)
        avg_reward = np.average(all_rewards)
        print "n_epoch=%d, avg_loss=%.6f, avg_reward=%.6f" % (n_epoch, avg_loss, avg_reward)
        collected_step_loss.append(avg_loss)
        collected_step_reward.append(avg_reward)
        # if n_epoch != total_epoch:
        #     line = "%f, %f\n" % (avg_loss, avg_reward)
        # else:
        #     line = "%f, %f" % (avg_loss, avg_reward)
        # log_lines.append(line)


        if n_epoch > LR_n_epoch:
            loss_step_track.append(avg_loss)


        if avg_reward > max_reward and n_epoch > save_n_epoch:
            max_reward = avg_reward
            best_reward_epoch = n_epoch
            util.del_file('../resources/%s/model/max_reward' % group_prefix)
            saver.save(sess, save_path='../resources/%s/model/max_reward/max_reward_%d.ckpt'%(group_prefix, best_reward_epoch))

        if avg_loss < min_loss and n_epoch > save_n_epoch:
            min_loss = avg_loss
            best_loss_epoch = n_epoch
            util.del_file('../resources/%s/model/min_loss' % group_prefix)
            saver.save(sess, save_path='../resources/%s/model/min_loss/min_loss_%d.ckpt'% (group_prefix, best_loss_epoch))


        if n_epoch > copy_n_epoch and n_epoch % copy_n_epoch == 0:
            copy_online_to_target.run()


        if n_epoch > LR_n_epoch and len(loss_step_track) == LR_decrease_length:
            loss_step_track, flag = util.isLossDecreased(loss_step_track, limitAmount=LR_decrease_length)
            if not flag:
                sess.run(learning_rate_decay_op)
                print 'decrease learning_rate=%.6f, n_epoch=%d, from 0.06' % (LR.eval(), n_epoch)


        util.del_file('../resources/%s/sum_logs' % group_prefix)
        aligned_summary = np.zeros((fileAmount, maxSen, embed_size))

    # Experiment-1:
    # logs_path = '../Experiments/Exp1/%s/f%d.txt' % (group, fold_n)
    # log_file = open(logs_path, 'w')
    # log_file.writelines(log_lines)
    # log_file.flush()
    # log_file.close()


    #7. save final model
    saver.save(sess, save_path='../resources/%s/model/final/final_model.ckpt' % group_prefix, )
    return min_loss, best_loss_epoch, max_reward, best_reward_epoch, collected_step_loss, collected_step_reward

def test_model(file_sentence_map, aligned_source, aligned_summary, sentence_mask, group, model_flag):
    util.del_file('../resources/%s/summary_agent/f%d' % (group,fold_n))

    test_start = time.time()
    q_values = online_q_values.eval(feed_dict={X_source: aligned_source, X_summary: aligned_summary}) * sentence_mask

    IIDs_map = Eval.generate_IIDs(q_values, test_set_fids, file_valid_sentenceCount_map)




    Eval.generate_text(file_sentence_map, IIDs_map, fold_n, group)


    r1, r2, rL = Eval.evaluate_content(test_set_fids, fold_n, group_prefix)
    print model_flag, "Rouge1=%.6f, Rouge2=%.6f, RougeL=%.6f" % (r1, r2, rL)


    agent_SentiSum, human_SentiSum, sentiSum_lift, agent_SentiMax, human_SentiMax, sentiMax_lift=Eval.evaluate_sentiment(test_set_fids, fold_n, group_prefix)
    test_end = time.time()
    test_duration = test_end-test_start

    print model_flag, "agent_Senti_Avg=%f, human_Senti_Avg=%f, sentiSum_lift=%f%%" % (agent_SentiSum, human_SentiSum, sentiSum_lift)
    print model_flag, "agent_Senti_Max=%f, human_Senti_Max=%f, sentiMax_lift=%f%%" % (agent_SentiMax, human_SentiMax, sentiMax_lift)
    print model_flag, "test duration (秒): ", test_duration

with tf.Session() as sess:

    if isTrain:
        sess.run(init)

        train_start = time.time()
        min_loss, best_loss_epoch, max_reward, best_reward_epoch, \
            collected_step_loss, collected_step_reward = train_model(aligned_source, aligned_summary, doc_vector_map, save_n_epoch, copy_n_epoch, group_prefix)
        train_end = time.time()

        global_avg_loss = np.average(collected_step_loss)
        global_avg_reward = np.average(collected_step_reward)
        max_optimize = collected_step_loss[0]-min(collected_step_loss)
        train_duration = train_end - train_start

        print
        print "global_avg_loss=%.6f, global_avg_reward=%.6f, max_optimize=%.6f" % (global_avg_loss, global_avg_reward, max_optimize)
        print "min_loss=%.6f, best_epoch=%d" % (min_loss, best_loss_epoch)
        print "max_reward=%.6f, best_epoch=%d" % (max_reward, best_reward_epoch)
        print "train duration (秒): ", train_duration

    if isTest:
        restore_path = None

        if isLoss:
            model_id = 56
            restore_path = "../resources/%s/model/min_loss/min_loss_%d.ckpt" % (group_prefix, model_id)
            model_flag = "min_loss_model"
        if isReward:
            model_id = 10
            restore_path = "../resources/%s/model/max_reward/max_reward_%d.ckpt" % (group_prefix, model_id)
            model_flag = "max_reward_model"
        assert restore_path is not None

        saver.restore(sess, save_path=restore_path)
        test_model(file_sentence_map, aligned_source, aligned_summary, sentence_amount_mask, group_prefix, model_flag)