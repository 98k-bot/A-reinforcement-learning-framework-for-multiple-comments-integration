#coding=utf-8

import pickle as pkl
import numpy as np
import random
import math
import os
from sklearn.model_selection import KFold

resource_dir = "../resources/"
final_model_save_path = '../resources/%s/model/final'

def loadObj(path):
    file = open(resource_dir+path, "rb")
    obj = pkl.load(file)
    file.close()
    return obj

def backup(group_prefix, filename, obj):
    path = resource_dir+"%s/"%group_prefix+"sum_logs/"+filename
    file = open(path, "wb")
    pkl.dump(obj,file)
    file.flush()
    file.close()
    #print "备份 "+filename+" 完成 !"

def getFidToIndexMap(file_index_map):
    fid_index_map = {}
    for index, fid in file_index_map.items():
        fid_index_map[fid] = index
    # print fid_index_map
    return fid_index_map


def extractInstances(train_set_fids, fid_index_map, aligned_source):
     indexs = [fid_index_map[fid] for fid in train_set_fids]
     # print indexs
     # print train_set_fids
     return aligned_source[indexs]

def createValidSentenceMap(train_set_fids, maxSen, validSentenceMap):
    mask = [[1]*validSentenceMap[fid]+[0]*(maxSen-validSentenceMap[fid]) for fid in train_set_fids]

    for index in range(len(mask)):
        fid = train_set_fids[index]
        assert sum(mask[index]) == validSentenceMap[fid]

    # print mask
    return np.array(mask)

def updateSummary(aligned_summary, sids, sentenceVectors, train_set, maxSen):
    for index in range(len(train_set)):
        fid = train_set[index]
        sid = sids[index]
        if sid == maxSen+1:
            assert np.sum(aligned_summary[index]) == 0
            continue
        vector = sentenceVectors[fid][sid]
        aligned_summary[index][sid] += vector
    return aligned_summary

def randomSelect(sentenceCount_map, train_set):
    sentence_count = [sentenceCount_map[fid] for fid in train_set]
    sids = [random.randint(0, amount-1) for amount in sentence_count]
    return sids

def epsilon_greedy(q_values, train_set, sentenceCount_map, step, decay_step, eps_min=0.1, eps_max=1.0):
    epsilon = max(eps_min, eps_max-(eps_max-eps_min)*step/decay_step)
    if np.random.rand() < epsilon:
        #print "random select"
        return randomSelect(sentenceCount_map, train_set)
    else:
        # print "max select"
        return np.argmax(q_values, axis=-1)

def init_records(sids, records, train_set):
    for index in range(len(train_set)):
        sid = sids[index]
        fid = train_set[index]
        if records.get(fid,None) == None:
            records[fid] = [[], []]
        records[fid][-1].append(sid)
    return records

def compute_reward_add_records(records, doc_vectors, summary, current_iteration, new_selected_sids, train_set,
                                             sentenceVectorMap, sentiSumMap,
                                             file_size, sentence_size, beta_reward, maxSen, embed_size, compress):

    for index in range(len(train_set)):

        #把新选的句子加进来
        fid = train_set[index]
        Int_snapshot = records[fid][-1]
        action = new_selected_sids[index]
        Int_snapshot.append(action)

        #确保当前summary已经选了3条句子
        if len(Int_snapshot) <= 2:
            continue

        #判断summary size是否超出压缩率
        current_sentence_sizes = [sentence_size[fid][sid] for sid in Int_snapshot]
        current_Int_size = sum(current_sentence_sizes)
        size_threshold = compress * file_size[fid]

        #说明选择当前句子后已经满了，清空summary，清空Int_snapshot
        if current_Int_size > size_threshold:
            #清空summary, 在这个时候就先更新, 但是其他的不变
            summary[index] = np.zeros((maxSen, embed_size))
            #清空Int_snapshot
            records[fid][-1] = []
            #清空当前选的句子
            new_selected_sids[index] = maxSen+1
            #清空当前reward
            reward = 0.0
            move_on = 0
        else:

            #stratety1: 前后比较
            # sid1, sid2, sid3 = Int_snapshot[-3:]
            #
            # content_diff_a = computeContentDiff(sentenceVectorMap[fid][sid2], sentenceVectorMap[fid][sid1])
            # content_diff_b = computeContentDiff(sentenceVectorMap[fid][sid3], sentenceVectorMap[fid][sid2])
            #
            # sentiment_diff_a = computeSentiDiff(sentiSumMap[fid][sid2], sentiSumMap[fid][sid1])
            # sentiment_diff_b = computeSentiDiff(sentiSumMap[fid][sid3], sentiSumMap[fid][sid2])
            # # print sentiment_diff_a, sentiment_diff_b
            #
            # content_acc = compute_acc_rate(content_diff_b, content_diff_a)
            # sentiment_acc = compute_acc_rate(sentiment_diff_b, sentiment_diff_a)
            # # print content_acc, sentiment_acc

            #strategy2:比较summary状态
            '''
            后一个diff不比前一个diff小即可
            这样还是可以捕捉当summary较满的时候这样的reward变化
            '''
            # assert -1 not in Int_snapshot and len(Int_snapshot) > 2
            # state1 = Int_snapshot[:-2]
            # state2 = Int_snapshot[:-1]
            # state3 = Int_snapshot
            #
            # state1_vector = [sentenceVectorMap[fid][sid] for sid in state1]
            # state1_vector = np.sum(state1_vector, axis=0)
            #
            # state2_vector = [sentenceVectorMap[fid][sid] for sid in state2]
            # state2_vector = np.sum(state2_vector, axis=0)
            #
            # state3_vector = [sentenceVectorMap[fid][sid] for sid in state3]
            # state3_vector = np.sum(state3_vector, axis=0)
            #
            # content_diff_a = computeContentDiff(state2_vector, state1_vector)
            # content_diff_b = computeContentDiff(state3_vector, state1_vector)
            # content_acc = compute_acc_rate(content_diff_b, content_diff_a)
            # if content_acc <= 0.01:
            #     content_acc = 0
            #
            # state1_senti = [sentiSumMap[fid][sid] for sid in state1]
            # state1_senti = sum(state1_senti)
            #
            # state2_senti = [sentiSumMap[fid][sid] for sid in state2]
            # state2_senti = sum(state2_senti)
            #
            # state3_senti = [sentiSumMap[fid][sid] for sid in state3]
            # state3_senti = sum(state3_senti)
            #
            # senti_diff_a = computeSentiDiff(state2_senti, state1_senti)
            # senti_diff_b = computeSentiDiff(state3_senti, state1_senti)
            # sentiment_acc = compute_acc_rate(senti_diff_b, senti_diff_a)
            # if sentiment_acc <= 0.01:
            #     sentiment_acc = 0

            #此处添加策略3的reward计算方法，注意:此时move_on始终是1
            ''' strategy3: 遵循ASRL的 score计算方法'''
            assert -1 not in Int_snapshot and len(Int_snapshot) > 2
            state1 = Int_snapshot[:-2]
            state2 = Int_snapshot[:-1]
            state3 = Int_snapshot
            doc_vector = doc_vectors[fid]

            state1_score = compute_ASRL_score(doc_vector, state1, sentenceVectorMap[fid])
            state2_score = compute_ASRL_score(doc_vector, state2, sentenceVectorMap[fid])
            state3_score = compute_ASRL_score(doc_vector, state3, sentenceVectorMap[fid])

            content_diff_a = state2_score - state1_score
            content_diff_b = state3_score - state1_score
            content_acc = compute_acc_rate(content_diff_b, content_diff_a)
            if content_acc <= 0.01:
                content_acc = 0

            state1_senti = [sentiSumMap[fid][sid] for sid in state1]
            state1_senti = sum(state1_senti)

            state2_senti = [sentiSumMap[fid][sid] for sid in state2]
            state2_senti = sum(state2_senti)

            state3_senti = [sentiSumMap[fid][sid] for sid in state3]
            state3_senti = sum(state3_senti)

            senti_diff_a = computeSentiDiff(state2_senti, state1_senti)
            senti_diff_b = computeSentiDiff(state3_senti, state1_senti)
            sentiment_acc = compute_acc_rate(senti_diff_b, senti_diff_a)
            if sentiment_acc <= 0.01:
                sentiment_acc = 0

            reward = beta_reward*content_acc + (1-beta_reward)*sentiment_acc
            move_on = 1

        if isRewardAbnormal(reward):
            reward = 0.0

        reward = round(reward, 6)
        records[fid][0].append((current_iteration-1, action, current_iteration, reward, move_on))

    return records, summary, new_selected_sids

def compute_ASRL_score(doc_vector, IIDs, sentence_vectors, weight=0.9):

    #计算redundancy
    if len(IIDs) == 1:
        redundancy = 0.0
    else:
        redundancy = 0.0
        for i in range(len(IIDs)):
            for j in range(len(IIDs)):
                if i >= j: continue
                A_sid = IIDs[i]
                B_sid = IIDs[j]
                A = sentence_vectors[A_sid]
                B = sentence_vectors[B_sid]
                sim = compute_cosin_sim(A, B)
                redundancy += sim

    #计算计算relative
    relative = 0.0
    for i in range(len(IIDs)):
        sid = IIDs[i]
        x = sentence_vectors[sid]
        relative += compute_cosin_sim(x, doc_vector) + 1.0/(sid+1)

    #计算reward
    reward = weight*relative - (1-weight)*redundancy
    return reward

def compute_cosin_sim(A, B):
    num = np.dot(A.T,B)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    sim = num / denom
    sim = 0.5 + sim*0.5
    return sim

def extractInstancesFromRecords(records, train_set, total_iter):
    #最后训练的是Q_values
    file_amount = len(train_set)
    instances = init_instances(total_iter, file_amount)

    for index in range(len(train_set)):
        fid = train_set[index]
        logs = records[fid][0]
        for j in range(len(logs)):
            start, action, next, reward, move_on = logs[j]
            instances[start][1][index] = action
            instances[start][-2][index] = reward
            instances[start][-1][index] = move_on

    return instances, file_amount

def checkInstances(records, instances, total_iteration, file_amount):

    assert len(instances) == total_iteration

    for _ in range(1, 4):
        start = random.randint(0, total_iteration-1)
        shuffled_actions = sorted(instances[start][1])
        shuffled_rewards = sorted(instances[start][-2])
        shuffled_moves = sorted(instances[start][-1])

        actions = []
        rewards = []
        moves = []

        for fid in records.keys():
            logs,_=records[fid]
            for item in logs:
                prev, action, _, reward, move = item
                if prev == start:
                    actions.append(action)
                    rewards.append(reward)
                    moves.append(move)

        sub = file_amount-len(actions)
        actions += [-1]*sub
        rewards += [0.0]*sub
        moves += [0]*sub

        actions = sorted(actions)
        rewards = sorted(rewards)
        moves = sorted(moves)

        checkSame(shuffled_actions, actions)
        checkSame(shuffled_rewards, rewards)
        checkSame(shuffled_moves, moves)

def checkSame(obj1, obj2):
    assert len(obj1) == len(obj2)
    for index in range(len(obj1)):
        assert obj1[index] == obj2[index]

def init_instances(total_iter, file_amount):
    instances = []
    for start in range(0, total_iter):
        instances.append([start, [-1]*file_amount, start+1, [0.0]*file_amount, [0]*file_amount])
    return instances

def isRewardAbnormal(reward):
    flag = False
    if math.isnan(reward):
        flag = True
    if math.isinf(reward):
        flag = True
    return flag

def computeContentDiff(after, prev):
    return np.sum(np.square(after-prev))

def computeSentiDiff(after, prev):
    assert after >=0 and prev >0
    return after-prev

def compute_acc_rate(after, prev):
    growth = after-prev
    if prev == 0:
        prev = 0.01

    acc_rate = growth*1.0 / prev
    return acc_rate

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def getActionMask(maxSen, fileAmount, actions, trainSet):
    mask = np.zeros((fileAmount, maxSen))
    for index in range(len(trainSet)):
        action = actions[index]
        if action == -1:continue
        mask[index][action] = 1
    return mask

def split_folds(fold_num, file_index_map, n_split=5):
    fids = file_index_map.keys()
    fids = np.array(fids)
    kf = KFold(n_splits=n_split)
    folds = kf.split(X=fids)

    index = 1
    for train_index, test_index in folds:
        if index != fold_num:
            index += 1
            continue
        else:
            X_train, X_test = fids[train_index], fids[test_index]
            # print X_train, len(X_train)
            # print X_test, len(X_test)
            break

    assert index == fold_num
    return X_train, X_test

def isLossDecreased(logs, limitAmount=5):
    assert len(logs) == limitAmount
    MIN = logs[0]
    position = 0

    for index in range(len(logs)):
        value = logs[index]
        if value < MIN:
            position = index
            MIN = value

    if position == 0:
        isDecreased = False
    else:
        isDecreased = True
    logs = []
    return logs, isDecreased

def compute_doc_vectors(data_set, sentence_vectors):
    doc_vectors = {}
    for index in range(len(data_set)):
        fid = data_set[index]
        vectors = sentence_vectors[fid]
        doc_vector = np.mean(vectors, axis=0)
        doc_vectors[fid] = doc_vector
        print doc_vector.shape, doc_vector
    return doc_vectors
