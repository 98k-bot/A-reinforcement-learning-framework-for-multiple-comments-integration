# coding=utf-8

import numpy as np
from rouge import Rouge
import nltk
from nltk.stem import WordNetLemmatizer
from senticnet.senticnet import SenticNet as SenticNet
from nltk.corpus import wordnet
import os

def generate_IIDs(q_values, data_set, file_valid_sentenceCount_map):

    IIDs_map = {}
    for index in range(len(data_set)):
        fid = data_set[index]
        valid_sentence_amount = file_valid_sentenceCount_map[fid]
        file_q_values = q_values[index]
        results = file_q_values[0:valid_sentence_amount]
        assert len(results) == valid_sentence_amount

        ascend_sids = np.argsort(results)
        descend_sids = np.flip(ascend_sids, axis=-1)
        assert len(descend_sids) == len(results)



        if IIDs_map.get(fid, None) is None:
            IIDs_map[fid] = descend_sids

    return IIDs_map

def generate_text(file_sentence_map, IIDs_map, n_fold, group_prefix):

    for fid, sids in IIDs_map.items():
        human_summary_path = '../resources/%s/summary_human/%d.txt' % (group_prefix, fid)
        size_threshold = getDocSize(human_summary_path)

        current_file_content_track = []
        first_sid = sids[0]


        agent_summary_path = '../resources/%s/summary_agent/f%d/%d.txt' % (group_prefix, n_fold, fid)
        sentence = file_sentence_map[fid][first_sid]+" "
        current_file_content_track.append(sentence)
        file = open(agent_summary_path, "a+")
        file.write(sentence)
        file.flush()
        file.close()

        j = 1
        while True:
            current_size = getDocSize(agent_summary_path)
            if current_size >= size_threshold:
                current_file_content_track = current_file_content_track[0:-1]
                break
            if j >= len(sids):
                break

            sid = sids[j]
            sentence = file_sentence_map[fid][sid]+" "
            current_file_content_track.append(sentence)
            file = open(agent_summary_path, "a+")
            file.write(sentence)
            file.flush()
            file.close()
            j += 1

        writeFinalContent(current_file_content_track, agent_summary_path)



def writeFinalContent(tracks, path):
    text = ""
    for item in tracks:
        text += item
    file = open(path, 'w')
    file.write(text)
    file.flush()
    file.close()

def getSummaryString(path):
    file = open(path, "r")
    content = file.read()
    file.close()
    content = content.strip()
    return content

def evaluate_content(data_set, n_fold, group_prefix):
    rouge = Rouge()
    r1_scores = []
    r2_sorces = []
    rL_scores = []

    for index in range(len(data_set)):
        fid = data_set[index]
        prediction_path = '../resources/%s/summary_agent/f%d/%d.txt' % (group_prefix, n_fold, fid)
        y_prediction = getSummaryString(prediction_path)

        true_path = '../resources/%s/summary_human/%d.txt' % (group_prefix, fid)
        y_true = getSummaryString(true_path)

        rouge_score = rouge.get_scores(y_prediction, y_true)[0]
        rouge1 = rouge_score['rouge-1']['f']
        rouge2 = rouge_score['rouge-2']['f']
        rougeL = rouge_score['rouge-l']['f']

        r1_scores.append(rouge1)
        r2_sorces.append(rouge2)
        rL_scores.append(rougeL)

    r1 = np.average(r1_scores)
    r2 = np.average(r2_sorces)
    rL = np.average(rL_scores)
    return r1, r2, rL

def evaluate_sentiment(data_set, n_fold, group_prefix):
    prediction_text_sentiAvgs = []
    true_text_sentiAvgs = []

    prediction_sentence_sentiMaxs = []
    true_sentence_sentiMaxs = []

    for index in range(len(data_set)):
        fid = data_set[index]
        prediction_path = '../resources/%s/summary_agent/f%d/%d.txt' % (group_prefix, n_fold, fid)
        true_path = '../resources/%s/summary_human/%d.txt' % (group_prefix, fid)

        agent_size = getDocSize(prediction_path)
        summary_size = getDocSize(true_path)
        assert agent_size < summary_size

        y_prediction = getSummaryString(prediction_path)
        prediction_text_sentiAvg, prediction_sentence_SentiMax = getMaxSum_senti(y_prediction)
        prediction_text_sentiAvgs.append(prediction_text_sentiAvg)
        prediction_sentence_sentiMaxs.append(prediction_sentence_SentiMax)

        y_true = getSummaryString(true_path)
        true_text_sentiAvg, true_sentence_sentiMax = getMaxSum_senti(y_true)
        true_text_sentiAvgs.append(true_text_sentiAvg)
        true_sentence_sentiMaxs.append(true_sentence_sentiMax)

    avg_prediction_text_Senti = round(np.average(prediction_text_sentiAvgs), 6)
    avg_prediction_sentence_SentiMax = round(np.average(prediction_sentence_sentiMaxs), 6)

    avg_true_text_Senti = round(np.average(true_text_sentiAvgs), 6)
    avg_true_sentence_SentiMax = round(np.average(true_sentence_sentiMaxs), 6)

    # print prediction_text_sentiAvgs, avg_prediction_text_Senti
    # print true_text_sentiAvgs, avg_true_text_Senti
    #
    # print prediction_sentence_sentiMaxs, avg_prediction_sentence_SentiMax
    # print true_sentence_sentiMaxs, avg_true_sentence_SentiMax

    senti_avg_lift = (avg_prediction_text_Senti-avg_true_text_Senti)/avg_true_text_Senti * 100
    senti_avg_lift = round(senti_avg_lift, 6)

    senti_Max_lift = (avg_prediction_sentence_SentiMax-avg_true_sentence_SentiMax)/avg_true_sentence_SentiMax * 100
    senti_Max_lift = round(senti_Max_lift, 6)

    return avg_prediction_text_Senti, avg_true_text_Senti, senti_avg_lift, \
           avg_prediction_sentence_SentiMax, avg_true_sentence_SentiMax, senti_Max_lift

def getMaxSum_senti(text):



    wnl = WordNetLemmatizer()
    sn = SenticNet()
    sentences = nltk.sent_tokenize(text)

    text_sentiAvg = 0
    sentence_maxSenti = 0

    for index in range(len(sentences)):
        sentence = sentences[index].strip()
        sentence = sentence[0:-1]

        assert '.' not in sentence
        words = nltk.word_tokenize(sentence.lower())
        pos_tags = nltk.pos_tag(words)
        sentence_sentiSum = getSentenceSentiSum(pos_tags, wnl, sn)
        # print sentence_sentiSum,

        if sentence_sentiSum > sentence_maxSenti:
            sentence_maxSenti = sentence_sentiSum

        text_sentiAvg += sentence_sentiSum

    text_sentiAvg = text_sentiAvg / len(sentences)
    text_sentiAvg = round(text_sentiAvg, 6)
    sentence_maxSenti = round(sentence_maxSenti, 6)

    return text_sentiAvg, sentence_maxSenti

def getSentenceSentiSum(pos_tags, wnl, sn):
    sentence_sentiSum = 0

    for item in pos_tags:
        word, pos = item
        pos = getWordNetPOS(pos)

        if pos is not None:
            word = str(wnl.lemmatize(word, pos=pos))

            try:
                polarity_value = sn.polarity_intense(word)
                polarity_value = float(polarity_value)
            except Exception:

                polarity_value = 0.0
            sentence_sentiSum += abs(polarity_value)

    return sentence_sentiSum

def getWordNetPOS(pos):
    result = None
    if pos.startswith("N"):
        result = str(wordnet.NOUN)
    if pos.startswith("V"):
        result = str(wordnet.VERB)
    if pos.startswith("R"):
        result = str(wordnet.ADV)
    if pos.startswith("J"):
        result = str(wordnet.ADJ)
    return result

# 获取文件大小
def getDocSize(path):
    try:
        size = os.path.getsize(path)
    except Exception as err:
        print(err)
    return size

def findSynWords(word):
    result = wordnet.synsets(word)
    syn_words = []
    for i in result:
        syn_word = i.lemmas()[0].name()
        if word != syn_word:
            syn_words.append(syn_word)
    return syn_words


def grabPolarityValueAndIntenseBySynWords(sn, syn_words):
    has_found = False


    for syn_word in syn_words:
        try:
            polarity_value = float(sn.polarity_intense(syn_word))
            has_found = True
            break
        except Exception:
            pass

    if not has_found:
        polarity_value = 0.0

    return polarity_value