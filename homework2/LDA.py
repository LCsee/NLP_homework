import warnings
import gensim
from pprint import pprint
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import numpy as np
import os
import random
import jieba.posseg as psg
import jieba
# from pprint import pprint
from sklearn.svm import SVC
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def is_chinese_words(words):
    for word in words:
        if u'\u4e00' <= word <= u'\u9fa5':
            continue
        else:
            return False
    return True


def cut_words(contents):
    cut_contents = jieba.lcut(contents)
    # cut_contents = map(lambda s: list(psg.cut(s)), contents)
    # cut_contents = [char for char in contents]
    # cut_contents = map(word_filter, list(cut_contents))
    # return list(cut_contents)
    return cut_contents


def drop_stopwords(context, stop_words_list):
    line_new = []
    for word in context:
        if word in stop_words_list:
            continue
        elif word != ' ':
            line_new.append(word)
    return line_new


def extract_paras(para_num, token_num, book_para_jieba, book_para_char, stop_words_list, cut_option):
    file_path = r"data/"
    # with open(file_path + "inf.txt", "r") as f:
    #     book_names = f.read().split(",")
    book_names = ['碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '射雕英雄传', '神雕侠侣', '书剑恩仇录',
                  '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记']
    book_names_id = {}
    for k in range(len(book_names)):
        book_names_id[book_names[k]] = k

    corpus = []
    src_labels = []
    for file_name in book_names:
        if cut_option == 'word':
            paragraphs = book_para_jieba[file_name]
            curr = []
            for words in paragraphs:
                curr.extend(words)
                if len(curr) < token_num:
                    continue
                else:
                    corpus.append(curr[0:token_num])
                    src_labels.append(file_name)
                    curr = []
        elif cut_option == 'char':
            paragraphs = book_para_char[file_name]
            curr = []
            for words in paragraphs:
                curr.extend(words)
                if len(curr) < token_num:
                    continue
                else:
                    corpus.append(curr[0:token_num])
                    src_labels.append(file_name)
                    curr = []

    dataset = []
    sampled_labels = []
    para_num_per_book = int(para_num / len(book_names)) + 1
    for label in book_names:
        label_paragraphs = [paragraph for paragraph, paragraph_label in zip(corpus, src_labels) if
                            paragraph_label == label]
        if len(label_paragraphs) < para_num_per_book:
            label_paragraphs = label_paragraphs * int(para_num_per_book / len(label_paragraphs) + 1)
        sampled_index_list = np.random.choice(len(label_paragraphs), para_num_per_book, replace=False)
        # sampled_paragraphs = np.random.choice(label_paragraphs, para_num_per_book, replace=False)
        sampled_paragraphs = []
        for index in sampled_index_list:
            sampled_paragraphs.append(label_paragraphs[index])
        dataset.extend(sampled_paragraphs)
        sampled_labels.extend([book_names_id[label]] * para_num_per_book)

    dataset = dataset[0:para_num]
    sampled_labels = sampled_labels[0:para_num]

    return dataset, sampled_labels


def main(token_num, topic_num, stop_words_list, book_para_jieba, book_para_char):
    para_num = 1000
    book_names = ['碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '射雕英雄传', '神雕侠侣', '书剑恩仇录',
                  '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记']

    dataset, labels = extract_paras(para_num, token_num, book_para_jieba, book_para_char, stop_words_list, 'word')
    id2word = corpora.Dictionary(dataset)
    dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.1,
                                                                              random_state=42)

    train_corpus = [id2word.doc2bow(text) for text in dataset_train]
    test_corpus = [id2word.doc2bow(text) for text in dataset_test]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=id2word, num_topics=topic_num,
                                                random_state=100, update_every=1, chunksize=1000, passes=10,
                                                alpha='auto', per_word_topics=True, dtype=np.float64)

    train_cla = []
    test_cla = []
    for i, item in enumerate(test_corpus):
        tmp = lda_model.get_document_topics(item)
        init = np.zeros(topic_num)
        for index, v in tmp:
            init[index] = v
        test_cla.append(init)

    for i, item in enumerate(train_corpus):
        tmp = lda_model.get_document_topics(item)
        init = np.zeros(topic_num)
        for index, v in tmp:
            init[index] = v
        train_cla.append(init)

    print("word")
    accuracy = np.mean(
        cross_val_score(RandomForestClassifier(), train_cla + test_cla, labels_train + labels_test, cv=10))
    print(f'Accuracy: {accuracy:.2f}')

    dataset, labels = extract_paras(para_num, token_num, book_para_jieba, book_para_char, stop_words_list, 'char')
    id2word = corpora.Dictionary(dataset)
    dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.1,
                                                                              random_state=42)
    train_corpus = [id2word.doc2bow(text) for text in dataset_train]
    test_corpus = [id2word.doc2bow(text) for text in dataset_test]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=id2word, num_topics=topic_num,
                                                random_state=100, update_every=1, chunksize=1000, passes=10,
                                                alpha='auto', per_word_topics=True, dtype=np.float64)

    train_cla = []
    test_cla = []
    for i, item in enumerate(test_corpus):
        tmp = lda_model.get_document_topics(item)
        init = np.zeros(topic_num)
        for index, v in tmp:
            init[index] = v
        test_cla.append(init)

    for i, item in enumerate(train_corpus):
        tmp = lda_model.get_document_topics(item)
        init = np.zeros(topic_num)
        for index, v in tmp:
            init[index] = v
        train_cla.append(init)

    print("char")
    accuracy = np.mean(
        cross_val_score(RandomForestClassifier(), train_cla + test_cla, labels_train + labels_test, cv=10))
    print(f'Accuracy: {accuracy:.2f}')


if __name__ == '__main__':
    file_path = r"data/"
    book_names = ['碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣', '书剑恩仇录',
                  '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记']
    book_names_id = {}
    for k in range(len(book_names)):
        book_names_id[book_names[k]] = k

    stop = open(file_path + 'cn_stopwords.txt', encoding='utf-8')
    stop_words = stop.read().split("\n")
    stop_words_list = list(stop_words)
    stop_words_list.append("\u3000")

    book_para_jieba = {}
    book_para_char = {}
    for file_name in book_names:
        print(file_name)
        with open(file_path + "/" + file_name + ".txt", "r", encoding='gb18030') as file:
            all_text = file.read()
            all_text = all_text.replace("本书来自www.cr173.com免费txt小说下载站", "")
            all_text = all_text.replace("更多更新免费电子书请关注www.cr173.com", "")
            all_text = all_text.replace("\u3000", "")
            paragraphs = all_text.split("\n")
            book_jieba = []
            book_char = []
            for para in paragraphs:
                if para == '':
                    continue
                book_jieba.append(drop_stopwords(jieba.lcut(para), stop_words_list))
                book_char.append(drop_stopwords([char for char in para], stop_words_list))
            book_para_jieba[file_name] = book_jieba
            book_para_char[file_name] = book_char

    token_list = [20, 100, 500, 1000]
    # token_list = [3000]
    topic_list = [5, 20, 100, 500]
    for token_n in token_list:
        for topic_n in topic_list:
            print(token_n, " ", topic_n)
            main(token_n, topic_n, stop_words_list, book_para_jieba, book_para_char)
