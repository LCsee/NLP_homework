import numpy as np
import os
from sklearn.cluster import KMeans
import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


def drop_stopwords(context, stop_words_list):
    line_new = []
    for word in context:
        if word in stop_words_list:
            continue
        elif word != ' ':
            line_new.append(word)
    return line_new


def preprocess_chinese_corpus(folder_path):
    corpus = []
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='gb18030') as f:
                content = f.read()
                content = content.replace(
                    "本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", "")
                content = content.replace("本书来自www.cr173.com免费txt小说下载站", "")
                content = content.replace(' ', '')
                content = content.replace('\u3000', '')
                tokenized_sentences = [jieba.lcut(sentence) for sentence in content.split('\n') if sentence.strip()]
                corpus = corpus + tokenized_sentences

    return corpus


def convert_para_to_vec(paragraph, model):
    # 预处理
    paragraph = paragraph.replace('\n', '')  # 去除换行符
    paragraph = paragraph.replace(' ', '')  # 去除空格
    paragraph = paragraph.replace('\u3000', '')  # 去除全角空白符
    # 停用词表
    stopwords_file_path = './/cn_stopwords.txt'
    stopword_file = open(stopwords_file_path, "r", encoding='utf-8')
    stop_words = stopword_file.read().split('\n')
    stopword_file.close()
    # 分词并去除停用词
    words = [word for word in jieba.lcut(paragraph) if word not in stop_words]
    # 计算词向量
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    # 取平均值作为段落向量
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def main():
    stop = open('./cn_stopwords.txt', encoding='utf-8')
    stop_words = list(stop.read().split("\n"))
    stop.close()

    corpus = preprocess_chinese_corpus("./corpus")

    all_words = [word for word_list in corpus for word in word_list]
    all_words = drop_stopwords(all_words, stop_words_list=stop_words)
    result = ' '.join(all_words)
    with open('./split_words.txt', 'w', encoding="utf-8") as f2:
        f2.write(result)

    # Word2Vec
    w2v_model = Word2Vec(sentences=LineSentence('./split_words.txt'), vector_size=200, window=5, min_count=5,
                         workers=20, epochs=10)
    w2v_model.save('./model/all_word2vec.model')

    model = Word2Vec.load('./model/all_word2vec.model')
    key_word = ["杨过", "段誉"]
    for word in key_word:
        similar_words = model.wv.most_similar(word, topn=5)
        print("与'" + word + "'最相似的词语: ", similar_words)

    word1 = "韦小宝"
    word2 = "鸠摩智"
    similarity_score = model.wv.similarity(word1, word2)
    print(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}")

    '''KMeans聚类'''
    # 获取词汇表中的所有词语及其向量
    words = list(model.wv.index_to_key)
    word_vectors = np.array([model.wv[word] for word in words])

    # 使用KMeans算法进行聚类
    num_clusters = 20  # 设定要分的簇数
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(word_vectors)

    # 获取每个词语对应的簇标签
    labels = kmeans.labels_

    # 打印每个簇中的词语
    clusters = {}
    for i in range(num_clusters):
        clusters[i] = []

    for word, label in zip(words, labels):
        clusters[label].append(word)

    with open("cluster.txt", "w", encoding="utf-8") as f:
        for cluster_id, cluster_words in clusters.items():
            # print(f"Cluster {cluster_id}: {', '.join(cluster_words)}")
            f.write(f"Cluster\n {cluster_id}: {', '.join(cluster_words)}\n")

    para1 = "自掌法练 成以来，直至此时，方遇到周伯通这等真正的强敌。 周伯通听说这是他自创的武功，兴致更高，说道：“正要见识见识！” 挥手而上，仍是只用左臂。杨过抬头向天，浑若不见，呼的一掌向自 己头顶空空拍出，手掌斜下，掌力化成弧形，四散落下。\
     周伯通知道这一掌力似穹庐，圆转广被，实是无可躲闪，当下举掌相 迎，“啪”的一下，双掌相交，不由得身子一晃，都只为他过于托大， 殊不知他武功虽然决不弱于对方，但一掌对一掌，却无不及杨过掌力 厚实雄浑。"
    para2 = "杨过倒持长剑，回掌相迎，砰的 一声响，两股巨力相交，两人同时一晃，木梯摇了几摇，几乎折断。 两人都是一惊，暗赞对手了得：“一十六年不见，他功力居然精进如 斯！” 杨过见情势危急，不能和他在梯上多拚掌力，长剑向上疾刺，或击小 腿，或削脚掌。法王身子在上，若出金轮与之相斗，则兵刃既短，俯 身弯腰实在大是不便，只得急奔上高台。杨过向他背心疾刺数剑，招 招势若暴风骤雨，但法王并不回头，听风辨器，一一举轮挡开，\
    便如 背上长了眼睛一般。杨过喝采道：“贼秃！恁的了得！” 法王刚刚踏上台顶回首就是一轮。杨过侧首让过，身随剑起，在半空 中扑击而下。法王举金轮一挡，左手银轮便往他剑上砸去。 "

    vec1 = convert_para_to_vec(para1, model)
    vec2 = convert_para_to_vec(para2, model)
    paragraph_similarity_score = cosine_similarity([vec1], [vec2])[0][0]
    print(f"段落间的语义相似度：{paragraph_similarity_score}")


if __name__ == "__main__":
    main()
