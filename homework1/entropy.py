import jieba
import math


# 一元模型词频统计
def get_tf(tf_dic, words):
    for i in range(len(words)):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1


# 二元模型词频统计
def get_bigram_tf(tf_dic, words):
    for i in range(len(words) - 1):
        tf_dic[(words[i], words[i + 1])] = tf_dic.get((words[i], words[i + 1]), 0) + 1


# 三元模型词频统计
def get_trigram_tf(tf_dic, words):
    for i in range(len(words) - 2):
        tf_dic[((words[i], words[i + 1]), words[i + 2])] = tf_dic.get(((words[i], words[i + 1]), words[i + 2]), 0) + 1


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    stop_path = "./data/cn_stopwords.txt"
    stop_words = []
    with open(stop_path, "r", encoding="utf-8") as f:
        for line in f:
            stop_words.append(line.strip())

    punc_path = "./data/cn_punctuation.txt"
    punc_words = []
    with open(punc_path, "r", encoding="utf-8") as f:
        for line in f:
            punc_words.append(line.strip())

    txt_path = "./data/inf.txt"
    with open(txt_path, 'r') as f:
        book_names = []
        for line in f:
            book_names = line.split(',')

    char_count = 0
    context = []
    for name in book_names:
        with open('data/' + name + '.txt', 'r', encoding='gb18030') as f:
            single_book = f.read()
            single_book = single_book.replace("本书来自www.cr173.com免费txt小说下载站", "")
            single_book = single_book.replace("更多更新免费电子书请关注www.cr173.com", "")
            single_book = single_book.replace("\n", "")
            single_book = single_book.replace("\u3000", "")
            single_book = single_book.replace(" ", "")
            context.append(single_book)

    count_all = 0
    count_use = 0
    # 分词模型
    print("结巴分词")
    words_tf = {}
    bigram_tf = {}
    trigram_tf = {}
    split_words = []
    for line in context:
        single_split_words = []
        for x in jieba.cut(line):
            if x not in punc_words:
                count_all = count_all + len(x)
            if x not in stop_words:
                single_split_words.append(x)
                count_use = count_use + len(x)
        get_tf(words_tf, single_split_words)
        get_bigram_tf(bigram_tf, single_split_words)
        get_trigram_tf(trigram_tf, single_split_words)
        split_words.append(single_split_words)

    print("语料库字数(去除符号):", count_all)
    print("去除停用词字数:", count_use)
    split_count = 0
    for single_split_words in split_words:
        split_count = split_count + len(single_split_words)
    print("分词个数:", split_count)
    print("平均词长:", round(count_use / split_count, 5))

    words_len = sum([dic[1] for dic in words_tf.items()])
    bigram_len = sum([dic[1] for dic in bigram_tf.items()])
    trigram_len = sum([dic[1] for dic in trigram_tf.items()])
    print("一元模型长度:", words_len)
    print("二元模型长度:", bigram_len)
    print("三元模型长度:", trigram_len)

    entropy = [-(uni_word[1] / words_len) * math.log(uni_word[1] / words_len, 2) for uni_word in words_tf.items()]
    print("基于jieba分割的一元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")

    entropy = []
    for bi_word in bigram_tf.items():
        jp_xy = bi_word[1] / bigram_len  # 计算联合概率p(x,y)
        cp_xy = (bi_word[1] / bigram_len) / (words_tf[bi_word[0][0]] / words_len)  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    print("基于jieba分割的二元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")

    entropy = []
    for tri_word in trigram_tf.items():
        jp_xy = tri_word[1] / trigram_len
        cp_xy = (tri_word[1] / trigram_len) / (bigram_tf[tri_word[0][0]] / bigram_len)
        entropy.append(-jp_xy * math.log(cp_xy, 2))
    print("基于jieba分割的三元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")

    # 按字分词
    print("按字分词")
    words_tf = {}
    bigram_tf = {}
    trigram_tf = {}
    split_words = []
    count_word = 0
    for line in context:
        single_split_words = []
        for x in line:
            if x not in stop_words:
                single_split_words.append(x)
                count_word = count_word + 1
        get_tf(words_tf, single_split_words)
        get_bigram_tf(bigram_tf, single_split_words)
        get_trigram_tf(trigram_tf, single_split_words)
        split_words.append(single_split_words)

    print("去除停用词字数：", count_word)
    words_len = sum([dic[1] for dic in words_tf.items()])
    bigram_len = sum([dic[1] for dic in bigram_tf.items()])
    trigram_len = sum([dic[1] for dic in trigram_tf.items()])
    print("一元模型长度:", words_len)
    print("二元模型长度:", bigram_len)
    print("三元模型长度:", trigram_len)

    entropy = [-(uni_word[1] / words_len) * math.log(uni_word[1] / words_len, 2) for uni_word in words_tf.items()]
    print("基于jieba分割的一元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")

    entropy = []
    for bi_word in bigram_tf.items():
        jp_xy = bi_word[1] / bigram_len  # 计算联合概率p(x,y)
        cp_xy = (bi_word[1] / bigram_len) / (words_tf[bi_word[0][0]] / words_len)  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    print("基于jieba分割的二元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")

    entropy = []
    for tri_word in trigram_tf.items():
        jp_xy = tri_word[1] / trigram_len
        cp_xy = (tri_word[1] / trigram_len) / (bigram_tf[tri_word[0][0]] / bigram_len)
        entropy.append(-jp_xy * math.log(cp_xy, 2))
    print("基于jieba分割的三元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")
