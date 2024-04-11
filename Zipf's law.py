import jieba
import collections
import matplotlib.pyplot as plt

if __name__ == "__main__":
    stop_path = "./data/cn_stopwords.txt"
    stop_words = []
    with open(stop_path, "r", encoding="utf-8") as f:
        for line in f:
            stop_words.append(line.strip())

    punc_path = "./data/cn_punctuation.txt"
    punc_words = []
    with open(stop_path, "r", encoding="utf-8") as f:
        for line in f:
            punc_words.append(line.strip())

    txt_path = "./data/inf.txt"
    with open(txt_path, 'r') as f:
        book_names = []
        for line in f:
            book_names = line.split(',')

    char_count = 0
    context = []
    count_all = 0
    for name in book_names:
        with open('data/' + name + '.txt', 'r', encoding='gb18030') as f:
            single_book = f.read()
            single_book = single_book.replace("本书来自www.cr173.com免费txt小说下载站", "")
            single_book = single_book.replace("更多更新免费电子书请关注www.cr173.com", "")
            single_book = single_book.replace("\n", "")
            single_book = single_book.replace("\u3000", "")
            single_book = single_book.replace(" ", "")
            count_all = count_all + len(single_book)
            context.append(single_book)

    split_words = []
    for line in context:
        for x in jieba.cut(line):
            if x not in stop_words:
                split_words.append(x)

    word_counts = collections.Counter(split_words)
    top_30_words = word_counts.most_common(30)
    print(top_30_words)

    rank = int(1)
    ranks = []
    freqs = []
    for _, value in top_30_words:  # 0 ('的', 87343)
        ranks.append(int(rank + 1))
        freqs.append(int(value))
        rank += 1

    plt.loglog(ranks, freqs)
    plt.title("结巴分词", fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.xlabel('词语名次', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.ylabel('词语频数', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.show()
