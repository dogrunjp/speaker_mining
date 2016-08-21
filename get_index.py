import json
import csv
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

target_files = ["shizuoka_15_fukushi.tsv", "shizuoka_15_kanko.tsv",
                "shizuoka_15_kyoiku.tsv", "shizuoka_15_saigai.tsv", "shizuoka_15_sangyo.tsv", "shizuoka_15_kankyo.tsv"]
words_file_path = "./data/shizuoka_15_web_words.json"
feature_words_path = "./data/shizuoka_15_feature_words.json"
feature_words_k_path = "./data/shizuoka_15_feature_word_keys.json"

"""
WARPのAPIで静岡市のHPを任意のカテゴリを設定して検索した際のキーワードから
カテゴリに特徴的な見出し語を生成する。
"""


def get_docs(n=10):
    # WARPから取得したTSVよりキーワードを文字列として取得する
    word_lists = []
    for i,file in enumerate(target_files):
        file_path = "./data/" + file
        site_feature = {}
        words = ""
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for i, row in enumerate(reader):
                if i == 0:
                    words = row[13].replace(",", " ")
                else:
                    words = words + "," + row[13].replace(",", " ")

        site_feature["category"] = file
        site_feature["text"] = words
        word_lists.append(site_feature)

    # 取得したキーワードをそのままカウントする場合
    #l = flatten(word_lists)
    #print(len(l))
    #word_count(l, 10)

    save2jsonfile(word_lists, words_file_path)
    w, t = get_tfidf(word_lists)

    # 各カテゴリの特徴的なキーワードn件の取得
    index_words = []
    for i, f in enumerate(target_files):
        index = {}
        index_word = []
        for x in feature_words(w, t, i, n):
            index_word.append(x)
        index["doc"] = f
        index["words"] = index_word
        index_words.append(index)

    # tf-idfの値を持たない特徴語のみの単語リストも生成
    index_words_k = []
    for item in index_words:
        d = ["".join(list(x.keys())) for x in item["words"]]
        index_words_k.extend(d)

    save2jsonfile(index_words, feature_words_path)
    save2jsonfile(index_words_k, feature_words_k_path)

    return index_words_k


def get_tfidf(word_info):
    """
    #ファイルからドキュメント毎の単語リストを読みこむ場合
    with open(words_file_path, "r") as f:
        word_info = json.load(f)
    """

    # list-dictの文字列部分のみとりだしリスト化
    word_list = []
    cat_list = {}
    for i, w in enumerate(word_info):
        word_list.append(w["text"])
        cat_list[i] = w["category"]  # 序数でspeaker nameを保存し取り出せるように。

    # tf-idfの出力
    # TfidfVectorizerにspeaker毎の単語リスト（スペース区切り）のリストを引数として渡す
    vectorizer = TfidfVectorizer(use_idf=True, max_df=3)
    tfidf = vectorizer.fit_transform(word_list).toarray()

    # 単語リストの確認
    terms = vectorizer.get_feature_names()

    with open("./data/index_words_list.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(terms)

    with open("./data/category_list.json", "w") as f:
        json.dump(cat_list, f, ensure_ascii=False)

    np.savetxt("./data/index_tfidf_array.csv", tfidf, delimiter=",")

    return terms, tfidf


def save2jsonfile(word_list, file_path):
    with open(file_path, "w") as f:
        json.dump(word_list, f, ensure_ascii=False)


# tf-idf の出力からi番目のドキュメントの特徴的な上位n語を取得
def feature_words(terms, tfidf, i, n):
    tfidf_array = tfidf[i]  # 各ドキュメントのtfidf array
    feature_word = tfidf_array.argsort()[-n:][::-1]
    # tfidf_array.argsort(): 各ドキュメントのtfidfのソート結果の配列のインデックス。昇順。
    # [::-1]:スライス表記の３つ目の引数は２つおきや逆順等のオプション。[::-1]は逆順
    # 昇順の末端よりn個のインデックスを逆順で取得するスライス。
    words = [{terms[idx]: tfidf_array[idx]} for idx in feature_word]
    # words_k = [terms[idx] for idx in feature_word]

    return words


def flatten(list):
    flat_list = []
    for item in list:
        flat_list.extend(item["text"].split(" "))

    return flat_list


def word_count(list, n):
    c = Counter(list)
    print(c.most_common(n))


get_docs()
