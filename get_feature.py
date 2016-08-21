import json
import csv
import re
import numpy as np
import get_index
import codecs
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

proposal_file_path = "./data/list_h28_shizuoka_proposal.json"
words_file_path = "./data/list_h28_word_count.json"
#tfidf_path = "./data/h28_shizuoka_councilors"
vectorizer_path = "./data/h28_shizuoka_words"


def get_tfidf():
    # ドキュメント毎の単語リストを読みこむ
    with open(words_file_path, "r") as f:
        noun_info = json.load(f)

    r = re.compile(r"）")  # 括弧付きでspeakerに含まれる役職等の値を除外する

    speakers = []  # speakerのリストを生成
    for info in noun_info:
        speaker = info["speaker"]
        m = r.search(speaker)
        if m is None:
            speakers.append(info["speaker"])

    speakers = list(set(speakers))

    name_list = {}
    noun_lists = []
    for i, speaker in enumerate(speakers):
        noun_list = [d["nouns"] for d in noun_info if d["speaker"] == speaker]
        # speaker_list = [d["speaker"] for d in noun_lists if d["speaker"] == speaker]
        nouns = []  # [[][]]形式でスピーカーごと階層化されたリストをflatternし、スページで区切られた単語のリストを作る
        for w in noun_list:
            #ws = "\s".join(w)
            #print(ws)
            nouns.extend(w)
        nouns = " ".join(nouns)  # リスト内の単語リストを空白区切りの文字列に変換する。リストは["文字列", "文字列",,]に
        noun_lists.append(nouns)
        name_list[i] = speaker  # 序数でspeaker nameを保存し取り出せるように。

    # tf-idfの出力
    # TfidfVectorizerに発言者毎の単語リスト（スペース区切り）のリストを引数として渡す
    vectorizer = TfidfVectorizer(use_idf=True, max_df=20)
    tfidf = vectorizer.fit_transform(noun_lists).toarray()

    # ユニークな単語リストの生成
    terms = vectorizer.get_feature_names()

    with open("./data/words_list.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(terms)

    with open("./data/speaker_list.json", "w") as f:
        json.dump(name_list, f, ensure_ascii=False)

    np.savetxt("./data/tfidf_array.csv", tfidf, delimiter=",")
    #print(tfidf.toarray()[0])

    nouns = []
    for n in noun_lists:
        ns = n.split(" ")
        nouns.extend(ns)
    #print(len(nouns))  # 89,687語
    #print(get_index.word_count(nouns, 20))  #議事録内の単語の頻度を取得。 市、事業、静岡、地域などが当然多くカウントされる

    # 任意に設定したWebサイトの検索カテゴリより、各カテゴリtop 5個の「特徴的な見出し単語をリストとして取得
    site_index = get_index.get_docs(5)

    # 各発言者の特徴的なキーワードn件を取得
    proposal_words = []
    for i, f in enumerate(speakers):
        p = {}
        proposal_word = []
        for x in get_index.feature_words(terms, tfidf, i, 20):
            proposal_word.append(x)
        p["speaker"] = f
        p["words"] = proposal_word
        proposal_words.append(p)

    # proposal_wordsよりtf-idfの値を省いた各発言者の特徴語をマージした単語リストを生成
    proposal_words_k = []
    for item in proposal_words:
        d = ["".join(list(x.keys())) for x in item["words"]]
        proposal_words_k.extend(d)

    proposal_words_k.extend(site_index)
    words = list(set(proposal_words_k))

    # 「各発言者の特徴語」をマージしたリスト＋「特徴的な見出し語」リストの単語で、tf-idf値のtop n 発言者ランキングを生成
    # 特定の単語（wordsにふくまれる）のtermsのidxを取得しtfidfからspeakerごとの単語のtfidfの値を取得する
    features = {}  # 各キーワードのtfidf値
    top_n_speaker = {}  # 各キーワードのtfidfの値が高いspeaker top n
    n = 5
    for w in words:
        # site_indexのtop nがtermsに含まれないケースもある
        try:
            idx = terms.index(w)  # ある単語のインデックス
            # speakerごと単語のtfidfを取得。配列はspearkersに準ずる。
            tfidf_w = []
            for tfidf_array in tfidf:
                tfidf_w.append(tfidf_array[idx])
            features[w] = tfidf_w
            # ソートした配列で順番にspeaker: tfidf値のdictをリストに追加する。
            x = np.array(tfidf_w)
            t = x.argsort()[-n:][::-1]  # tfidfの値でソートしソートされたindexの配列tを返す
            top_n_speaker[w] = [{speakers[idx]: x[idx]} for idx in t]
        except:
            features[w] = []
            top_n_speaker[w] = []

    get_index.save2jsonfile(top_n_speaker, "./data/h28_top_5_speaker.json")

    # 各speakerの重要なキーワード；tfidfのdictを生成。
    # キーワードが重複する際にはtfidfの最大値を採用。
    top_n_ws = [w["words"] for w in proposal_words]
    top_n_w = []
    for w in top_n_ws:
        x = [k for k in w]
        top_n_w.extend(x)

    top_n_w_list = []
    top_n_w_ks =[]
    for w in top_n_w:
        #print(list(w.keys())[0])
        l = []
        for x in top_n_w:
            if list(w.keys())[0] == list(x.keys())[0]:
                l.append(x)
                #l.append(list(x.values())[0])

        top_n_w_dict = {}
        if list(w.keys())[0] not in top_n_w_ks:
            if len(l) == 1:
                # dictに追加の場合
                top_n_w_dict = {list(w.keys())[0]: list(l[0].values())[0]}
                top_n_w_list.append(top_n_w_dict)
                top_n_w_ks.append(list(w.keys())[0])

            elif len(l) > 1:
                m = max(list(l[0].values()))
                top_n_w_dict = {list(w.keys())[0]: m}
                top_n_w_list.append(top_n_w_dict)
                top_n_w_ks.append(list(w.keys())[0])

    get_index.save2jsonfile(top_n_w_list, "./data/h28_top5_keywords.json")


'''
def get_text():
    with open(proposal_file_path, "r") as f:
        text_info = json.load(f)

    speakers = []  # speakerのリストを生成
    r = re.compile(r"）")

    for noun_list in text_info:
        speaker = noun_list["speaker"]
        m = r.search(speaker)
        if m is None:
            speakers.append(noun_list["speaker"])


def tokenize(text):
    tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/ipadic/")
    tagger.parse('')
    return tagger.parse(text)


# dict(発言者毎に文を格納）を渡すとtf-idfが返る関数
def get_tfidf(token_dict):
    #
    vectorizer = TfidfVectorizer(tokenizer=get_nouns, min_df=1, max_df=50)
    corpus = []  # 文字列のリスト
    tfs = vectorizer.fit_transform(token_dict)  # リスト
    return tfs


def get_nouns(text):
    # 取得した文を名詞のみtuple(["a","b",,,])化して返す関数
    tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/ipadic/")
    tagger.parse('')
    node = tagger.parseToNode(text)
    nouns = []
    while node:
        if node.feature.split(",")[0] == "名詞":
            nouns.append(node.surface)
        node = node.next

    nouns = tuple(nouns)
    return nouns
'''

get_tfidf()
