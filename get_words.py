import MeCab
import json
import re
import codecs
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

"""
形態粗解析モジュール
発言を形態素解析し名詞のみ取得し、メタデータと共にファイルに保存する
"""

proposal_file_path = "./data/list_h28_shizuoka_proposal.json"
words_file_path = "./data/list_h28_word_count.json"


def get_words():
    # 取得した議会の発言文字列からメタデータを付加した名詞のリスト生成する
    with open(proposal_file_path, "r") as f:
        jsondata = json.load(f)

    word_count_list = []
    for i,item in enumerate(jsondata):
        word_count_obj = {}
        word_count_obj["id"] = "h28_{}".format(i)
        word_count_obj["year"] = item["year"]
        word_count_obj["month"] = item["month"]
        word_count_obj["conference"] = item["class"]
        word_count_obj["speaker"] = item["speaker"]
        text = item["text"]
        nouns = get_nouns(text)
        word_count_obj["nouns"] = nouns
        word_count_list.append(word_count_obj)

    save2jsonfile(word_count_list)


def get_nouns(text):
    # Mecabで取得した文を名詞のみリスト化して返す関数
    tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/ipadic/")
    tagger.parse('')
    node = tagger.parseToNode(text)
    nouns = []
    while node:
        if node.feature.split(",")[0] == "名詞" and re.match('^[0-9]{1,}$', node.surface) == None:
            nouns.append(node.surface)
        node = node.next
    return nouns


def save2jsonfile(word_list):
    with open(words_file_path, "w") as f:
        json.dump(word_list, f, ensure_ascii=False)


# Mecabの動作テスト
def test_mecab():
    tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/ipadic/")
    node = tagger.parseToNode("三日月をせもたれにして魚釣り")
    keywords = []
    while node:
        if node.feature.split(",")[0] == "名詞":
            keywords.append(node.surface)
        node = node.next
    #print(tagger.parse("三日月をせもたれにして魚釣り"))
    print(keywords)


get_words()

