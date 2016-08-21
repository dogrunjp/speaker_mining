import urllib.request
import urllib
from urllib.parse import urlparse
from html.parser import HTMLParser
from bs4 import BeautifulSoup, BeautifulStoneSoup
import json
import csv
import re

"""
議事録ページから発言を抜き出し保存するモジュール
"""

# file path
target_url_list = "./data/h28_url_list.csv"
speach_data = "./data/list_h28_shizuoka_proposal.json"


def get_proposals():
    page_list = get_url()
    proposal_list = get_paragraph(page_list)
    save2csvfile(proposal_list)


# メタ情報付き議事録ページのURLデータから必用な情報を取得
def get_url():
    page_list = []
    with open(target_url_list, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダを飛ばす

        for row in reader:
            if len(row[0]) > 0:
                data_obj = {}
                data_obj["class"] = row[0]
                data_obj["term"] = row[1]
                data_obj["year"] = row[2]
                data_obj["month"] = row[3]
                data_obj["order"] = row[4]
                data_obj["url"] = row[5]
                page_list.append(data_obj)

    return page_list


def get_paragraph(page_list):
    proposal_list = []
    for i, item in enumerate(page_list):
        targetURL = item["url"]
        response = urllib.request.urlopen(targetURL)
        data = response.read()
        soup = BeautifulSoup(data, "html.parser")
        # a_name = soup.findAll("a")
        p_tag = soup.findAll("p")
        # pタグの中を一つ一つの発言に分解する（発言者ごと）

        for i, p in enumerate(p_tag):  # 本文中に福間列pタグは発言全体を囲うタグとフッタのタグ
            para = p.text  # 本文ページのpタグにかこまれた全テキスト
            proposal_pat = re.compile("◯[0-9一-龥]+.+\n")
            m = proposal_pat.findall(para)  #ページに含まれた、発言者から始まり、改行で終わる全てのテキスト
            speaker_pat = re.compile("(（[0-9一-龥]+）)")
            brackets_pat = re.compile("（|君）")
            space_pat = re.compile('\\u3000', re.MULTILINE)

            for match_item in m:
                proposal = {}
                m_s = re.split(speaker_pat, match_item)  #pタグ中のテキストが、発言ごとにリストに分割される

                if len(m_s) > 1:
                    proposal["text"] = re.sub(space_pat, "", m_s[2])
                    proposal["speaker"] = re.sub(brackets_pat, "", m_s[1])
                    proposal["class"] = item["class"]
                    proposal["year"] = item["year"]
                    proposal["month"] = item["month"]
                    proposal["order"] = item["order"]
                    proposal_list.append(proposal)

    return proposal_list


def save2csvfile(proposal_list):
    with open(speach_data, "w") as f:
        json.dump(proposal_list, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    get_proposals()
