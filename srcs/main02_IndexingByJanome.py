# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 07:33:05 2022
ゼミ：Janomeでの語分割 main02
@author: Kazuaki Kishida
"""

# main02: Janomeによる語分割
# CMD.exe Promptでの準備→「コピー」してから、CMD.exeに移り、「Cntl」+「v」
#  >pip install janome==0.4.2
#  >pip install mojimoji==0.0.12 →mojimojiのインストールがうまくいかない場合にはネットで調べること
# ※半角は全角に変換され、数字は削除される
# ※(1)ストップワードの除去、(2)表記のゆれの統一、(3)切られすぎた文字の連結が可能

DATA_PATH = 'Data/' #データのディレクトリを指定
encoding = 'utf_8_sig' #これ以外だと結果が不正になる可能性あり（変えないこと）
pos = ['名詞', '形容詞'] #Janomeで抽出する品詞のリスト
#JanomeはIPADIC（IPA辞書）に基づいている。この辞書での品詞体系については，以下を参照
# https://hayashibe.jp/tr/mecab/dictionary/ipadic
stopwfile = DATA_PATH + 'stopwords_slothlib.txt' #Janomeの結果から除く語（不要語）のリスト
cnctfile = DATA_PATH + 'Demo_concat.csv' #結合する語の組み合わせを指定したファイル
thesfile = DATA_PATH + 'JpnThesaurus_utf.csv' #表記のゆれを統一するためのシソーラス
workfile1 = DATA_PATH + 'Demo_textdata2.csv' #入力ファイル：main01の結果または単純なCSVファイル
workfile2 = DATA_PATH + 'Demo_textdata3b.csv' #出力ファイル

#定義(1)：Janomeのトークナイザ（関数）
import re
import mojimoji as mj
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter, LowerCaseFilter, ExtractAttributeFilter
def janome_run(rawtexts, pos=['名詞']):
    #数字は削除
    rawtexts2 = [re.sub(r'[\d,]+(\.\d+)?', '', data) for data in rawtexts] 
    #語分割のための形態素解析:指定した品詞だけ抽出
    jm = Analyzer(token_filters=[POSKeepFilter(pos), LowerCaseFilter(), 
                  ExtractAttributeFilter('surface')])
    return [list(jm.analyze(mj.han_to_zen(data))) for data in rawtexts2]
    #以下はmojimojiがインストールできない場合（半角→全額を諦める場合）
    #return [list(jm.analyze(data)) for data in rawtexts2]
#定義(2)ストップワードの除去（クラス）
class Stopword:
    #参考：『機械学習・深層学習による自然言語処理入門』第4章   
    def __init__(self, stwlist, encoding="utf-8"):
        #リストの読み込み：リストは単に不要語を改行で区切って縦に並べたもの
        with open(stwlist, 'r', encoding=encoding) as f:
            self.stopwords = [w.strip() for w in f]
            self.stopwords = set(self.stopwords)
    def remove(self, words):
        #不要語の除去
        words = [w for w in words if w not in self.stopwords]
        return words
    
#実行(1)：Janomeでの語分割
import pandas as pd
df = pd.read_csv(workfile1, header=None) #単純なCSVファイルの読み込み
jm_out = janome_run(df.iloc[:, 0], pos) #Janomeでの語分割→「0」ならば列A
print(jm_out)
stpw = Stopword(stopwfile) #インスタンス生成
jm_out2 = [stpw.remove(data) for data in jm_out] #ストップワード除去
print(jm_out2)

#実行(2)：表記のゆれの統一 例：「たぬき」→「狸」　※辞書必要
from txt_preproc import WordReplacing
wr = WordReplacing(thesfile, encoding=encoding) #インスタンス生成
wr_out = wr.run(jm_out2, encoding=encoding)
print(wr_out)

#実行(3)：分割されすぎている場合に結合　例：「情報,検索」→「情報検索」　※辞書必要
from txt_preproc import WordConcatenating
wc = WordConcatenating(cnctfile, encoding=encoding) #インスタンス生成
wc_out = wc.run(wr_out, encoding=encoding) #実行
print(wc_out)

#実行(4)：ファイル出力
outf = wc_out #適宜、出力したい処理結果を指定（jm_out, jm_out2, wr_out, wc_out）
import csv
with open(workfile2, 'w', encoding=encoding) as f: 
     writer = csv.writer(f, lineterminator='\n')
     for row in outf:
         writer.writerow(row)
         