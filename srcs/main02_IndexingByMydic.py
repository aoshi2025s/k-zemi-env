# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:59:03 2022
ゼミ：自作辞書中の語の抽出 main02 
@author: Kazuaki Kishida
"""

#main02:自作辞書と突き合わせて、その見出し語のみを抽出
# ※自作辞書中の語を含まない文書は、コンソール表示される。その場合には元データを手作業で修正
#   →消滅した文書は取り除いて調整し、処理をし直すのがおススメ

DATA_PATH = '/root/srcs/Data/' #データのディレクトリを指定
encoding = 'utf_8_sig' #これ以外だと結果が不正になる可能性あり（変えないこと）
dicfile = DATA_PATH + 'Demo_dict.csv' #辞書ファイル
    #dicfileは列Aに抽出する語、列Bに変換後の語（表記のゆれの修正用）、ヘッダーなし
workfile1 = DATA_PATH + 'Demo_textdata2.csv' #入力ファイル：main01の結果または単純なCSVファイル
workfile2 = DATA_PATH + 'Demo_textdata3a.csv' #出力ファイル

#(1)自作辞書の設定（ファイルからの読み込み）
from txt_preproc import Indexing
import pandas as pd
df_vocab = pd.read_csv(dicfile, header=None, encoding=encoding) #自作辞書の読み込み
print('自作辞書中での見出し語数=', len(df_vocab))
print(df_vocab.head())
indexing = Indexing() #インスタンス生成
indexing.set_dic(df_vocab.iloc[:, 0]) #自作辞書のセット（列Aの見出し語）

#(2)第1段階：自作辞書の見出し語をテキストから抽出（語分割）
df = pd.read_csv(workfile1, header=None, encoding=encoding) #辞書の読み込み
docs, removed = indexing.dic_match(df.iloc[:, 0]) #辞書見出しとの突合せ処理
print(f'文書集合=\n{docs}')
print('削除された文書の番号=', removed) #

#(3)統制語への変換
conv = {} #統制語への変換用辞書
for row in df_vocab.itertuples():
    conv[row[1]] = row[2] #キー：列Aの見出し語、値：列Bの統制語
newdocs = [] #変換後の文書集合
for doc in docs: 
    wordlist = doc.split(' ') #語分割
    newdoc = []
    for word in wordlist:
        if len(word) > 0:
            newdoc.append(conv[word]) #変換してから追加
    newdocs.append(newdoc)
print(f'変換後の文書集合=\n{newdocs}')

#(4)ファイル出力
import csv
with open(workfile2, 'w', encoding=encoding) as f: 
     writer = csv.writer(f, lineterminator='\n')
     for row in newdocs:
         writer.writerow(row)