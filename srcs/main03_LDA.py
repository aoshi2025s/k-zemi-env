# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:06:48 2024
ゼミ：LDA main03
@author: Kazuaki Kishida
"""

# main03: HDP、LDAの実行
# CMD.exe Promptでの準備→「コピー」してから、CMD.exeに移り、「Cntl」+「v」
#  >pip install gensim==4.3.0

DATA_PATH = "/root/srcs/Data/" #データのディレクトリを指定
infile = DATA_PATH + 'Demo_textdata3b.csv' #入力ファイル（1行に1文書の語の集合）
orgfile = DATA_PATH + 'Demo_textdata2.csv' #入力ファイル（語分割前の元の文）
outfile = DATA_PATH + 'lda.csv' #結果ファイル：LDAによる文書クラスタリングの結果
n_mojis = 50 #クラスタリング結果を表示する際に、元のテキストを先頭から何字表示させるか

#ファイル（1行に1文書の語の集合）の読み込み→SVMのときと同じ、main02の出力ファイル
import csv
corpus = [] #CSVファイルの読み込み
with open(infile, 'r', encoding='utf_8_sig') as f:
    reader = csv.reader(f)
    for row in reader:
        corpus.append([w for w in row if len(w) > 0])
print(f'文書件数 = {len(corpus)} in ファイル{infile}')

#元の文を読み込んでおく（表示用）→main02への入力ファイル
import pandas as pd
df = pd.read_csv(orgfile, header=None) #単純なCSVファイルの読み込み

#HDPの実行
import gensim
dic = gensim.corpora.dictionary.Dictionary(corpus) #辞書の生成
docs = [dic.doc2bow(text) for text in corpus] #hdpとldaへの入力データを作成
#HDPモデルによるクラスタ数（潜在トピック数）の自動推定
hdp = gensim.models.hdpmodel.HdpModel(docs, dic) #HDPモデルの実行
rslts = [hdp[doc] for doc in docs] #文書ごとに、各トピックの確率を取得
topic_set = set() #トピックを記録
for rslt in rslts: #文書（テキスト）ごとに処理
    max_score = max(rslt, key=lambda x: x[1]) #最大値を取得
    topic_set.add(max_score[0]) #最大トピックの通し番号を記録
n_topics = len(topic_set) #トピック数（クラスタ数）を取得
print("推定された潜在トピック数=", n_topics) #推定されたクラスタ数をコンソール表示

#トピック数を手動で指定した場合には、以下の文を書き換えて、コメントアウト
#n_topics = 10

#LDAの実行
lda = gensim.models.ldamodel.LdaModel(corpus=docs, id2word=dic, #LDAの実行
                                      num_topics=n_topics, iterations=5000) 
clusters = [[] for i in range(n_topics)] #クラスタごとの元のテキストを格納
for score, text in zip(lda[docs], df[0]): #文書ごとに処理
    max_score = max(score, key=lambda x: x[1]) #最大値を取得
    clusters[max_score[0]].append(text) #当該クラスタに元テキストを保管
print(f'クラスタごとの主要語=\n{lda.print_topics()}') #クラスタごとに主要語をコンソール表示（参考）

#ファイルへの結果の書き出し
#予想される実行時エラー：PermissionError　書き出すファイルがExcelで表示されている→閉じる
tms = lda.show_topics(num_topics=n_topics, num_words=20, 
                      formatted=False) #トピックごとの語と確率を取得
with open(outfile, 'w', encoding='utf_8_sig') as f: #ファイル出力の実行   
    f.write('クラスタ番号,文書\n')
    for k in range(n_topics): #クラスタごとに元のテキストをファイル出力
        for text in clusters[k]:
            f.write(str(k+1) + ',' + text[:n_mojis] + '\n')
    f.write('クラスタ番号,語,確率\n')
    for dat in tms: #クラスタごとに語の確率を出力
        for tp in dat[1]:
            f.write(str(dat[0]+1) + ',' + tp[0] + ',' + 
                    str(tp[1]) + '\n')
