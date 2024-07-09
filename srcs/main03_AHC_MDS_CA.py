# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:06:48 2024
ゼミ：AHC、MDS、CA main03
AHC:凝集型階層的クラスタリング
MDS:多次元尺度構成法
CA:対応分析
@author: Kazuaki Kishida
"""

# main03: AHC、MDS、CAの実行
# CMD.exe Promptでの準備→「コピー」してから、CMD.exeに移り、「Cntl」+「v」
#  >pip install scipy==1.10.0
#  >pip install matplotlib==3.7.0
#  >pip install japanize_matplotlib==1.1.3
#  >pip install scikit-learn==1.2.1
#  >pip install mca==1.0.3

DATA_PATH = "/root/srcs/Data/" #データのディレクトリを指定
infile = DATA_PATH + 'Demo_textdata3b.csv' #入力ファイル（1行に1文書の語の集合）
orgfile = DATA_PATH + 'Demo_textdata2.csv' #入力ファイル（語分割前の元の文）
outfile = DATA_PATH + 'mds.csv' #MDSの結果ファイル
min_df = 2 #最小出現文書数（これを下回る場合、語は抽出されない）
n_features = 5 #選択する語の総数（tf-idfの重み上位）
n_mojis = 3 #文書クラスタリング結果を表示する際に、元のテキストを先頭から何字表示させるか

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
dlbs = [str(i+1)+':'+text[:n_mojis] #文書のラベルを生成
        for i, text in enumerate(df[0])]
df_dlbs = pd.DataFrame(dlbs) #文書のラベル

#特徴選択：最小出現文書数とtf-idf重みの合計値で語を選択
from txt_preproc import Features
ft = Features(corpus, min_df=min_df)
mat, selected_words = ft.select(n_features=n_features) #抽出実行
print('選択された語=', selected_words)

#特徴選択により、語を含まなくってしまった文書を除去
errorrow = [] #消去する行番号を記録
for index, row in mat.iterrows():
    values = row[0:len(row)-1] 
    if sum(values) == 0.0:
        errorrow.append(index) #消去対象
print('削除されたレコード番号 =', errorrow)
X_train = mat.drop(mat.index[errorrow]) #データから消去
print(f'データX=\n{X_train}')
df_dlbs = df_dlbs.drop(mat.index[errorrow]) #文書ラベルから削除
print(f'文書=\n{df_dlbs[0]}')

#(1)AHC:凝集型階層的クラスタリング

#Ward法によるクラスタリングを実行する関数
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import japanize_matplotlib #このモジュールは明示的には使用しないが、この1行は必要
def ahc_ward(X, labels, cut_v=1.0):
    X = X / np.linalg.norm(X, axis = 1, #行ベクトルを単位ベクトルに変換
                           keepdims=True)
    rslt = linkage(X, method = "ward", metric = "euclidean")
    dendrogram(rslt, labels=labels, #デンドログラム生成
               orientation = 'right')
    plt.show() #再描画
    pt = fcluster(rslt, cut_v, criterion='distance') #距離1.0で切断
    flat_p = []
    for clst, label in zip(pt, labels): #平坦な分割（カット）
        flat_p.append((clst, label))
    return flat_p #カットによる平坦な分割結果を戻す
        
#(1-1)AHCのWard法による文書のクラスタリングの実行
X = X_train.to_numpy() #PandasのDataFrameをNumpyndarrayに変換
flat_p = ahc_ward(X, list(df_dlbs[0]), cut_v=1.0)
print(f'文書の平坦な分割=\n{flat_p}')

#(1-2)AHCのWard法による語のクラスタリングの実行
flat_p = ahc_ward(X.T, selected_words, cut_v=1.2)
print(f'語の平坦な分割=\n{flat_p}')

#(2)MDS:多次元尺度構成法

#MDSの実行
from sklearn.manifold import MDS
from plotter import mds_plt
def mds(X, labels, n_comps=2):
    X = X / np.linalg.norm(X, axis = 1, #行ベクトルを単位ベクトルに変換
                           keepdims=True)
    D = 1.0-X @ X.T #1-cosを距離とする
    mds_ = MDS(n_components=n_comps, metric=True, 
               dissimilarity='precomputed', 
               normalized_stress='auto')
    dis = mds_.fit_transform(D) #MDSの実行
    mds_plt(dis, labels, mds_.stress_, font_size=16) #プロットの実行
    df1 = pd.DataFrame(dis) #座標データ
    df2 = pd.DataFrame(labels) #ラベルデータ
    return pd.concat([df2, df1], axis=1) #座標とラベルを戻す

#(2-1)文書についてのMDS実行
df_doc = mds(X, list(df_dlbs[0]))
print(f'文書の座標=\n{df_doc}')

#(2-2)語についてのMDS実行
df_term = mds(X.T, selected_words)
print(f'語の座標=\n{df_term}')

#(3)CA:対応分析

#特徴選択された語に限定して、tfをカウント
from sklearn.feature_extraction.text import CountVectorizer
newcorpus, newdoc_labels = [], []
features = set(selected_words)
for text, label in zip(corpus, df[0]):
    row = [word for word in text if word in features]
    if len(row) > 0:
        newcorpus.append(row)
        newdoc_labels.append(label[:n_mojis])
skcorpus = []
for text in newcorpus: #X1は訓練データ
    s = text[0] #先頭の語
    for i in range(1, len(text)):
        if len(text[i]) > 0: #空の文字列でない場合
            s += (' ' + text[i]) #追加
    skcorpus.append(s) #リストに追加
vecs = CountVectorizer(ngram_range=(1, 1), analyzer='word')
X = vecs.fit_transform(skcorpus)
df_cross = pd.DataFrame(X.toarray(), 
                        columns=vecs.get_feature_names_out())
print(f'クロス集計（文書×語のtf）=\n{df_cross}')

import mca
mca_rslt = mca.MCA(df_cross, benzecri=False) #一般化特異値分解の実行
AQ = mca_rslt.fs_r(N=2) #行のカテゴリの主座標
BQ = mca_rslt.fs_c(N=2) #列のカテゴリの主座標
Q2 = mca_rslt.L #特異値の2乗

from plotter import ca_plt
Q = np.sqrt(Q2)[:2]
#主座標（Principal coordinates）
ca_plt(AQ, BQ, Q, list(df_dlbs[0]), df_cross.columns)
#バイプロット
ca_plt(AQ, BQ*(1/Q), np.sqrt(Q2), list(df_dlbs[0]), df_cross.columns,
       line='column')



