# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:06:48 2024
ゼミ:マルチクラス・シングルラベルのSVM main03
@author: Kazuaki Kishida
"""

# main03: マルチクラス・シングルラベルのSVMを実行
# CMD.exe Promptでの準備→「コピー」してから、CMD.exeに移り、「Cntl」+「v」
#  >pip install scikit-learn==1.2.1

DATA_PATH = 'Data/' #データのディレクトリを指定
encoding = 'utf-8' #文字コード　→シフトJISは'shift_jis'（非推奨）
orgfile = DATA_PATH + 'Demo_textdata2.csv' #入力ファイル（語分割前の元の文）
infile = DATA_PATH + 'Demo_textdata3b.csv' #入力ファイル（1行に1文書の語の集合）
infile2 = DATA_PATH + 'user_input_data3b.csv' #入力ファイル(語分割後の集合)
ansfile = DATA_PATH + 'ans_v2.csv' #'ans_test.csv' # 'ans_test.csv' #'Demo_answer.csv' #正解ラベルファイル
rstfile = DATA_PATH + 'Demo_classpred.csv' #評価データの予測と正解を格納
random_seed = 123 #乱数のシード：これを変更すると結果が変わる
p_tr = 0.70 #訓練データの割合 -> 1.00
p_vd = 0.00 #検証データの割合→訓練・評価の2分割の場合、p_vd = 0.0と設定
p_ts = 0.30 #評価データの割合 0.00
min_df = 0.1 # default = 2 #最小出現文書数（これを下回る場合、語は抽出されない）
n_features = -1 # default = 5 #選択する語の総数（tf-idfの重み上位）
cv_split = 2 #交差検証の数

#正解ラベルの設定
import pandas as pd
import collections
df_label = pd.read_csv(ansfile, header=None) #正解ラベルファイルの読み込み（UTF-8）
#print(df_label.head()) #確認
dic_label = {} #キー：ラベル（文字列）、値：ID（通し番号:0,1,...）
for label in df_label.iloc[:,0]:
    if not label in dic_label:
        dic_label[label] = len(dic_label) #ラベルに通し番号を付与
print(f'正解ラベル={dic_label}')
labels = [dic_label[label] for label in df_label.iloc[:,0]]
print(f'正解ラベルデータ={labels}')
c = collections.Counter(labels) #正解ラベル別レコード数の集計
print('正解ラベルの出現文書数 = ', c)
n_labels = len(c) #正解ラベルの種類の数（目的クラス数）を取得

#データを訓練・検証・評価に3分割：訓練・評価の2分割の場合、p_vd = 0.0と設定
import numpy as np
import csv
corpus = [] #CSVファイルの読み込み
with open(infile, 'r', encoding='utf_8_sig') as f: # infile2で読んだcorpusがzip(X3, Y3)のX3になる
    reader = csv.reader(f)
    for row in reader:
        corpus.append([w for w in row if len(w) > 0])

# ============ infle2 =================
x3 = []
with open(infile2, 'r', encoding='utf_8_sig') as f:
    reader = csv.reader(f)
    for row in reader:
        x3.append([w for w in row if len(w) > 0])
y3 = [0 for i in range (len(x3))]
# =========================================
print(f'文書件数 = {len(corpus)} in ファイル{infile}')
np.random.seed(random_seed) #乱数のシードの設定
by_class = collections.defaultdict(list) #正解ラベルごとのデータのリスト
for idx, label in enumerate(labels): 
    by_class[label].append(idx) #正解ラベルごとにテキストデータを追加
X1, y1, X2, y2, X3, y3 = [], [], [], [], [], [] #訓練／検証／評価
ind_X1, ind_X2, ind_X3 = [], [], [] #各データの元の通し番号
for label, inds in sorted(by_class.items()): #ラベルごとに3分割
    np.random.shuffle(inds) #1つのラベルでデータをシャッフル
    n_tr = int(len(inds)*p_tr) #訓練レコード数
    n_vd = int(len(inds)*p_vd) #検証レコード数
    for x in inds[:n_tr]: #訓練データ
        X1.append(corpus[x]) #追加
        y1.append(label) #追加
        ind_X1.append(x)
    for x in inds[n_tr:n_tr+n_vd]: #検証データ
        X2.append(corpus[x]) #追加
        y2.append(label) #追加
        ind_X2.append(x)
    for x in inds[n_tr+n_vd:]: #評価データ
        X3.append(corpus[x]) #追加
        y3.append(label) #追加
        ind_X3.append(x)
print('訓練レコード数=', len(X1))
print('検証レコード数=', len(X2))
print('評価レコード数=', len(X3))

#特徴選択：最小出現文書数とtf-idf重みの合計値で語を選択
from txt_preproc import Features
ft = Features(X1, min_df=min_df)
mat, selected_words = ft.select(n_features=n_features) #抽出実行
print('選択された語=', selected_words)

#特徴選択により、語を含まなくってしまった文書を除去
errorrow = [] #消去する行番号を記録
for index, row in mat.iterrows():
    values = row[0:len(row)]
    if sum(values) == 0.0:
        errorrow.append(index) #消去対象
print('削除されたレコード番号 =', errorrow)
X_train = mat.drop(mat.index[errorrow]) #データから消去
print(f'訓練用データX=\n{X_train}')
df_labels = pd.DataFrame(y1) #訓練集合のラベルデータをPandasに変換
df_labels = df_labels.drop(mat.index[errorrow]) #文書ラベルから削除
y_train = list(df_labels[0]) #list型に変換
print(f'訓練用データy=\n{y_train}')

#SVMの実行（グリッドサーチと分類器の学習）
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
param_list = [0.001, 0.01, 0.1, 1, 10, 100] #グリッドサーチ用（適宜設定）
best_score = 0.0 #評価指標の最大値（初期化）
best_parameters  = {} #パラメータを記録するための辞書型
for C in param_list: #ハイパーパラメータの候補値ごとに繰り返す
    svm = SVC(kernel='linear', C=C) #分類器の学習
    #cross_val_scoreによる交差検証：cvで交差検証の分割数を指定
    scores = cross_val_score(svm, X_train, y_train, cv=cv_split)
    score = np.mean(scores)#評価指標の値の平均を算出
    if score > best_score: #値がそれまでの最大値を超えた場合
        best_score = score #最大値を書き換え
        best_parameters = {'C' : C} #パラメータの値を記録しておく
print(f'最良スコア={best_score}') #表示
print(f'最良パイパーパラメータ={best_parameters}')
#最終的な分類器の学習
svm = SVC(kernel='linear', **best_parameters) #モデルの指定(線型)
svm.fit(X_train, y_train) #分類器の学習
#訓練データでの学習内容の確認
y_pred = svm.predict(X_train) #訓練データ自体で予測してみる
print(classification_report(y_train, y_pred)) #結果表示
#混同行列：行が正解、列が分類器による予測
df_cfm = pd.DataFrame(confusion_matrix(y_train, y_pred))
print(df_cfm) #混同行列の表示

def tf_counter(doc): #語のリストを受け取り、tfを計算するメソッド
    dic = {} #辞書　キー：語、値：tf
    for word in doc:
        if word in dic: #語が辞書に既存かどうか確認
            dic[word] += 1 #既存の場合1を足し込む
        else:
            dic[word] = 1 #新規の語では1を入れる（新規）
    return dic
  
#選択された語の辞書を2つ作成：通し番号IDとidfがそれぞれの値
terms, idf = ft.get_idf() #TfidfVectorizerからidfを取得
dic_idf ={word: idf for word, idf in zip(terms, idf)} #キー：語、値：idf
vocab = {word: no for no, word in enumerate(selected_words)} #キー：語、値：ID

#評価用のテキストデータから、分類器への入力データを作成
import numpy as np
X_data = [] #評価用の文書データを格納する
y_ans = [] #正解ラベルのリスト
inds = [] #元ファイルでの通し番号
n_docs = 0
for doc, label in zip(X3, y3): #1文書ずつ処理 
    dic_tf = tf_counter(doc) #評価データでtfベクトルを求める
    row = np.zeros(len(vocab)) #ベクトルの初期化
    for word in dic_tf: #1語ずつ処理
        idx = vocab.get(word) #分類器の語彙を検索
        if idx != None: #分類器に含まれる語の場合
            row[idx] = dic_tf[word] * dic_idf[word] #tf-idfの計算
    if row.sum() > 0.0: #評価レコードが分類器中の語を1つ以上含んでいる
        X_data.append(list(row))  #文書データに追加
        y_ans.append(label)
        inds.append(ind_X3[n_docs])
    else: #分類器中の語を1つも持たない場合
        print(f'警告：有効な特徴を含まないため、評価から除外={doc}')
    n_docs += 1
X_eval = pd.DataFrame(X_data, columns=X_train.columns.values)
y_pred = svm.predict(X_eval) #評価用文書データで予測を行う
print(f'予測結果=\n{y_pred}') #確認
print(f'評価指標=\n{classification_report(y_ans, y_pred)}') #結果表示
#混同行列：行が正解、列が分類器による予測
df_cfm = pd.DataFrame(confusion_matrix(y_ans, y_pred))
print(f'混同行列=\n{df_cfm}') #混同行列の表示

#評価データの予測と正解をファイル出力
df_org = pd.read_csv(orgfile, header=None) #正解ラベルファイルの読み込み（UTF-8）
id2label = {dic_label[k]:k for k in dic_label} #ラベルの逆引き辞書作成
y_pred_label = [id2label[v] for v in y_pred] #ラベル番号を文字列に置換
y_ans_label =  [id2label[v] for v in y_ans] #ラベル番号を文字列に置換
texts = [df_org.iloc[x, 0] for x in inds] #元のファイルからテキストを取り出す
df_pred = pd.DataFrame(zip(y_pred_label, y_ans_label, inds, texts), 
                       columns=['予測', '正解', 'no（0から開始）', 'テキスト'])
df_pred.to_csv(rstfile, encoding='utf_8_sig') #ファイル出力

