# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:06:48 2024
ゼミ：マルチクラス・シングルラベルのBERT main02
@author: Kazuaki Kishida
"""

# CMD.exe Promptでの準備→「コピー」してから、CMD.exeに移り、「Cntl」+「v」
#  >pip install mojimoji==0.0.12 →mojimojiのインストールがうまくいかない場合にはネットで調べること
#  >pip install torch==2.0.1
#  >pip install　transformers==4.31.0
#  >pip install scikit-learn==1.2.1

DATA_PATH = '/root/srcs/Data/' #データのディレクトリを指定
encoding = 'utf-8' #文字コード　→シフトJISは'shift_jis'（非推奨）
infile = DATA_PATH + 'Demo_textdata2.csv' #入力ファイル：main01の結果または単純なCSVファイル
ansfile = DATA_PATH + 'Demo_answer.csv' #正解ラベルファイル
rstfile = DATA_PATH + 'Demo_classpred.csv' #評価データの予測と正解を格納
random_seed = 123 #乱数のシード：これを変更すると結果が変わる
p_tr = 0.70 #訓練データの割合
p_vd = 0.30 #検証データの割合→通常は0.15
p_ts = 0.15 #評価データの割合→通常は0.15
batch_size = 2  # 一括して処理する文書数、全文書数よりも大きいとエラー、通常は8、16、32など
max_length = 128  # 最大512
max_epochs = 5  # 学習の際の最大エポック数
lr = 1e-5  # 学習率
# 事前学習済み日本語BERT
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

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

#テキスト中の数字を「0」に一括変換、半角を全角に変換
import re
import mojimoji as mj
def txt_conv(text):
    data = re.sub(r'[\d,]+(\.\d+)?', '0', text) #数字を0に置換
    return mj.han_to_zen(data) #半角を全角に変換して戻す

#データを訓練・検証・評価に3分割：訓練・評価の2分割の場合、p_vd = 0.0と設定
import numpy as np
import csv
#CSVファイルの読み込み
with open(infile, 'r', encoding='utf_8_sig') as f:
    reader = csv.reader(f)
    corpus = [txt_conv(row[0]) for row in reader]
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
print(f'確認：訓練データ=\n{X1}')

#BERT実行の準備
from nn_tool import BertTool
import torch
from transformers import logging
bt = BertTool(m_name=MODEL_NAME)  # BERTの実行ツールを実装
# データローダの生成
dl_tr = bt.gen_dl(X1, y1, max_length=max_length,  # 訓練用
                  batch_size=batch_size, shuffle=True)
dl_vd = bt.gen_dl(X2, y2, max_length=max_length,  # 検証用
                  batch_size=batch_size, shuffle=False)

# デバイスの設定：NVIDIAのGPUを使うには、PC自体での設定が必要
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')

# BERTのファインチューニング
logging.set_verbosity_error()  # 警告文の出力抑制
tr_loss, vd_loss = bt.run_train_tc(dl_tr, dl_vd, device,  # 実行
                                   no_labels=n_labels,
                                   n_epochs=max_epochs, lr=lr)

# 最終エポック数を決定し、分類器の再学習を行う
final_epochs = 3  # 実行前に手入力→学習段階の検証データの損失関数の値で決定
bt = BertTool(m_name=MODEL_NAME)  # BERTの実行ツールを再実装
tr_loss, vd_loss = bt.run_train_tc(dl_tr, dl_vd, device,  # 実行
                                   no_labels=n_labels,
                                   n_epochs=final_epochs, lr=lr)

# BERTモデルの評価
# データローダの生成
dl_ev = bt.gen_dl(X3, y3, max_length=max_length, #評価用
                  batch_size=batch_size, shuffle=False)
y_pred, y_ans = [], []
with torch.no_grad():  # 勾配を計算しない
    for batch in dl_ev:  # 評価データのミニバッチごとの処理
        b_in_ids = batch['input_ids'].to(device)
        b_in_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        outputs = bt.model(b_in_ids, token_type_ids=None,
                           attention_mask=b_in_mask,
                           labels=b_labels)
        # 予測を実行し、リストに追加
        y_pred += torch.argmax(outputs.logits.to(device),
                               1).tolist()
        y_ans += b_labels.tolist()  # 正解ラベルの取得

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(f'予測結果=\n{y_pred}') #確認
print(f'評価指標=\n{classification_report(y_ans, y_pred)}') #結果表示
# 混同行列：行が正解、列が分類器による予測
df_cfm = pd.DataFrame(confusion_matrix(y_ans, y_pred))
print(f'混同行列=\n{df_cfm}') #混同行列の表示

#評価データの予測と正解をファイル出力
id2label = {dic_label[k]:k for k in dic_label} #ラベルの逆引き辞書作成
y_pred_label = [id2label[v] for v in y_pred] #ラベル番号を文字列に置換
y_ans_label =  [id2label[v] for v in y_ans] #ラベル番号を文字列に置換
texts = [corpus[x] for x in inds] #元のファイルからテキストを取り出す
df_pred = pd.DataFrame(zip(y_pred_label, y_ans_label, inds, texts), 
                       columns=['予測', '正解', 'no（0から開始）', 'テキスト'])
df_pred.to_csv(rstfile, encoding='utf_8_sig') #ファイル出力