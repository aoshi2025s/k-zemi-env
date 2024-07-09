# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:06:48 2024
ゼミ：マルチラベルのBERT main02
@author: Kazuaki Kishida
"""

# CMD.exe Promptでの準備→「コピー」してから、CMD.exeに移り、「Cntl」+「v」
#  >pip install mojimoji==0.0.12 →mojimojiのインストールがうまくいかない場合にはネットで調べること
#  >pip install torch==2.0.1
#  >pip install　transformers==4.31.0
#  >pip install　datasets==2.13.1
#  >pip install scikit-learn==1.2.1

DATA_PATH = '/root/srcs/Data/' #データのディレクトリを指定
encoding = 'utf-8' #文字コード　→シフトJISは'shift_jis'（非推奨）
infile = DATA_PATH + 'Demo_textdata2.csv' #入力ファイル：main01の結果または単純なCSVファイル
ansfile = DATA_PATH + 'Demo_answer_mlabel.csv' #正解ラベルファイル:マルチラベル
rstfile = DATA_PATH + 'Demo_classpred.csv' #評価データの予測と正解を格納
random_seed = 123 #乱数のシード：これを変更すると結果が変わる
p_tr = 0.60 #訓練データの割合→通常は0.70
p_vd = 0.35 #検証データの割合→通常は0.15
p_ts = 0.15 #評価データの割合→通常は0.15
batch_size = 2  # 一括して処理する文書数、全文書数よりも大きいとエラー、通常は8、16、32など
max_length = 128  # 最大512
n_train_epochs = 5 #学習時のエポック数
TH = 0.5 #TH以上の確率をもつラベルを付与する
lr = 2e-5  # 学習率
# 事前学習済み日本語BERT
model_ckpt = "cl-tohoku/bert-base-japanese" #日本語BERT

#テキスト中の数字を「0」に一括変換、半角を全角に変換
import re
import mojimoji as mj
def txt_conv(text):
    data = re.sub(r'[\d,]+(\.\d+)?', '0', text) #数字を0に置換
    return mj.han_to_zen(data) #半角を全角に変換して戻す

import csv
#CSVファイルの読み込み
with open(infile, 'r', encoding='utf_8_sig') as f:
    reader = csv.reader(f)
    corpus = [txt_conv(row[0]) for row in reader] #テキストデータ

#正解ラベルの設定：マルチラベルの場合
import pandas as pd
import collections
import numpy as np
df_labels = pd.read_csv(ansfile, header=None) #正解ラベルファイルの読み込み（UTF-8）
labels = list(df_labels[0]) #listに変換
label_list = {} #正解ラベルの一覧
by_class = collections.defaultdict(list) #正解ラベルごとのデータのリスト
for idx, items in enumerate(df_labels[0]):
    label = items.split('/') #個別ラベルに分解
    if len(label) > 1: #ラベルが複数ある場合には無作為抽出
        k = np.random.randint(0, len(label)-1) #ラベルを1つ抽出
    else: #ラベルが1つのみの場合
        k = 0
    for item in label: #ラベルの一覧を作成（辞書型）
        if item not in label_list:
            label_list[item] = len(label_list)
    by_class[label[k]].append(idx) #正解ラベルごとにテキストデータを追加
X1, y1, X2, y2, X3, y3 = [], [], [], [], [], [] #訓練／検証／評価
ind_X1, ind_X2, ind_X3 = [], [], [] #各データの元の通し番号
for label, inds in sorted(by_class.items()): #ラベルごとに3分割
    np.random.shuffle(inds) #1つのラベルでデータをシャッフル
    n_tr = int(len(inds)*p_tr) #訓練レコード数
    n_vd = int(len(inds)*p_vd) #検証レコード数
    for x in inds[:n_tr]: #訓練データ
        X1.append(corpus[x]) #追加
        y1.append(labels[x]) #追加
        ind_X1.append(x)
    for x in inds[n_tr:n_tr+n_vd]: #検証データ
        X2.append(corpus[x]) #追加
        y2.append(labels[x]) #追加
        ind_X2.append(x)
    for x in inds[n_tr+n_vd:]: #評価データ
        X3.append(corpus[x]) #追加
        y3.append(labels[x]) #追加
        ind_X3.append(x)
print('訓練レコード数=', len(X1))
print('検証レコード数=', len(X2))
print('評価レコード数=', len(X3))
print('ラベル=', label_list)

def label_conv(label_list, labels): #文字列でのラベルを変換
    #labelsは1件の文書の正解ラベル　例：'図書館/経営/統計'
    items = labels.split('/') #各ラベルに分割
    mlabel = [0] * len(label_list) #ラベルの初期化
    for item in items:
        mlabel[label_list[item]] = 1 #該当個所「1」
    return mlabel #戻り値の例：[1,0,1,1]

def conv(x): #マルチラベルの変換　例：'図書館/経営'→[0,1,0,1]
    return label_conv(label_list, x)

#データをいったんPandasに変換し、そこからDatasetインスタンスを作成
from datasets import Dataset
df_tr = pd.DataFrame(list(zip(X1, y1)), 
                     columns=['text', 'label'])
df_tr['label'] = df_tr['label'].map(conv) #マルチラベルの変換
ds_tr = Dataset.from_pandas(df_tr) #訓練データのDataset作成
df_vd = pd.DataFrame(list(zip(X2, y2)), 
                     columns=['text', 'label']) 
df_vd['label'] = df_vd['label'].map(conv) #マルチラベルの変換
ds_vd = Dataset.from_pandas(df_vd) #検証データのDataset作成

#語分割とラベルデータの変換
from transformers import AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
#Datasetインスタンスの生成時のトークナイザ
def tokenize(record):
  text = record['text']
  encoding = tokenizer(text, padding='max_length', 
                       truncation=True, max_length=max_length)
  encoding['labels'] = record['label']
  return encoding
#エンコーディング：トークン化とマルチラベルデータの値の形式変換
def ds_encoding(ds): 
    dsenc = ds.map(tokenize, batched=True, 
                   remove_columns=ds.column_names)
    dsenc.set_format('torch')
    #マルチラベルのデータをintからfloatに変換
    dsenc = dsenc.map(lambda x: {'label_ids': 
                      x['labels'].to(torch.float)},
                      remove_columns=['labels'])
    return dsenc
#変換の実行
dsenc_tr = ds_encoding(ds_tr) #訓練データ
dsenc_vd = ds_encoding(ds_vd) #検証データ

#デバイスの設定：NVIDIAのGPUを使うには、PC自体での設定が必要
device = torch.device('cuda' if torch.cuda.is_available() 
                              else 'cpu') 
print(f'device = {device}')

#パラメータの設定
n_labels = len(label_list) #マルチラベルを構成するラベル個数
metric_name = "f1" #学習時の確認に使用する評価指標
from transformers import TrainingArguments
args = TrainingArguments(
    output_dir= DATA_PATH + "/results", #出力
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=lr, #学習率
    per_device_train_batch_size=batch_size, #バッチサイズ
    per_device_eval_batch_size=batch_size, #バッチサイズ
    num_train_epochs=n_train_epochs, #訓練時エポック数
    weight_decay=0.01, #減衰率
    load_best_model_at_end=True,
    metric_for_best_model=metric_name, #評価指標
    report_to='none'
)

#モデルの設定
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.\
            from_pretrained(model_ckpt, #マルチラベル分類
                problem_type='multi_label_classification', 
                num_labels=n_labels).to(device)

#BERTマルチラベル分類での学習の実行
from sklearn.metrics import f1_score, accuracy_score, \
                            precision_score, recall_score
from transformers import EvalPrediction
import torch
import numpy as np
##マルチラベリングにおける評価指標の計算
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, y_true, threshold=TH):
    sigmoid = torch.nn.Sigmoid() #シグモイド関数
    probs = sigmoid(torch.Tensor(predictions)) #スコア→確率
    y_pred = np.zeros(probs.shape) #マルチラベルの初期化
    y_pred[np.where(probs >= threshold)] = 1 #閾値より大きければ「1」
    f1_average = f1_score(y_true=y_true, #f1マクロ
                          y_pred=y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred) #正解率
    metrics = {'f1': f1_average, 'accuracy': accuracy}
    return metrics #指標の値を戻す

#Trainerで使われる評価指標の定義
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        y_true=p.label_ids)
    return result

##学習フェーズの定義と実行
from transformers import Trainer, ProgressCallback
trainer = Trainer(model, args,
    train_dataset=dsenc_tr, #訓練データ
    eval_dataset=dsenc_vd, #検証データ
    tokenizer=tokenizer, #トークナイザ
    compute_metrics=compute_metrics, #評価指標
    callbacks=[ProgressCallback()] #コールバック
)
trainer.train() #学習の実行
trainer.save_state() #評価指標の保存
trainer.save_model() #モデルの保存

#評価フェーズの実行
def encoding(model, X):
    prob_list = []
    for text in X: #評価レコードを1件ずつ処理して確率を計算
        encoding = tokenizer(text, return_tensors="pt") #トークン化
        encoding = {k: v.to(model.device) 
                        for k,v in encoding.items()}
        with torch.no_grad():  # 勾配を計算しない
            outputs = model(**encoding) #スコア計算
            sigmoid = torch.nn.Sigmoid() #シグモイド関数
            probs = sigmoid(outputs.logits.squeeze().cpu()) #確率化
            prob_list.append(probs.tolist()) #確率のリスト化
    return prob_list

#評価データのエンコーディング
prob_list = encoding(trainer.model, X3)

#評価データに対する正解ラベルの作成
y_ans = [conv(x) for x in y3]

#ラベル付与の決定と評価指標の計算
y_pred = []
for probs in prob_list:
    pred = [1 if p >= TH else 0 for p in probs] #ラベル付与
    y_pred.append(pred) #追加
print(f'予測結果=\n{y_pred}') #確認
print('acc = ', accuracy_score(y_ans, y_pred))
print('precision(macro) = ', precision_score(y_ans, 
                                y_pred, average='macro'))
print('recall(macro) = ', recall_score(y_ans, y_pred, 
                                       average='macro'))
print('f1(macro) = ', f1_score(y_true=y_ans, y_pred=y_pred, 
                               average='macro'))  

#評価データの予測と正解をファイル出力
id2label = {label_list[k]:k for k in label_list} #ラベルの逆引き辞書作成
def inv_conv(mlabel):
    s = ''
    for idx, k in enumerate(mlabel):
        if k > 0:
            s += id2label[idx] + '/'
    return s[:len(s)-1]
y_pred_label = [inv_conv(v) for v in y_pred] #ラベル番号を文字列に置換
y_ans_label =  [inv_conv(v) for v in y_ans] #ラベル番号を文字列に置換
texts = [corpus[x] for x in inds] #元のファイルからテキストを取り出す
df_pred = pd.DataFrame(zip(y_pred_label, y_ans_label, inds, texts), 
                       columns=['予測', '正解', 'no（0から開始）', 'テキスト'])
df_pred.to_csv(rstfile, encoding='utf_8_sig') #ファイル出力
