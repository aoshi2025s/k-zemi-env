# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:27:03 2024

@author: kz_ki
"""

# CMD.exe Promptでの準備→「コピー」してから、CMD.exeに移り、「Cntl」+「v」
#  >pip install torch==2.0.1
#  >pip install　transformers==4.31.0

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, \
    BertForSequenceClassification
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

class BertTool:  # BERT実行のためのツール(日本語のみ)
    # コンストラクタ
    def __init__(self, m_name=None):
        self.tokenizer = \
            BertJapaneseTokenizer.from_pretrained(m_name)
        self.bert_prt = m_name
    # データローダの生成
    def gen_dl(self, data, labels, max_length=128,
               batch_size=32, shuffle=True):
        dataset = []
        for text, label in zip(data, labels):
            encoding = self.tokenizer(text,  # 語分割
                                      max_length=max_length,
                                      padding='max_length',
                                      truncation=True)
            encoding['labels'] = label  # 正解ラベルの設定
            encoding = {k: torch.tensor(v) for k, v
                        in encoding.items()}  # テンソルに変換
            dataset.append(encoding)  # リストに追加
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=shuffle)  # DataLoaderを戻す
    # 正解率（予測と正解が一致する割合）の計算
    def binary_accuracy(self, preds, y):
        indx = torch.argmax(preds, dim=1)
        correct = (indx == y).float()
        acc = correct.sum() / len(correct)
        return acc
    # テキスト分類での学習フェーズ（1エポック分）
    def train_tc(self, model, dl, optimizer):
        model.train()  # 訓練モードで実行
        train_loss = 0.0  # 損失関数の値を初期化
        train_acc = 0.0  # 正解率を初期化
        for batch in dl:  # バッチごとに処理
            b_in_ids = batch['input_ids'].to(self.device)
            b_in_mask = batch['attention_mask'].to(self.device)
            b_labels = batch['labels'].to(self.device)
            optimizer.zero_grad()  # 勾配を初期化
            outputs = model(b_in_ids, token_type_ids=None,
                            attention_mask=b_in_mask,
                            labels=b_labels)
            outputs.loss.backward()  # 勾配の計算
            # 勾配のクリッピング
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # パラメータの更新
            train_acc += self.binary_accuracy(  # 正解率の計算
                outputs.logits.to(self.device),
                b_labels)
            train_loss += outputs.loss.item()  # 損失関数の値を記録
        return train_loss/len(dl), train_acc/len(dl)
    # テキスト分類での検証フェーズ（1エポック分）
    def valid_tc(self, model, dataloader):
        model.eval()  # 訓練モードをオフ
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():  # 勾配を計算しない
            for batch in dataloader:  # バッチごとに処理
                b_in_ids = batch['input_ids'].to(self.device)
                b_in_mask = batch['attention_mask']. \
                    to(self.device)
                b_labels = batch['labels'].to(self.device)
                outputs = model(b_in_ids, token_type_ids=None,
                                attention_mask=b_in_mask,
                                labels=b_labels)
                val_acc += self.binary_accuracy(
                    outputs.logits.to(self.device),
                    b_labels)
                val_loss += outputs.loss.item()  # 損失関数の値を記録
        return val_loss/len(dataloader), val_acc/len(dataloader)
    # テキスト分類での学習フェーズ（実験の実行）
    def run_train_tc(self, train_dl,  # 訓練データ
                     valid_dl,  # 検証データ
                     device,  # 実行デバイス
                     no_labels=2,  # 目的変数のクラス数
                     n_epochs=10,  # 最大エポック数
                     lr=2e-5):  # 学習率
        # モデルの生成：BertForSequenceClassificationを利用
        model = BertForSequenceClassification.from_pretrained(
            self.bert_prt,
            num_labels=no_labels,  # ラベル数
        )
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr)  # オプティマイザ
        self.no_labels = no_labels  # 正解ラベルの種類数を記録
        self.device = device  # デバイスを記録
        for name, param in model.named_parameters():  # 勾配計算の設定
            param.requires_grad = True  # すべての層で勾配を計算
        loss_tr, loss_vd = [], []
        for e in range(n_epochs):  # 実行(エポックを回す)
            # 訓練フェーズ
            train_loss, acc_tr = self.train_tc(model, train_dl,
                                               optimizer)
            loss_tr.append(train_loss)
            # 検証フェーズ
            valid_loss, acc_vd = self.valid_tc(model, valid_dl)
            loss_vd.append(valid_loss)
            print(f'Epoch: {e+1:03d}/{n_epochs:03d} '  # 表示
                  f'| Tr_loss:  {train_loss:.3f} '
                  f'| Vd_Loss:  {valid_loss:.3f} '
                  f'| Tr_Acc:  {acc_tr*100:.2f}% '
                  f'| Vd_Acc:  {acc_vd*100:.2f}%')
        self.model = model  # 評価フェーズを実行するために分類器を記録しておく
        return loss_tr, loss_vd