# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:19:19 2024
ゼミ：作図用モジュール
@author: Kazuaki Kishida
"""

import matplotlib.pyplot as plt

def mds_plt(XY, labels, stress, font_size=16):
    #GSVDの結果：W=AQB^T
    #label1:列方向のラベルのリスト、label2:行方向のラベルのリスト
    font1 = "MS Gothic" #フォントの種類
    font_size = font_size #プロット中のフォントの大きさ
    font_size2 = font_size - 4 #軸目盛のフォントの大きさ
    max_v = XY.max() #最大値を求める
    min_v = XY.min()#最小値を求める
    ty = max_v / 5 #ラベルが横にはみ出る場合には「5」を小さくする
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(min_v-ty, max_v+ty), 
           ylim=(min_v-ty, max_v+ty)) #プロットの範囲を設定
    for i in range(XY.shape[0]): #プロット
        x = XY[i][0] #x座標
        y = XY[i][1] #y座標
        ax.text(x, y, labels[i], size=font_size, 
                horizontalalignment="center", 
                verticalalignment="center", fontname=font1, 
                color="g")
    x_label = "次元1" + '  *stress=' + str(stress.round(2)) #軸ラベル作成
    ax.set_xlabel(x_label, fontname=font1, size=font_size)
    y_label = "次元２" #軸ラベル作成
    ax.set_ylabel(y_label, fontname=font1, size=font_size)
    ax.tick_params(axis="x", labelsize=font_size2) #軸目盛の文字サイズ
    ax.tick_params(axis="y", labelsize=font_size2) #同上        
    plt.show() #描画実行

def ca_plt(A, B, Q, label1, label2, line=None):
    #GSVDの結果：W=AQB^T
    #label1:列方向のラベルのリスト、label2:行方向のラベルのリスト
    font1 = "MS Gothic" #フォントの種類
    font_size = 20 #プロット中のフォントの大きさ
    font_size2 = 16 #軸目盛のフォントの大きさ
    max_v = max(A.max(), B.max()) #最大値を求める
    min_v = min(A.min(), B.min()) #最小値を求める
    ty = max_v / 5 #ラベルが横にはみ出る場合には「5」を小さくする
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(min_v-ty, max_v+ty), 
           ylim=(min_v-ty, max_v+ty)) #プロットの範囲を設定
    for i in range(A.shape[0]): #行方向の変数のプロット
        x = A[i][0] #x座標
        y = A[i][1] #y座標
        ax.text(x, y, label1[i], size=font_size, 
                horizontalalignment="center", 
                verticalalignment="center", fontname=font1, 
                color="k")
        if line == 'row': #バイプロットにおける線の描画
            plt.plot([0, x], [0, y], color="red")
    for i in range(B.shape[0]): #列方向の変数のプロット
        x = B[i][0] #x座標
        y = B[i][1] #y座標
        ax.text(x, y, label2[i], size=font_size, 
                horizontalalignment="center", 
                verticalalignment="center", fontname=font1, 
                color="g")
        if line == 'column': #バイプロットにおける線の描画
            plt.plot([0, x], [0, y], color="red")
    #原点を指し示す点線を引く
    w = (0 - min_v + 1.2*ty) / (max_v - min_v + 2*ty)
    ax.axhline(y=0, xmin=0, xmax=w, color="k", 
              linestyle="dashed", lw=1)
    ax.axvline(x=0, ymin=0, ymax=w, color="k", 
               linestyle="dashed", lw=1)
    #X軸とY軸のラベルの設定
    Q2 = Q * Q #特異値の2乗
    Q2 = (Q2 / sum(Q2) * 100).round(2) #％を計算
    x_label = "成分１：" + str(Q2[0]) + "％" #軸ラベル作成
    ax.set_xlabel(x_label, fontname=font1, size=font_size)
    y_label = "成分２：" + str(Q2[1]) + "％" #軸ラベル作成
    ax.set_ylabel(y_label, fontname=font1, size=font_size)
    ax.tick_params(axis="x", labelsize=font_size2) #軸目盛の文字サイズ
    ax.tick_params(axis="y", labelsize=font_size2) #同上        
    plt.show() #描画実行