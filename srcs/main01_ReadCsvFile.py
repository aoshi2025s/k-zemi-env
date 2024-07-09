# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:46:06 2022
ゼミ：CSVファイルの読み込み main01
@author: Kazuaki Kishida
"""
#main01: 複雑なExcelファイルの読み込み
# セル中にカンマや改行キーが含まれるCSVファイルの読み込む。
# ヘッダーは付けないこと（付いていたら、手で削除）
# カンマや改行を空白に置き換え、単純なCSV形式で出力

# カンマがあったら列が複数あるとみなされてparseエラーになる

DATA_PATH = 'Data/' #データのディレクトリを指定
encoding = 'utf-8' #文字コード　→シフトJISは'shift_jis'（非推奨）
infile = DATA_PATH + 'Demo_textdata.csv' #入力ファイル:　複雑なExcelファイル
workfile1 = DATA_PATH + 'Demo_textdata2.csv' #出力ファイル: 簡単なExcelファイル

#定義：複雑なExcelファイルを読み込むための独自関数
import pandas as pd
def read_csv(filename, #ファイル名
             col=0,  # 対象列番号の指定→1列しか処理できない
             moji=' ',  # 置換後の文字列（全角可）
             encoding='utf_8_sig'):
    df = pd.read_csv(filename, header=None) #ヘッダーなし
    newsr = [] #新規データ
    for x in df.iloc[:, col]: #指定列を1行ずつ処理
        newsr.append(moji.join(x.splitlines()).replace(',', moji))
    df = pd.DataFrame(newsr)
    return df #Pandasオブジェクトとして戻す

#実行：read_csv関数の呼び出し
df = read_csv(infile, #ファイル名
              col=0, #対象列番号の指定→列Aならば0、列Bならば1など。1つの列しか扱えない
              moji=' ', #セル中のカンマや改行キーを置換する文字列（全角可）
              encoding=encoding)
print(df.head()) #dfはPandasのDataFrameインスタンス
df.to_csv(workfile1, header=None, index=None,  #ヘッダーなし
          encoding='utf_8_sig') #BOM付きUTF-8で書き出し（推奨）
