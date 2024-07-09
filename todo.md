# 作成したSVMを、任意のデータに対して再利用できるようにする

- user_input_data.csv
    - これはユーザが指定した企業の経営状況の部分の文章
- user_input_data2.csv
    - read_csv.pyにかけた後のファイル
- user_input_data3b.csv
    - IndexingByJanome.pyにかけた後のファイル

本当はそれぞれのpythonファイルをモジュール化して外から呼べるようにしたい
そのためには機能ごとに関数化する必要があるけれど。。。

一旦は同じ内容の名前と入力ファイルだけ違うプログラムを複製

分類機作成用のデータ=Demo_textdata, Demo_answer

分類したい本当のデータ=user_input_data

これらをindexingByJanomeしてから、SVMに一緒にかける？

ここのSVMのコードがわからないのと、

出力は２つとも一緒になる？

Demo_classpred.csvのどれを抽出すれば良い？？
予測と正解とテキストの列がある。
正解が1になってるやつ？？

- 語形マッチングで一気に抽出した方が良いかも？先行研究と同じように。