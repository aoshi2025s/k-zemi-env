tp_depth = 3 #トピックの階層数
depth = 1 #クラスタを構成する際のトピック階層の深さ：0～tp_depth-1
n_ites = 5000 #学習時の反復回数

#モデルの初期化とデータの入力
hlda = tp.HLDAModel(depth=tp_depth, seed=123)
for text in corpus: #corpusの例：[['東京','日本'],['大阪']]
    hlda.add_doc(text) #1文書ずつ追加

#学習段階
for i in range(0, n_ites, 10): #10回刻みで対数尤度を表示
    hlda.train(iter=10, workers=1)
    print('Iteration: {}\tLog-likelihood: {}'.
          format(i, hlda.ll_per_word))
hlda.summary(topic_word_top_n=5) #結果のコンソール表示

#文書が割り当てられているトピックを確認
clusters = {} #クラスタを記録
tp2cl = {} #トピック番号を文書クラスタ番号に変換するための辞書
cl2tp = {} #逆引き用
pred = [] #予測クラスタ番号のリスト
docno = 0 #文書番号（0からの通し番号）
for doc, text in zip(hlda.docs, texts2): #文書ごとに処理
    if not doc.path[depth] in tp2cl: #当該階層で文書が属するトピック確認
        tp2cl[doc.path[depth]] = len(tp2cl) #新規トピックに通し番号付与
        cl2tp[tp2cl[doc.path[depth]]] = doc.path[depth] #逆引き
    cl = tp2cl[doc.path[depth]] #文書クラスタ番号を取得
    if not cl in clusters: #クラスタ記録用リストの初期化（初出の場合）
        clusters[cl] = [] #初期化
    clusters[cl].append([docno, text]) #クラスタ記録実行
    pred.append(cl) #文書ごとの潜在トピック番号（予測）
    docno += 1 #文書の通し番号を更新

#最終的な文書クラスタ数を算出
print(f'\n文書クラスタ数：{len(tp2cl)}, トピック番号={tp2cl}')

n_docs = [hlda.num_docs_of_topic(k) for k in tp2cl]
print(f'各クラスタの文書数： {n_docs}')

#ファイルへの結果の書き出し
with open(outfile, 'w', encoding='utf_8_sig') as f: #ファイル出力
    f.write('クラスタ番号,文書番号,テキスト\n')
    for k in clusters: #クラスタごとに処理
        for dat in clusters[k]:
            f.write(str(k) + ',' + str(dat[0]) + ',' +
                    dat[1][:n_mojis] + '\n') #元のテキストを出力
    f.write('クラスタ番号,語,確率\n')
    for i, k in enumerate(tp2cl): #クラスタごとに語の確率を出力
        words = hlda.get_topic_words(k, top_n=20)
        for wd in words:
            f.write(str(i) + ',' + wd[0] + ',' +
                    str(wd[1]) + '\n')