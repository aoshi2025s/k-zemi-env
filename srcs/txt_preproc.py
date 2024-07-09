# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:04:11 2022
ゼミ：テキスト処理用モジュール 
@author: Kazuaki Kishida
"""

#Wtreeのノード
class Node:
    def __init__(self, key):
        self.key = key
        self.ch = {}
        
    def getKey(self):
        return self.key
        
    def getChild(self):
        return self.ch
    
    def addChild(self, ch):
        self.ch[ch.getKey()] = ch

#先頭の文字をノードとする文字列を構成する木
class Wtree:
    
    def __init__(self, word):
        self.root = Node(word)
        self.current = self.root
        
    def add(self, words):
        w = [a for a in words if a != ''] #空列の削除
        # print(w)
        # print(w[len(w) - 1])
        if len(w) == 1: #１文字の登録
            self.root.addChild(Node('\t'))
            return
        node = self.root
        for i in range(1, len(w)):
            ch_node = node.getChild().get(w[i]) #次の階層に当該文字が存在するかどうか
            if ch_node == None:
                break
            else:
                node = ch_node #存在すれば階層を下がる
        if ch_node == None: #新規の見出し語の処理
            for k in range(i, len(w)):
                n = Node(w[k]) #ノードの新規作成
                node.addChild(n)
                node = n
            node.addChild(Node('\t'))   
            #n = Node(w[i]) #ノードの新規作成
            #n.addChild(Node('\t')) #そのノードの子ノードとして終端記号を追加
            #node.addChild(n) #１つ上の階層に追加
        else: #ここに来た場合には、重複なので警告を出す（いちおう）。動作的には問題なし
            print('Warning: dupicate entry in the dictionary: ', w)
        
    def clear(self):
        self.current = self.root
    
    #与えられた文字列を先頭とする見出しが存在するかどうかの探索
    def search(self, word):
        node = self.current.getChild().get(word) #当該ノードの子ノードの確認
        #ch2 = self.current.getChild()
        #print(word, node)
        if node == None: #子ノードに当該文字はない
            if self.current.getChild().get('\t') == None: #語が登録されていない          
                return -1
            else: #語が登録されている（つまり、見出し語の最後の文字を発見）
                return 1
        else: #子ノードに当該文字がある            
            if self.current.getChild().get('\t') != None: #そこまでで見出しになっている
                self.current = node #探索継続(次に呼び出されたときに備える)
                return 0
            else:
                self.current = node #探索継続(次に呼び出されたときに備える)
                return 99
            
#語の辞書
class Wdic:
    def __init__(self):
        self.dic = {}
        self.nentries = 0
    
    #辞書の見出しの数を返す
    def size(self):
        return self.nentries

    #見出し語の追加
    def add(self, words):
        node = self.dic.get(words[0]) #先頭1文字が登録されているかどうか
        if node == None:
            node = Wtree(words[0]) #登録されていない場合、その文字についてWtreeを生成
            self.dic[words[0]] = node #Wtreeを辞書に登録（キー：当該文字）
        node.add(words) #Wtreeに文字列を登録
        self.nentries += 1
    
    #与えられた文字（または語）の列を左から読み、辞書に登録された見出しを最長一致で抽出
    def concat(self, tokens, #文字または語のリスト
               longmtch=True, 
               alltokens=False): #alltokens=F:見出しのみ抽出、alltokens=T：その他も抽出
        tokens = [a for a in tokens if a != ''] #空列の削除
        tokens.append('\n') #番兵を置く
        pos = 0
        rslt = []
        #print(tokens)
        while pos < len(tokens):
            node = self.dic.get(tokens[pos]) #先頭1文字を探索
            phrase = []
            mtch = -1
            mtch2 = -1
            if node != None: #辞書に１文字目が登録されているので最長一致を開始
                node.clear()
                for k in range(pos+1, len(tokens)): 
                    val = node.search(tokens[k])
                    #print(val)
                    if val == -1: #tokens[pos]を先頭とする文字列は辞書中になし
                        break
                    elif val == 0: #とりあえず中間的に一致した場合
                        mtch = k #いちおう記録しておく
                        if not longmtch: #最長一致でない場合、中間的な一致を採用
                            phrase2 = []
                            for h in range(pos, mtch):
                                phrase2.append(tokens[h]) #語の構成
                            if len(phrase2) > 0:
                                rslt.append(phrase2) #追加
                            mtch2 = mtch #検証用に記録しておく
                    elif val == 1: #辞書中の見出しと一致した場合
                        mtch = k
                        break
            if mtch == -1: #tokens[pos]を先頭とする文字列が辞書中にない場合の処理
                if alltokens:
                    if tokens[pos] != '\n':
                        phrase.append(tokens[pos]) #未登録文字列も出力する場合
                pos += 1 #次の文字に移る
            else: #posからmtchの長さの文字列が辞書中に見出しとして存在
                if longmtch: #最長一致の場合
                    for h in range(pos, mtch):
                        phrase.append(tokens[h])
                    pos = mtch #次の処理は、一致した文字列の次から
                else: #辞書中の見出しをすべて識別する場合
                    if mtch != mtch2: #中間的な一致で抽出されていなければ改めて抽出
                        for h in range(pos, mtch):
                            phrase.append(tokens[h])
                    pos += 1 #見出しをすべて抽出するので、次は1文字ずらすだけ
            if len(phrase) > 0:
                rslt.append(phrase) #結果に格納     
        return rslt

import csv
class WordConcatenating:
    #MeCabなどが切りすぎた語を連結（最長一致）
    
    def __init__(self, dicfile, encoding='utf-8'):
        #引数 dicfile： 連結する語を１行にカンマ区切りで入力したCSVファイル
        #     [例]機械,学習,アルゴリズム<CR>
        self.wdic = Wdic()
        counter = 0
        with open(dicfile, 'r', encoding=encoding) as f: 
            reader = csv.reader(f)
            for row in reader:
                self.wdic.add(row)
                #print(row)
                counter += 1
        print("No. of records in the dictionary for concatenation", counter)
    
    def run(self, docs, encoding='utf-8'):
        #引数 docsの例：[['データベース', '管理', 'システム'], ['情報', '検索', '理論']]
        newdocs = []
        for doc in docs:
            rslt = self.wdic.concat(doc, longmtch=True, alltokens=True)
            #print(rslt)
            row = []
            for words in rslt:
                item = ''
                for word in words:
                    item += word
                row.append(item)
            newdocs.append(row)
        return newdocs
                    
class Indexing:
    #語のリスト（list形式）での見出し語の登録
    def set_dic(self, wordlist):
        self.wdic = Wdic()
        for word in wordlist:
            self.wdic.add(word)
        print(f"no. of records in the dictionary: {self.wdic.size()}")
    
    #抽出の実行
    def dic_match(self, texts, longmtch=1, alltokens=0, encoding="utf-8"):
        #引数 infile： 入力ファイル（第１列に、テキストを改行して列挙）
        docs = []
        removed = []
        for i in range(len(texts)):
            rslt = self.wdic.concat(texts[i], longmtch=longmtch, 
                                    alltokens=alltokens)
            if len(rslt) > 0:
                doc = ''.join(rslt[0])
                for j in range(1, len(rslt)):
                    doc += (' ' + ''.join(rslt[j]))
                docs.append(doc)
            else:
                print('Warning!: No effective word in the doc:', i)
                removed.append(i)
                docs.append('')
        return docs, removed

    def to_csv(self, docs, outfile, encoding="utf_8_sig"):
        no_records = 0
        with open(outfile, 'w', encoding="utf_8_sig") as f2: 
            writer = csv.writer(f2, lineterminator='\n')
            for doc in docs:
                writer.writerow(doc.split(' '))
                no_records += 1
        print(f"No. of records in the output file: {no_records} - {outfile}")    

class WordReplacing:
    #MeCab等の結果に対して語を置換
    
    def __init__(self, dicfile, encoding='utf-8'):
        #引数 dicfile： 辞書（CSVファイルの第1列に置換する語、それに続けて置換される語を列挙）
        #              [例]狸蕎麦,たぬきそば,たぬき蕎麦
        self.wdic = {}
        no_records = 0
        with open(dicfile, 'r', encoding=encoding) as f: 
            reader = csv.reader(f)
            for row in reader:
                key = row[0]
                for word in row:
                    self.wdic[word] = key
                no_records += 1
        print("No. of records in the dictionary", no_records)
        
    def run(self, docs, encoding='utf-8'):
        #引数 docsの例：[['データベース', '管理', 'システム'], ['情報', '検索', '理論']]
        newdocs = []
        for doc in docs:
            out = []
            for word in doc:
                newword = self.wdic.get(word)
                if newword == None:
                    out.append(word)
                else:
                    out.append(newword)
            newdocs.append(out)
        return newdocs

class Vocab: #文字列としての語をID（通し番号）に変換するクラス
    def __init__(self):
        self.wrd2idx = {} #キー：token、値：indexの辞書
        self.idx2wrd = {} #キー：index、値：tokenの辞書
    def add_token(self, token): #トークンの追加
        if token not in self.wrd2idx: #初出の場合
            self.wrd2idx[token] = len(self.wrd2idx)
            self.idx2wrd[len(self.idx2wrd)] = token
    def lookup_idx(self, token): #ID（通し番号）の探索
        return self.wrd2idx.get(token, -1)
    def lookup_word(self, idx):
        return self.idx2wrd[idx] #添え字が範囲外ならエラー
    def __len__(self): #登録されている語数を返す
        return len(self.wrd2idx)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
class Features:
    #語の特徴選択用クラス
    
    def __init__(self, corpus, min_df=1):
        #sklearn用にデータを変換
        #文書データの形式変換:['西洋','料理','方法']→['西洋 料理 方法']
        skcorpus = []
        for text in corpus: #X1は訓練データ
            s = text[0] #先頭の語
            for i in range(1, len(text)):
                if len(text[i]) > 0: #空の文字列でない場合
                    s += (' ' + text[i]) #追加
            skcorpus.append(s) #リストに追加
        self.vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b',
                                          min_df=min_df)
        self.tfidf = self.vectorizer.fit_transform(skcorpus) #テキストデータの入力
        self.sk_vocab = Vocab() #Vocabインスタンス生成
        for word in self.vectorizer.get_feature_names_out(): #語のリスト取得
            self.sk_vocab.add_token(word) #語彙に追加
    
    def select(self, n_features=500):
        mat = pd.DataFrame(self.tfidf.toarray()) #tf-idf行列の取得
        mat.loc['Total'] = mat.sum(numeric_only=True) #tf-idfの合計を計算
        mat2 = mat.sort_values(by='Total', 
                               axis=1, ascending=False) #列方向に降順ソート
        mat3 = mat2.iloc[0:len(mat2)-1, 0:n_features] #合計欄は落とす
        selected_words = [self.sk_vocab.lookup_word(index) #選択された語のリスト
                          for index in list(mat3.columns)] 
        return mat3, selected_words
    
    def get_idf(self):
        return self.vectorizer.get_feature_names_out(), self.vectorizer.idf_
 