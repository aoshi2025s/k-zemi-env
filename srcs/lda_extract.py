import pandas as pd

def extract_and_save_text_column(input_file_path, output_file_path):
    """
    指定されたCSVファイルから5列目のテキスト情報を抽出し、新たなCSVファイルに保存します。

    :param input_file_path: 入力CSVファイルのパス
    :param output_file_path: 出力CSVファイルのパス
    """
    # CSVファイルを読み込む
    df = pd.read_csv(input_file_path)
    
    # 5列目のテキスト情報のみを抽出
    text_column = df.iloc[:, 4]
    
    # 新たなCSVファイルに保存
    text_column.to_csv(output_file_path, index=False, header=False)

input_file_path = 'Data/Demo_classpred.csv'
output_file_path = 'Data/lda_ans.csv'
extract_and_save_text_column(input_file_path, output_file_path)
