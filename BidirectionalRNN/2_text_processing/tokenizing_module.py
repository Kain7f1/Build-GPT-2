import MeCab
import pandas as pd
import ast
import time


# 시간 측정 데코레이터
def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time} seconds.")
        return result
    return wrapper


##############################
# 기능 : text를 토큰화하여 리스트로 반환한다
def tokenize_text(text):
    mecab = MeCab.Tagger()
    parsed = mecab.parse(text)
    tokens = [line.split('\t')[0] for line in parsed.split('\n') if line and line != 'EOS']
    return tokens


###########################
# 기능 : .csv 파일에 토큰화를 적용하여 새로운 파일로 저장한다
@time_decorator
def make_tokenized_csv_file(fpath_to_read, text_column_name, fpath_to_save):
    # 1. 데이터 읽어오기
    print("[progress 1/3] 데이터 읽어오기")
    df = pd.read_csv(fpath_to_read, encoding="utf-8")

    # 2. csv 파일에 spacing 적용시키기
    print("[progress 2/3] csv 파일에 tokenizing 적용시키기")
    df['tokens'] = df[text_column_name].apply(tokenize_text)

    # 3. spacing이 적용된 데이터를 파일로 저장하기
    print("[progress 3/3] tokenizing이 적용된 데이터를 파일로 저장하기")
    df.to_csv(fpath_to_save, encoding='utf-8', index=False)  # csv 파일로 저장

    print("[done]")
