import pandas as pd
import string

def lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize(text):
    digit_dict = {
        '0': u' không ',
				'1': u' một ',
				'2': u' hai ',
				'3': u' ba ',
				'4': u' bốn ',
				'5': u' năm ',
				'6': u' sáu ',
				'7': u' bảy ',
				'8': u' tám ',
				'9': u' chín '
        }

    new_d = {ord(k): v for k, v in digit_dict.items()}
    return text.translate(new_d)

def remove_extra_whitespace(text):
    return " ".join(text.split())

def preprocess(text):
    steps = [lowercase, remove_punctuation, normalize, remove_extra_whitespace]
    for step in steps:
        text = step(text)
    return text

def preprocess_qa(_df):
    df = _df.copy()
    df['question'] = df['question'].apply(preprocess)
    df['text'] = df['text'].apply(preprocess)
    return df

if __name__ == "__main__":
	sentence = u"[ [ SI ] ] đơn vị áp suất [ [ pascal ( đơn vị ) | pascal ] ] ( Pa ) , bằng một [ [ niutơn ( đơn vị ) | niutơn ] ] mỗi [ [ mét vuông ] ] ( N · m hoặc kg · m · s ) . Tên này đặc biệt cho các đơn vị đã được bổ sung vào năm 1971 , trước đó , áp lực trong SI được thể hiện trong các đơn vị như N / m2 ."
	print(preprocess(sentence))
