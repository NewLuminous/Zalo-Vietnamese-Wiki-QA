from pyvi import ViTokenizer

def tokenize(text):
    return ViTokenizer.tokenize(text).split()
    
if __name__ == '__main__':
    print(tokenize('Trường học là nơi đào tạo nhân tài.'))