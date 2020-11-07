import pandas as pd
import json
from googletrans import Translator

SOURCES = ['zaloai', 'squad', 'mailong25']

class QADataLoader:
    def __init__(self):
        self.data = pd.DataFrame()
        
    def read_csv(self, filepath, simplify=False):
        self.data = self.data.append(pd.read_csv(filepath))
        if simplify:
            return self.simplify_dataset()
        return self.data
        
    def simplify_dataset(self):
        return self.data[['question', 'text', 'label']]

class ZaloLoader(QADataLoader):
    def read_json(self, filepath, simplify=False):
        with open(filepath, encoding='utf-8') as file:
            inp = json.load(file)

        self.data = self.data.append(pd.DataFrame(inp))
        if simplify:
            return self.simplify_dataset()
        return self.data
        
class SquadLoader(QADataLoader):
    def read_json(self, filepath, simplify=False, translate_to_vn=False):
        with open(filepath, encoding='utf-8') as file:
            inp = json.load(file)
            
        if translate_to_vn:
            translator = Translator()
            
        data = []
        for case in inp['data']:
            if set(case) != {'paragraphs', 'title'}:
                raise Exception(f"Wrong format: {set(case)}")
                
            title = case['title']
            if translate_to_vn:
                title = translator.translate(title, src='en', dest='vi').text
            
            paragraphs = case['paragraphs']
            for paragraph in paragraphs:
                if set(paragraph) != {'context', 'qas'}:
                    raise Exception(f"Wrong format: {set(paragraph)}")
                    
                context = paragraph['context']
                if translate_to_vn:
                    context = translator.translate(context, src='en', dest='vi').text
                    
                qas = paragraph['qas']
                for qa in qas:
                    if set(qa) != {'answers', 'id', 'question', 'is_impossible', 'plausible_answers'} \
                    and set(qa) != {'answers', 'id', 'question', 'is_impossible'}:
                        raise Exception(f"Wrong format: {set(qa)}")
                        
                    series = pd.Series()
                    series['id'] = qa['id']
                    series['question'] = qa['question']
                    if translate_to_vn:
                        series['question'] = translator.translate(series['question'], src='en', dest='vi').text
                    
                    series['title'] = title
                    series['text'] = context
                    
                    if len(qa['answers']) > 0:
                        series['answer'] = qa['answers'][0]['text']
                    elif 'plausible_answers' in qa and len(qa['plausible_answers']) > 0:
                        series['answer'] = qa['plausible_answers'][0]['text']
                    else:
                        series['answer'] = None
                        
                    series['label'] = not qa['is_impossible']
                        
                    data.append(series)
                        
        self.data = self.data.append(data)
        if simplify:
            return self.simplify_dataset()
        return self.data

def load(src=['zaloai']):
    dataset = pd.DataFrame()
    for source in src:
        if source == 'zaloai':
            dataset = dataset.append(ZaloLoader().read_csv("data/zaloai/train.csv", simplify=True))
        elif source == 'squad':
            dataset = dataset.append(SquadLoader().read_json("data/squad/train-v2.0.json", simplify=True))
        elif source == 'mailong25':
            dataset = dataset.append(SquadLoader().read_csv("data/mailong25/squad-v2.0-mailong25.csv", simplify=True))
    
    return dataset