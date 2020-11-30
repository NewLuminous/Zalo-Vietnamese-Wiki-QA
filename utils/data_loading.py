import pandas as pd
import json
from utils.translating import GoogleTranslator

SOURCES = ['zaloai', 'squad', 'mailong25', 'facebook']

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
    def __init__(self):
        super().__init__()
        self.translator = GoogleTranslator(src='en', dest='vi')

    def read_json(self, filepath, version=2, simplify=False, translate_to_vn=False):
        with open(filepath, encoding='utf-8') as file:
            inp = json.load(file)
            
        data = []
        for case in inp['data']:
            if set(case) != {'paragraphs', 'title'}:
                raise Exception(f"Wrong format: {set(case)}")
                
            title = case['title']
            if translate_to_vn:
                title = self.translator.translate(title)
            
            paragraphs = case['paragraphs']
            for paragraph in paragraphs:
                if set(paragraph) != {'context', 'qas'}:
                    raise Exception(f"Wrong format: {set(paragraph)}")
                    
                context = paragraph['context']
                if translate_to_vn:
                    context = self.translator.translate(context, patience_lim=100)
                    
                qas = paragraph['qas']
                for qa in qas:
                    if version == 1:
                        if set(qa) != {'answers', 'id', 'question'}:
                            raise Exception(f"Wrong format: {set(qa)}")
                    else:
                        if set(qa) != {'answers', 'id', 'question', 'is_impossible', 'plausible_answers'} \
                        and set(qa) != {'answers', 'id', 'question', 'is_impossible'}:
                            raise Exception(f"Wrong format: {set(qa)}")
                        
                    series = pd.Series()
                    series['id'] = qa['id']
                    series['question'] = qa['question']
                    if translate_to_vn:
                        series['question'] = self.translator.translate(series['question'], patience_lim=100)
                    
                    series['title'] = title
                    series['text'] = context
                    
                    if len(qa['answers']) > 0:
                        series['answer'] = qa['answers'][0]['text']
                    elif 'plausible_answers' in qa and len(qa['plausible_answers']) > 0:
                        series['answer'] = qa['plausible_answers'][0]['text']
                    else:
                        series['answer'] = None
                        
                    series['label'] = True if version == 1 else not qa['is_impossible']
                        
                    data.append(series)
                        
        self.data = self.data.append(data)
        if simplify:
            return self.simplify_dataset()
        return self.data
        
def load_train(src=['zaloai']):
    dataset = pd.DataFrame()
    for source in src:
        if source == 'zaloai':
            input = ZaloLoader().read_csv("data/zaloai/train.csv", simplify=True)
            dataset = dataset.append(input[:int(len(input)*0.9)])
        elif source == 'squad':
            dataset = dataset.append(SquadLoader().read_csv("data/squad/train-v2.0_part_1.csv", simplify=True))
            dataset = dataset.append(SquadLoader().read_csv("data/squad/train-v2.0_part_2.csv", simplify=True))
        elif source == 'mailong25':
            input = SquadLoader().read_csv("data/mailong25/squad-v2.0-mailong25.csv", simplify=True)
            dataset = dataset.append(input[:int(len(input)*0.9)])
        elif source == 'facebook':
            dataset = dataset.append(SquadLoader().read_csv("data/facebook/test-context-vi-question-vi_fb.csv", simplify=True))
        else:
            raise Exception(f"Source '{source}' not found. Try 'utils.data_loading.SOURCES' for available sources.")
    
    return dataset
    
def load_test(src=['zaloai']):
    dataset = pd.DataFrame()
    for source in src:
        if source == 'zaloai':
            input = ZaloLoader().read_csv("data/zaloai/train.csv", simplify=True)
            dataset = dataset.append(input[int(len(input)*0.9):])
        elif source == 'squad':
            dataset = dataset.append(SquadLoader().read_csv("data/squad/dev-v2.0.csv", simplify=True))
        elif source == 'mailong25':
            input = SquadLoader().read_csv("data/mailong25/squad-v2.0-mailong25.csv", simplify=True)
            dataset = dataset.append(input[int(len(input)*0.9):])
        elif source == 'facebook':
            dataset = dataset.append(SquadLoader().read_csv("data/facebook/dev-context-vi-question-vi_fb.csv", simplify=True))
        else:
            raise Exception(f"Source '{source}' not found. Try 'utils.data_loading.SOURCES' for available sources.")
    
    return dataset

def load(src=['zaloai']):
    dataset = pd.DataFrame()
    dataset = dataset.append(load_train(src))
    dataset = dataset.append(load_test(src))
    return dataset