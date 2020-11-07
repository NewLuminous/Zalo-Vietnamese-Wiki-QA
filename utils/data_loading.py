import pandas as pd
import json

SOURCES = ['zaloai', 'mailong25']

class QADataLoader:
    def __init__(self):
        self.data = pd.DataFrame()
        
    def simplify_dataset(self):
        return self.data[['question', 'answer', 'label']]

class ZaloLoader(QADataLoader):
    def read_json(self, filepath, simplify=False):
        with open(filepath, encoding='utf-8') as file:
            inp = json.load(file)

        self.data = self.data.append(pd.DataFrame(inp).rename(columns={'text': 'answer'}))
        if simplify:
            return self.simplify_dataset()
        return self.data
        
class SquadLoader(QADataLoader):
    def read_json(self, filepath, simplify=False):
        with open(filepath, encoding='utf-8') as file:
            inp = json.load(file)
            
        data = []
        for case in inp['data']:
            if list(case) != ['paragraphs', 'title']:
                raise Exception(f"Wrong format: {list(case)}")
                
            title = case['title']
            paragraphs = case['paragraphs']
            for paragraph in paragraphs:
                if list(paragraph) != ['context', 'qas']:
                    raise Exception(f"Wrong format: {list(paragraph)}")
                    
                id = None
                answer = None
                is_impossible = True
                context = paragraph['context']
                qas = paragraph['qas']
                for qa in qas:
                    if list(qa) != ['answers', 'id', 'question', 'is_impossible', 'plausible_answers']:                        raise Exception(f"Wrong format: {list(qa)}")
                    
                    if qa['question'] != title:
                        raise Exception(f"Inconsistent question")
                        
                    id = qa['id']
                    if len(qa['answers']) > 0:
                        answer = qa['answers'][0]['text']
                    if not qa['is_impossible']:
                        is_impossible = False
                        break
                series = pd.Series()
                series['id'] = id
                series['question'] = title
                series['title'] = answer
                series['answer'] = context
                series['label'] = not is_impossible
                data.append(series)
        self.data = self.data.append(data)
        if simplify:
            return self.simplify_dataset()
        return self.data

def load(src=['zaloai']):
    dataset = pd.DataFrame()
    for source in src:
        if source == 'zaloai':
            dataset = dataset.append(ZaloLoader().read_json("data/zaloai/train.json", simplify=True))
        elif source == 'mailong25':
            dataset = dataset.append(SquadLoader().read_json("data/mailong25/squad-v2.0-mailong25.json", simplify=True))
    
    return dataset