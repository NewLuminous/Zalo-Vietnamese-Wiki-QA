import pandas as pd
import json

class DataLoader:
    def __init__(self):
        self.data = pd.DataFrame()

class ZaloLoader(DataLoader):
    def read_json(self, filepath, simplify=False):
        with open(filepath, encoding='utf-8') as file:
            inp = json.load(file)

        self.data = self.data.append(pd.DataFrame(inp))
        if simplify:
            return self.simplify_dataset()
        return self.data

    def simplify_dataset(self):
        renamed_data = self.data.rename(columns={'text': 'answer'})
        return renamed_data[['question', 'answer', 'label']]

def load(src=['zaloai']):
    dataset = pd.DataFrame()
    for source in src:
        if source == 'zaloai':
            dataset = dataset.append(ZaloLoader().read_json("data/zaloai/train.json", simplify=True))
    
    return dataset