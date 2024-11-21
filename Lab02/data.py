'''
В этом файле описывается класс для загрузки и предобработки данных, 
которые будут использованы для обучения и валидации модели.
'''

import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTekenizer

class MyDataModule(pl.LightningDataModule):
    def __init__(self, 
                 model_name="google-bert/bert-base-uncased", #! could be hydra parametr
                 batch_size=32):
        
        self.batch_size = batch_size
        self.tokenizer = AutoTekenizer.from_pretrained(model_name)

    def prepare_data(self):
        dataset = load_dataset("glue", "cola") #? dataset name could be hydra parametr, also it could be class constructor parametr
        self.train_dataset = dataset["train"]
        self.validation_dataset = dataset["validation"]
    
    '''
    Модель принимает на вход батчи последовательностей токенов.
    Прежде, чем подавать данные на вход модели, их надо предобработать.
    Важноотметить, что модель ожидает на вход последовательности одинаковой 
    фиксированной длинны, поэтому некоторые входные данные нужно обрезать или
    наоборот дополнять паддингом.
    '''
    def tokenize_data(self, example):
        return self.tokenizer(example["sentence"],
                              trunkation=True, # отвечает за обрезание слишком длинного предложения
                              padding="max_length", # отвечает за дополнение слишком короткого предложения
                              max_length=512)
    
    def setup(self, stage=None):
        if stage == "fir" or stage == None:
            '''
            Выполняем предобработку данных для обучения: получаем эмбеддинги
            и делим данные по батчам.
            '''
            self.train_dataset = self.train_dataset.map(self.tokenize_data, batched=True)
            '''
            Языковые модели с hugging face ожидают на вход данные определенного формата, 
            а именно -- torch-тензоры с батчами равной длинны.
            
            input_ids -- коды токенов в эмбеддинге
            attention_mask -- вектор меток, которые указывают, является ли код эмбеддинга паддингом
            label -- целевая метка
            '''
            self.train_dataset.set_format(type="torch",
                                          columns=["input_ids", "attention_mask", "label"])
            
            '''
            Заметим, что нам не обязательно делить на батчи валидационную выборку. На ней не будет
            происходить стохастический градиентный спуск, она нужна только чтобы посчитать промежуточную
            функцию потерь.
            '''
            self.validation_dataset = self.validation_dataset.map(self.tokenize_data, batched=False)
            self.validation_dataset.set_format(type="torch",
                                               columns=["input_ids", "attention_mask", "label"])
    
    '''
    Возвращает специальный объект с методом __getitem__,
    который будет выдавать батчи данных один за другим.
    '''
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           shuffle=True)
    
    def validation_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_data,
                                           batch_size=self.batch_size,
                                           shuffle=False)