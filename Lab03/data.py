'''
В этом файле описывается класс для загрузки и предобработки данных, 
которые будут использованы для обучения и валидации модели.
'''

import torch
import datasets
import pytorch_lightning as pl
import random

from datasets import load_dataset
from transformers import AutoTokenizer

class MyDataModule(pl.LightningDataModule):
    '''
    Заметим, что здесь задаются аргументы конструктора по умолчанию.
    При использовании класса в тренеровочном цикле можно будет поменять
    эти гиперпараметры вручную или с помощью Hydra config.
    '''
    def __init__(self, 
                 model_name="google/bert_uncased_L-2_H-128_A-2",
                 tokenizer_name="google/bert_uncased_L-2_H-128_A-2",
                 batch_size=32,
                 max_length=128):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def prepare_data(self):
        dataset = load_dataset("glue", "cola") #? dataset name could be hydra parametr, also it could be class constructor parametr
        self.train_dataset = dataset["train"]
        self.validation_dataset = dataset["validation"]

        features = dataset['train'].features
        if 'label' in features:
            label_names = features['label'].names
            label_mapping = {i: name for i, name in enumerate(label_names)}
            self.idx2label = label_mapping
        else:
            self.idx2label = None
    
    '''
    Модель принимает на вход батчи последовательностей токенов.
    Прежде, чем подавать данные на вход модели, их надо предобработать.
    Важноотметить, что модель ожидает на вход последовательности одинаковой 
    фиксированной длинны, поэтому некоторые входные данные нужно обрезать или
    наоборот дополнять паддингом.
    '''
    def tokenize_data(self, example):
        return self.tokenizer(example["sentence"],
                              truncation=True, # отвечает за обрезание слишком длинного предложения
                              padding="max_length", # отвечает за дополнение слишком короткого предложения
                              max_length=self.max_length)
    
    def setup(self, stage=None):
        if stage == "fit" or stage == None:
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
            self.validation_dataset = self.validation_dataset.map(self.tokenize_data, batched=True)
            self.validation_dataset.set_format(type="torch",
                                               columns=["input_ids", "attention_mask", "label"])
    
    '''
    Возвращает специальный объект с методом __getitem__,
    который будет выдавать батчи данных один за другим.
    '''
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True)
    
    # очень важно, чтобы метод имел именно такое название =)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False)
    
    def sample_random_item(self):
        dataloader = self.val_dataloader()
        random_batch_num = len(dataloader)
        
        while random_batch_num != 0:
            random_batch_num -= 1
            random_batch = next(iter(dataloader))

        random_sentence_idx = random.randint(0, self.batch_size)
        random_item = {'input_ids': random_batch['input_ids'][random_sentence_idx], 
                       'attention_mask': random_batch['attention_mask'][random_sentence_idx], 
                       'label': int(random_batch['label'][random_sentence_idx])}

        return random_item
