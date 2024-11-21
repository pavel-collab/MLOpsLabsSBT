'''
В этом файле мы напишем класс, который отвечает за цикл обучения модели.
Кроме того, здесь будет точка входа.
'''

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import MyDataModule
from model import MyModel

def main():
     # инициализируем объект для работы с данными
    data = MyDataModule()
    # выкачиваем данные для обучения и валидации
    data.prepare_data()
    # делаем препроцессинг
    data.setup()
    #TODO: вывести справочную информацию о датасете

    model = MyModel()

    '''
    Иногда обучение модели может прерваться или пойти не по плану
    (например, начнет расти validation loss, что будет свидетельствовать о
    переобучении модели). ModelCheckpoint будет сохранять checkpoints 
    обучения модели по эпохам. Это позволит по окончании обучения
    вытащить checkpoint с наилучшей версией обученной модели.
    '''
    checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints",
                                          save_top_k=2,
                                          monitor="validation_loss",
                                          mode="min")
    
    '''
    EarlyStopping позволит остановить обучение модели в случае, если
    начнется переобучение.
    '''
    early_stopping_callback = EarlyStopping(monitor="validation_loss",
                                            patience=3, # сколько чекпоинтов ждать, если валидационная ошибка начнет расти
                                            verbose=True, # сохранять отладочную информацию
                                            mode="min")
    
    trainer = pl.Trainer(default_root_dir="logs", #?
                        #  accelerator="auto",
                        accelerator="cpu",
                         max_epochs=5,
                         fast_dev_run=False, #?
                         logger=pl.loggers.TensorBoardLogger("logs", name="cola", version=1), # можно вставить свой логгер
                         callbacks=[checkpoint_callback, early_stopping_callback])
    
    try:
        trainer.fit(model, data)
    except RuntimeError as ex:
        print(f"[ERROR] model training was interrupted cz of error {ex}")

if __name__ == '__main__':
    # вызываем основной цикл обучения
    main()