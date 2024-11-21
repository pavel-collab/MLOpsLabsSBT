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
    data = MyDataModule()
    model = MyDataModule()

    '''
    Иногда обучение модели может прерваться или пойти не по плану
    (например, начнет расти validation loss, что будет свидетельствовать о
    переобучении модели). ModelCheckpoint будет сохранять checkpoints 
    обучения модели по эпохам. Это позволит по окончании обучения
    вытащить checkpoint с наилучшей версией обученной модели.
    '''
    checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints",
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
                         gpus=(1 if torch.cuda.is_available() else 0),
                         max_epoch=5,
                         fast_dev_run=False, #?
                         logger=pl.loggers.TansorBoardLogger("logs", name="test", version=1), # можно вставить свой логгер
                         callbacks=[checkpoint_callback, early_stopping_callback])
    
    try:
        trainer.fit(model, data)
    except RuntimeError:
        print("[ERROR] model training was interrupted cz of some error")

if __name__ == '__main__':
    # инициализируем объект для работы с данными
    data_model = MyDataModule()
    # выкачиваем данные для обучения и валидации
    data_model.prepare_data()
    # делаем препроцессинг
    data_model.setup()
    #TODO: вывести справочную информацию о датасете

    # вызываем основной цикл обучения
    main()