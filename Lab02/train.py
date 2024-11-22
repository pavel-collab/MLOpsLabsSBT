'''
В этом файле мы напишем класс, который отвечает за цикл обучения модели.
Кроме того, здесь будет точка входа.
'''

import torch
import hydra
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig
from datetime import datetime

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger

from data import MyDataModule
from model import MyModel

'''
Декоратор hydra накидывается на функцию-точку входа в программу. 
'''
@hydra.main(config_path="./conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    '''
    Фиксируем random seed для torch, numpy и Python's random module
    '''
    pl.seed_everything(42)

    '''
    инициализируем объект для работы с данными,
    передаем все параметры конструктора через hydra config,
    в любой момент их можно заменить в конфиге или передать
    через comand prompt
    '''
    data = MyDataModule(model_name=cfg.model.name,
                        tokenizer_name=cfg.model.tokenizer,
                        batch_size=cfg.processing.batch_size,
                        max_length=cfg.processing.max_length)
    # выкачиваем данные для обучения и валидации
    data.prepare_data()
    # делаем препроцессинг
    data.setup()
    #TODO: вывести справочную информацию о датасете

    model = MyModel(model_name=cfg.model.name,
                    lr=cfg.optimizer.learning_rate)

    date = datetime.strftime(datetime.now(), "%d.%m.%Y-%H.%M.%S")

    '''
    Pytorch lightning поддерживает большое количество логгеров.
    Мы используем TensorBoardLogger для локального логирования и
    WandbLogger для выгрузки логов на сервис WandB.
    '''
    loggers = [
        TensorBoardLogger(
            "./.logs/my-tb-logs", 
            name=cfg.artefacts.experiment_name
        ),
        WandbLogger(
            project="mlops-logging-demo", 
            name=f"{cfg.artefacts.experiment_name}-{date}",
            log_model='all'
        )
    ]

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
                         accelerator=cfg.training.accelerator, 
                         max_epochs=cfg.training.max_epochs,
                         fast_dev_run=False, #?
                         logger=loggers,
                         callbacks=[checkpoint_callback, early_stopping_callback])
    
    try:
        trainer.fit(model, data)
    except RuntimeError as ex:
        print(f"[ERROR] model training was interrupted cz of error {ex}")
    
if __name__ == '__main__':
    # вызываем основной цикл обучения
    main()