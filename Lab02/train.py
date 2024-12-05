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
@hydra.main(config_path="./conf", config_name="config", version_base="1.2 ")
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
                    lr=cfg.optimizer.learning_rate,
                    dropout=cfg.model.dropout,
                    weight_decay=cfg.optimizer.weight_decay)

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
            project="mlops-vm-train-experiments", 
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
    
    trainer = pl.Trainer(default_root_dir="logs", # название каталога с логами
                         accelerator=cfg.training.accelerator, # устройство, на котором будет выполнятся обучение (может быть cpu, gpu, tpu)
                         log_every_n_steps=cfg.training.log_every_n_steps,
                         precision=cfg.training.precision, # определяет тип числа с плавающей точкой, которое будет использовано привычислениях
                         accumulate_grad_batches=cfg.training.grad_accum_steps, # за сколько шагов будут накапливаться градиенты перед обновлением весов
                         val_check_interval=cfg.training.val_check_interval, # через сколько шагов будет проводится оценка на валидационной выборке
                         deterministic=cfg.training.full_deterministic_mode, # если True, то гарантируется полностью детерменированное поведение во время обучения
                         gradient_clip_val=cfg.training.gradient_clip_val, # значение, по которому будет определятся значение градиента для обрезания, чтобы предотвратить взрыв градиентов
                         max_epochs=cfg.training.max_epochs, # сколько эпох будет пр обучении
                         logger=loggers, # список логеров
                         callbacks=[checkpoint_callback, early_stopping_callback])
    
    try:
        trainer.fit(model, data)
    except RuntimeError as ex:
        print(f"[ERROR] model training was interrupted cz of error {ex}")
    
if __name__ == '__main__':
    # вызываем основной цикл обучения
    main()
