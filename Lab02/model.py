'''
В этом файле мы опишем архитектуру модели, которую будем тренировать
'''

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class MyModel(pl.LightningModule):
    '''
    Заметим, что здесь задаются аргументы конструктора по умолчанию.
    При использовании класса в тренеровочном цикле можно будет поменять
    эти гиперпараметры вручную или с помощью Hydra config.
    '''
    def __init__(self,
                 model_name="google/bert_uncased_L-2_H-128_A-2",
                 dropout=0.2,
                 weight_decay=0.02,
                 lr=1e-3):
        super(MyModel, self).__init__()
        self.save_hyperparameters()

        '''
        вспомогательный метод, позволяет, в частности,
        отображать гиперпараметры при логировании.
        '''
        self.save_hyperparameters() 
        self.num_classes = 2
        self.dropout = dropout

        self.model = AutoModel.from_pretrained(model_name)
        '''
        К уже имеющейся языковой модели добавляем дополнительно 
        1 полносвязный выходной слой. Он будет преобразовывать вектор
        скрытого состояния в n выхожных нейронов, которые соотвествуют
        меткам классов. В нашем случае n = 2.
        '''
        self.final_layer = nn.Linear(self.model.config.hidden_size, 
                                     self.num_classes)
        # сделаем еще один слой droput
        self.dropout_layer = torch.nn.Dropout(p=self.dropout)    
        
    def forward(self, input_ids, attention_mask):
        # получаем выход языковой модели
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # получаем последний вектор скрытого состояния
        hid_vec = outputs.last_hidden_state[:, 0]

        hid_vec = self.dropout_layer(hid_vec)

        # получаем логиты с помощью полносвязного слоя
        logits = self.final_layer(hid_vec)
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        _, preds = torch.max(logits, dim=1)
        '''
        Заметим, что вычисление нейронной сети может проводится как на cpu, 
        так и на gpu, но для расчета метрики валидации необходимо перенести
        все компоненты расчета на один девайс.
        '''
        validation_accuracy = accuracy_score(preds.cpu(), batch["label"].cpu())
        validation_precision = precision_score(preds.cpu(), batch["label"].cpu())
        validation_recall = recall_score(preds.cpu(), batch["label"].cpu())
        validation_f1_micro = f1_score(preds.cpu(), batch["label"].cpu(), average='micro')

        validation_accuracy = torch.tensor(validation_accuracy)
        validation_precision = torch.tensor(validation_precision)
        validation_recall = torch.tensor(validation_recall)
        validation_f1_micro = torch.tensor(validation_f1_micro)

        self.log("validation_loss", loss, prog_bar=True)
        self.log("validation_accuracy", validation_accuracy, prog_bar=True)
        self.log("validation_precision", validation_precision, prog_bar=True)
        self.log("validation_recall", validation_recall, prog_bar=True)
        self.log("validation_f1_micro", validation_f1_micro, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])