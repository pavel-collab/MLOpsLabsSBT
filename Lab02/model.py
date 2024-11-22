'''
В этом файле мы опишем архитектуру модели, которую будем тренировать
'''

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import AutoModel
from sklearn.metrics import accuracy_score #TODO: добавить еще метрик для логгирования

class MyModel(pl.LightningModule):
    def __init__(self,
                 model_name="google/bert_uncased_L-2_H-128_A-2", #! повторяется 2 раза, точно сделать гиперпараметром
                 lr=1e-3):
        super(MyModel, self).__init__()

        '''
        вспомогательный метод, позволяет, в частности,
        отображать гиперпараметры при логировании.
        '''
        self.save_hyperparameters() 
        self.num_classes = 2

        self.model = AutoModel.from_pretrained(model_name)
        '''
        К уже имеющейся языковой модели добавляем дополнительно 
        1 полносвязный выходной слой. Он будет преобразовывать вектор
        скрытого состояния в n выхожных нейронов, которые соотвествуют
        меткам классов. В нашем случае n = 2.
        '''
        self.final_layer = nn.Linear(self.model.config.hidden_size, 
                                     self.num_classes)
        
    def forward(self, input_ids, attention_mask):
        # получаем выход языковой модели
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # получаем последний вектор скрытого состояния
        hid_vec = outputs.last_hidden_state[:, 0]
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
        validation_accuracy = torch.tensor(validation_accuracy)
        self.log("validation_loss", loss, prog_bar=True)
        self.log("validation_accuracy", validation_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])