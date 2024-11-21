'''
В этом файле напишем код для ручного тестирования модели на инференсе.
'''

import torch
import pytorch_lightning

from data import MyDataModule
from model import MyModel

class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = MyModel.load_from_checkpoint(model_path)

        '''
        Переводим модель в режим инференса.
        Например в этом режиме не работает dropout, 
        который, очевидно, нужен только для обучения.
        '''
        self.model.eval()
        '''
        Замораживаем веса модели, чтобы они не обновлялись при
        обратном распространении.
        (сам немного не понимаю, зачем это надо, ведь здесь мы нигде не
        вызываем backpropogation).
        '''
        self.model.freeze()

        self.preprocessor = MyDataModule()
        self.softmax = torch.nn.Softmax(dim=1)
        self.labels = ["unacceptable", "acceptable"]

    def predict(self, text):
        inference_sample = {"sentence": text}
        model_input_data = self.preprocessor.tokenize_data(inference_sample)
        logits = self.model(torch.tensor(model_input_data["input_ids"]),
                            torch.tensor(model_input_data["attention_mask"]))
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions
    
if __name__ == '__main__':
    input_text = input("Input some sentense: ")
    predictor = Predictor() #! set model path
    print(predictor.predict(input_text))
        