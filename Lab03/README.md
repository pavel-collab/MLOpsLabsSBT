Для начала нам надо натренировать модель. Для этого мы используем модель и конфигурационные файлы с настройками с предыдущей лабы.
Для тренировки модели запустите скрипт.
```
python3 train.py
```
В результате в корневой дериктории лабы должен появиться каталог __checkpoints__ с сохраненной моделью __best-checkpoint.ckpt__.
Это по-сути, сохраненная натренированная модель. Далее нам надо преобразовать ее в формат onnx. Onnx это формат, который
является своего рада intermediate representation модели в том плане, что в таком формате моделями можно обмениваться между
различными фреймворками.

Для преобразования модели запустите скрипт
```
python3 convert_model_to_onnx.py
```
В результате в корневой дериктории лабы должен появиться файл __model.onnx__.

```
mkdir -p ./model_repository/onnx-bert-uncased/1
cp model.onnx ./model_repository/onnx-bert-uncased/1
```

Нужный образ для запуска тритон-сервера

```
docker pull nvcr.io/nvidia/tritonserver:23.01-py3
```

Запуск triton inference server
```
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ./model_repository:/models nvcr.io/nvidia/tritonserver:23.01-py3 tritonserver --model-repository=/models
```

В случае ошибки при запуске для отладки можно добавить в конце команды --log-verbose=1

Может возникнуть ошибка
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
Фиксится:
```
# устанавливаем toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
# перезапускаем docker deamon
sudo systemctl restart docker
```