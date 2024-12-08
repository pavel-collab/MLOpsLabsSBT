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
mkdir models
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

После того, как запустили triton inference server можно делать запросы к нему.
Для этого предназначен скрипт client.py. В другом терминале запустите скрипт, который выберет из данных рандомное предложение, отправит его на сервер, там нейронка в inference режиме его обработает и пришлет ответ, который отобразится в терминале.

```
python3 client.py
```

### Работа с TensorRT

Скачиваем необходимый докер-контейнер
```
docker pull nvcr.io/nvidia/tensorrt:23.12-py3
```

Запускаем его и подключаемся (прокидываем нужную дерикторию с моделями)
```
mkdir models
cp model.onnx models
docker run -it --rm --gpus=all -v ./models:/models nvcr.io/nvidia/tensorrt:23.12-py3
```

Ура, ура, можно запускать скрипт для конвертации моделей в tensorrt. Запускаем скрипт
```
./onnx2tensorrt.sh
```