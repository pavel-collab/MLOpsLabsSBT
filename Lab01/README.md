## README

Сборка питоновского пакета из C++ исходников. Пакет представляет собой класс, содержащий функции расчета нормы L2
вектора и расчета косинусного расстояния между двумя векторами.

Сборку и установку пакета будем тестировать в докер-контейнере.

### Сборка и установка пакета в докере

Сборка докер образа с нужными зависимостями.
```
docker build -t build_python_package .
```

Запуск докер-контейнера, сборка и установка пакета
```
docker run --rm -it build_python_package
make
python3 -m build
pip3 install dist/*.whl
```

### Проверка установки

Проверьте список установленных pip пакетов (пакет cosindistance должен быть в списке).
```
pip3 list
```

Проверьте дерикторию /usr/local/lib/python3.10/dist-packages/, в нем должна быть дериктория cosindistance.
```
tree /usr/local/lib/python3.10/dist-packages/cosindistance
```
ожидаемый вывод:
```
/usr/local/lib/python3.10/dist-packages/cosindistance
|-- __init__.py
|-- __pycache__
|   `-- __init__.cpython-310.pyc
`-- python
    `-- cosin_distance.cpython-310-x86_64-linux-gnu.so
```