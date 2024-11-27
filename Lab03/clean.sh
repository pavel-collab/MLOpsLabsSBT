#!/bin/bash

# Этот скрипт используется, чтобы отчистить рабочую дерикторию от ненужных каталогов с логами
# В частности, удаляются 
# .logs -- калалог с логами TensorBoardLogger
# wandb -- католог с логами WandB
# outputs -- каталог с логами hydra (логи по разным запускам)
# __pycache__

rm -rf __pycache__
rm -rf .logs
rm -rf wandb
rm -rf outputs

# Заметим, что скрипт НЕ удаляет каталог checkpoints с чекпоинтами модели