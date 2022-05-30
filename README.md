# NEURO-fest
- папка "data" содержит исходные видео, на которых обучалась модель;
- папка "data_npy" содержит записанные в формате "npy" данные о каждом фрейме; 
- файл "new.h5" - обученная модель;
- файл "model.py" - модель, активизация которой необходима для запуска программы;
- файл "new.ipynb", разворачиваемый в jupyter notebook, показывает ход построения и тренировки модели;
- файл "bv.py" - прототип программы;

Для запуска программы необходимо активизировать виртуальное окружение:
- открыть командную строку, перейти в каталог, где будет разворачиваться окружение и ввести "python -m venv ar", где "ar" - имя среды;
- вручную скопировать файлы "bv.py", "new.h5", "model.py" в тот же каталог;
- затем ".\ar\Scripts\activate", чтобы активизировать окружение;
- установить необходимые python пакеты (python -m pip install --upgrade pip
					pip install PyQt5
					pip install opencv-python
					pip install mediapipe
					pip install tensorflow);
- запустите "bv.py"

Язык программирования: python

Используемые библиотеки:
- Tensorflow: Библиотека для машинного обучения
- Opencv-python: Библиотека компьютерного зрения
- Mediapipe: Библиотека для обнаружения на видео таких объектов, как руки
- Numpy: Библиотека для работы с Python
- PyQt: набор расширений (например, набор и стили виджетов графического интерфейса) графического фреймворка. На основе этой библиотеки была создана двухпоточная программа.
- Qt для языка программирования Python. Обеспечивает поддержку воспроизведения видео и аудио.
