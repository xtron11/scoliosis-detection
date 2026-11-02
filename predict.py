import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Загружаем обученную модель
model = keras.models.load_model("scoliosis_model.h5")

# Загружаем снимок
img = image.load_img("normal.jpg", target_size=(180, 180))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

# Предсказание
pred = model.predict(x)
print("Результат:", "Сколиоз" if pred[0][0] > 0.5 else "Норма")
