import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

df = pd.read_csv("tic-tac-toe.csv")
# print(df.head())
# print(df.describe())
# print(df.info)

x = df.iloc[:, :9].to_numpy()
y = df.iloc[:, -1].to_numpy()

for rowIndex in range(len(x)):
    row = x[rowIndex]
    for cellIndex in range(len(row)):
        cellString = row[cellIndex]

        if cellString == 'x':
            x[rowIndex, cellIndex] = 0.0

        elif cellString == 'o' or cellString == 'negative':
            x[rowIndex, cellIndex] = 1.0

        else:
            x[rowIndex, cellIndex] = 2.0


for rowIndex in range(len(y)):
    if y[rowIndex] == 'positive':
        y[rowIndex] = 1.0

    if y[rowIndex] == 'negative':
        y[rowIndex] = 0.0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=21)


dataset_shape = [np.shape(x_train)[1]]
print(dataset_shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=dataset_shape),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

print(model.summary())

model.compile(optimizer="adam", loss="mae", metrics=['accuracy'])

losses = model.fit(
    np.asarray(x_train).astype('float32'),
    np.asarray(y_train).astype('float32'),
    validation_data=(
        np.asarray(x_test).astype('float32'),
        np.asarray(y_test).astype('float32'),
    ),
    batch_size=256,
    epochs=256
)

print("Hasil NN:")
print(model.predict(np.asarray(x_test).astype('float32')[10:20, :]))
print("")
print("Jawaban yang benar:")
print(np.asarray(y_test).astype('float32')[10:20])
