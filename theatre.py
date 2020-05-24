
# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
# import tensorflow as tf
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from pylab import rcParams
rcParams['figure.figsize'] = 22, 10
sns.set(color_codes=True)


# # Загрузка данных В качестве данных мы берем показатели датчиков двигателей, которые вращают шкив, измеряющий угол
# от 0 до 3600. В последствии, этот угол, в зависимости от кол-ва оборотов, преобразуется в позицию (в см) декорации.


Position_df = pd.read_csv('Position.csv', sep=',', encoding='UTF-8')
Position_df['Time'] = pd.to_datetime(Position_df['time'])


def print_position(df, number):
    i = 0
    while i < number:
        name = 'theatre.D' + str(i) + '_Position_AS'
        item = df[df.name == name]
        if (len(item) > 100):
            plt.plot(item.Time, item.value, label=name)
            plt.legend()
        i = i + 1


COUNT = 15
print_position(Position_df, COUNT)

# На графике хорошо видно, что датчик на двигателе №14 имеет анамальное значение. Именно его мы и возьмем для
# обучения нашей модели

Position_D14 = Position_df[Position_df.name == 'theatre.D14_Position_AS1']
Position_D14 = pd.DataFrame(Position_D14, columns=['Time', 'value'])
Position_D14 = Position_D14.set_index('Time')

# # Подготовка данных Определяем наборы данных для обучения и тестирования нашей модели. Разделяем наши данные на две
# части. Первая часть (train), в котором мы тренируемся в набора данных, которая представляет нормальные условия
# работы. Вторая часть (test), которая содержит аномалии.


train_size = int(len(Position_D14) * 0.82)
test_size = len(Position_D14) - train_size
train, test = Position_D14.iloc[2:train_size], Position_D14.iloc[train_size:len(Position_D14)]
print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

# # Нормализация и стандартизация данных

# normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)


# # Преобразование данных для LTSM Затем мы преобразуем наши данные в формат, подходящий для ввода в сеть LSTM.
# Ячейки LSTM ожидают трехмерный тензор формы [выборки данных, временные шаги, особенности]


# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)


# # Построение нейронной сети LSTM Для нашей модели обнаружения аномалий мы будем использовать архитектуру
# нейроконтроллеров с автоматическим кодированием. Архитектура автоэнкодера по существу изучает функцию
# «идентичности». Он примет входные данные, создаст сжатое представление основных движущих характеристик этих данных,
# а затем научится восстанавливать их снова. Основанием для использования этой архитектуры для обнаружения аномалий
# является то, что мы обучаем модель «нормальным» данным и определяем полученную ошибку реконструкции. Затем,
# когда модель встретит данные, которые находятся за пределами нормы, и попытается восстановить их, мы увидим
# увеличение ошибки восстановления, поскольку модель никогда не обучалась для точного воссоздания элементов вне нормы.


# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model


# create the autoencoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()


# fit the model to the data
nb_epochs = 100
batch_size = 10
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size, validation_split=0.05).history


# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()
fig.savefig('Model less.png')

# График тренировочных потерь для оценки производительности модели

# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index


xx = X_train.reshape(X_train.shape[0], X_train.shape[2])
xx = pd.DataFrame(xx, columns=train.columns)
xx.index = train.index


plt.plot(X_pred, color='blue', label='pred')
plt.plot(xx, color='red', label='real')
plt.legend()

# # Распределение убытков Составляя график распределения вычисленных потерь в обучающем наборе, мы можем определить
# подходящее пороговое значение для выявления аномалии. При этом можно убедиться, что этот порог установлен выше
# «уровня шума», чтобы ложные срабатывания не срабатывали.


scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred - Xtrain), axis=1)
plt.figure(figsize=(16, 9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins=100, kde=True, color='blue')
plt.savefig('Loss Distribution.png')

# Исходя из приведенного выше распределения потерь, возьмем пороговое значение 0,01 для обозначения аномалии. Затем
# мы рассчитываем потери на реконструкцию в обучающем и тестовом наборах, чтобы определить, когда показания датчика
# пересекают порог аномалии.

# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
xtest = pd.DataFrame(xtest, columns=test.columns)
xtest.index = test.index

plt.plot(X_pred, color='blue')
plt.plot(xtest, color='red')
plt.savefig('Prediction.png')

# # Аномалии
scored_test = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored_test['Loss_mae'] = np.mean(np.abs(X_pred - Xtest), axis=1)
scored_test['Threshold'] = 0.01
scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']
# scored['value'] = test.value

# calculate the same metrics for the training set 
# and merge all data in a single dataframe for plotting
X_pred_train = model.predict(X_train)
X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns)
X_pred_train.index = train.index

scored_train = pd.DataFrame(index=train.index)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train - Xtrain), axis=1)
scored_train['Threshold'] = 0.01
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
# scored_train['value'] = train.value
scored = pd.concat([scored_train, scored_test])

# Чтобы визуализировать результаты с течением времени. Красная линия указывает на наше пороговое значение 0,01.

# plot bearing failure time plot
scored.plot(logy=True, figsize=(16, 9), color=['blue', 'red'])
plt.savefig('Threshold.png')


test_score_df = scored_test
test_score_df['value'] = test.value

anomalies = test_score_df[test_score_df.Anomaly == True]

plt.plot(
    test.index,
    test.value,
    label='value'
)

sns.scatterplot(
    anomalies.index,
    anomalies.value,
    color=sns.color_palette()[3],
    s=52,
    label='anomaly'
)
plt.xticks(rotation=25)
plt.legend()
plt.savefig('Anomalies.png')

# по графику видно, какие значения двигателя приводят к анамалии, а значит в дальyейшем мы можем заранее определять
# по первым показателям, когда у нас возникают анамалии

model.save('model.h5')

