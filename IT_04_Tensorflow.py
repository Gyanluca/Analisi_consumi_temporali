import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MaxNLocator, FuncFormatter
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

# Il codice prende una serie temporale di dati sui consumi energetici, li normalizza e li prepara per essere utilizzati in un modello LSTM. 
# Dopo l'addestramento, il modello viene utilizzato per fare previsioni future. 
# Le LSTM sono particolarmente utili per catturare le tendenze temporali e le stagionalità nei dati, fornendo previsioni basate su un'analisi sequenziale dei dati storici.

#--------------------------------------------------------------------------------------------------------------------------------------ORDINAMENTO_DEI_DATI
file_path = '01_Analisi_temporali_consumi/estat_ten00129.tsv'
data = pd.read_csv(file_path, sep='\t')

data.rename(columns={data.columns[0]: 'info'}, inplace=True)

info_split = data['info'].str.split(',', expand=True)
data['freq'] = info_split[0]
data['nrg_bal'] = info_split[1]
data['siec'] = info_split[2]
data['unit'] = info_split[3]
data['geo'] = info_split[4]

data.drop(columns=['info'], inplace=True)

data_long = data.melt(id_vars=['freq', 'nrg_bal', 'siec', 'unit', 'geo'], var_name='year', value_name='value')

data_long['year'] = data_long['year'].str.strip().astype(int)

data_long['value'] = data_long['value'].str.strip()
data_long['value'] = pd.to_numeric(data_long['value'].str.replace(':', ''), errors='coerce')

italy_data = data_long[data_long['geo'] == 'IT']

q1 = italy_data['value'].quantile(0.25)
q3 = italy_data['value'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

italy_data_cleaned = italy_data[(italy_data['value'] >= lower_bound) & (italy_data['value'] <= upper_bound)]

italy_data_cleaned = italy_data_cleaned.sort_values(by='year')

# Creazione della serie temporale
time_series = italy_data_cleaned.set_index('year')['value']

# Preparazione dei dati per la rete neurale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series.values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 12
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Creazione e addestramento del modello LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

# Previsione con il modello addestrato
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Previsione futura (10 anni)
future_predict = []
last_data = test_data[-look_back:]
for i in range(10):
    prediction = model.predict(np.reshape(last_data, (1, look_back, 1)))
    future_predict.append(prediction[0, 0])
    last_data = np.append(last_data[1:], prediction, axis=0)

future_predict = scaler.inverse_transform(np.array(future_predict).reshape(-1, 1))

# Visualizzazione
plt.figure(figsize=(14, 8))
plt.plot(time_series.index, time_series.values, label='Storico', color='blue')
plt.plot(np.arange(time_series.index[-1] + 1, time_series.index[-1] + 11), future_predict, label='Previsione Futura', color='red')
plt.axvline(x=time_series.index[-1], color='grey', linestyle='--', linewidth=1)  # Linea verticale per separare i dati storici dalla previsione futura
plt.title('Previsione dei Consumi Energetici Annuali in Italia')
plt.xlabel('Anno')
plt.ylabel('Consumo energetico (KTOE)')
plt.legend()

# Migliora la leggibilità dell'asse y
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):,}'))

# Abilita la griglia
plt.grid(True)

plt.show()

# Stampa i dati utilizzati nel grafico
print("Dati storici (ultimi 5 anni):")
print(time_series.tail())
print("\nPrevisione futura:")
print(pd.DataFrame({'Anno': np.arange(time_series.index[-1] + 1, time_series.index[-1] + 11), 'Previsione': future_predict.flatten()}))