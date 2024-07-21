import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.signal import savgol_filter
import numpy as np

# In questo file utilizzo il modello SARIMA per prevedere i consumi energetici annuali in Italia, utilizzando i dati preparati nel file data_management.py
# SARIMA (Seasonal Autoregressive Integrated Moving Average) è un modello statistico per l'analisi e la previsione di serie temporali, che tiene conto della stagionalità dei dati
# L'obiettivo è quello di prevedere i consumi energetici futuri in Italia e visualizzare le previsioni tramite un grafico.
# Inoltre vengono applicate tecniche di smoothing per migliorare la visualizzazione delle previsioni, in particolare il filtro di Savitzky-Golay,
# il quale è un filtro di smoothing che può essere utilizzato per ridurre il rumore nei dati e ottenere una rappresentazione più chiara delle tendenze.

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
#--------------------------------------------------------------------------------------------------------------------------------------PREVISIONI_CON_IL_MODELLO_SARIMA

# Preparazione dei dati per il modello SARIMA
time_series = italy_data_cleaned.set_index('year')['value']

# Rimozione dei valori nulli o zero
time_series = time_series[time_series > 0].dropna()

# Divisione del dataset
train_size = int(len(time_series) * 0.8)
train, test = time_series.iloc[:train_size], time_series.iloc[train_size:]

# Stampa dei dati di test per diagnosticare il problema
print("Dati di test:")
print(test)

# Verifica se i dati di test contengono valori nulli
if test.isnull().sum() > 0:
    print("I dati di test contengono valori nulli.")
else:
    print("I dati di test sono validi.")

# Utilizziamo il modello SARIMA con parametri aggiustati
try:
    model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # Valutazione del modello sui dati di test (Previsione)
    forecast = model_fit.get_forecast(steps=len(test))
    forecast_series = forecast.predicted_mean
    
    # Calcola il fattore di scala per allineare la previsione sui dati di test con i dati storici
    scale_factor = train.iloc[-1] / forecast_series.iloc[0]
    forecast_series *= scale_factor * 0.1  # Moltiplica per 0.1 (range 0-1), in tal modo la previsione è allineata ai dati storici

    # Applica il filtro di Savitzky-Golay per lo smoothing della previsione sui dati di test
    smoothed_forecast_series = savgol_filter(forecast_series, window_length=3, polyorder=1)
except Exception as e:
    print(f"Errore durante la costruzione o la previsione del modello SARIMA: {e}")
    forecast_series = pd.Series([np.nan] * len(test), index=test.index)
    smoothed_forecast_series = pd.Series([np.nan] * len(test), index=test.index)

# Previsione futura (10 anni)
try:
    future_steps = 10
    future_forecast = model_fit.get_forecast(steps=future_steps)
    future_forecast_series = future_forecast.predicted_mean

    # Applica il fattore di scala alla previsione futura
    future_forecast_series *= scale_factor * 0.1

    future_years = np.arange(time_series.index[-1] + 1, time_series.index[-1] + future_steps + 1)
    
    # Applica il filtro di Savitzky-Golay per lo smoothing della previsione futura
    smoothed_future_forecast_series = savgol_filter(future_forecast_series, window_length=3, polyorder=1)
except Exception as e:
    print(f"Errore durante la previsione futura del modello SARIMA: {e}")
    future_forecast_series = pd.Series([np.nan] * future_steps)
    smoothed_future_forecast_series = pd.Series([np.nan] * future_steps)
    future_years = np.arange(time_series.index[-1] + 1, time_series.index[-1] + future_steps + 1)

# Funzione di formattazione dei tick per l'asse y
def format_y_ticks(value, _):
    return f'{int(value):,}'

# Visualizzazione dei risultati
plt.figure(figsize=(14, 8))
plt.plot(time_series, label='Storico', color='blue')
plt.plot(test.index, smoothed_forecast_series, label='Previsione', color='orange')
plt.plot(future_years, smoothed_future_forecast_series, label='Previsione Futura', color='red')
plt.axvline(x=time_series.index[-1], color='grey', linestyle='--', linewidth=1)  # Linea verticale per separare i dati storici dalle previsioni future
plt.title('Previsione dei Consumi Energetici Annuali in Italia')
plt.xlabel('Anno')
plt.ylabel('Consumo energetico (KTOE)')
plt.legend()

# Migliora la leggibilità dell'asse y
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

# Abilita la griglia
plt.grid(True)

plt.show()

# Verifica i dati utilizzati per il grafico
print("Dati storici (ultimi 5 anni):")
print(time_series.tail())
print("\nPrevisione sui dati di test:")
print(smoothed_forecast_series)
print("\nPrevisione futura:")
print(smoothed_future_forecast_series)