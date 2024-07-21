import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

# In questo file rimuovo i valori anomali identificati nel file data_analysis.py per una migliore visualizzazione delle tendenze temporali,
# inoltre imposto una soglia per rimuovere i valori anomali in modo automatico utilizzando il metodo statistico IQR (Interquartile Range),
# infine visualizzo le tendenze temporali per l'Italia.
# L'obiettivo è quello di preparare i dati per l'analisi temporale e la previsione dei consumi energetici in Italia, nel file SARIMA.py

# Carica il file TSV---------------------------------------------------------------------------------------------------------------
# sep ='\t' indica che il separatore è una tabulazione e non la virgola
file_path = '01_Analisi_temporali_consumi\estat_ten00129.tsv'
data = pd.read_csv(file_path, sep='\t')                           

# Rinomina la colonna per una migliore leggibilità---------------------------------------------------------------------------------
# inplace=True indica che la modifica è fatta direttamente sul DataFrame senza crearne uno nuovo
data.rename(columns={data.columns[0]: 'info'}, inplace=True)

# Separa le informazioni nella prima colonna---------------------------------------------------------------------------------------
info_split = data['info'].str.split(',', expand=True)
data['freq'] = info_split[0]
data['nrg_bal'] = info_split[1]
data['siec'] = info_split[2]
data['unit'] = info_split[3]
data['geo'] = info_split[4]

# Rimuovi la colonna originale-----------------------------------------------------------------------------------------------------
data.drop(columns=['info'], inplace=True)

# Trasforma i dati in formato long per facilitare l'analisi------------------------------------------------------------------------
data_long = data.melt(id_vars=['freq', 'nrg_bal', 'siec', 'unit', 'geo'], var_name='year', value_name='value')

# Converti l'anno in formato numerico----------------------------------------------------------------------------------------------
data_long['year'] = data_long['year'].str.strip().astype(int)

# Rimuovi eventuali spazi bianchi e converte i valori in numerico, gestendo i valori mancanti--------------------------------------
# errors = 'coerce' indica che i valori non convertibili saranno impostati a NaN
data_long['value'] = data_long['value'].str.strip()
data_long['value'] = pd.to_numeric(data_long['value'].str.replace(':', ''), errors='coerce')

# Verifica il tipo di dati della colonna 'value'-----------------------------------------------------------------------------------DEBUG/TEST
#print(data_long['value'].dtype)

# Filtra i dati per un particolare paese (es. Italia)------------------------------------------------------------------------------
italy_data = data_long[data_long['geo'] == 'IT']

# Verifica i valori unici per identificare eventuali problemi----------------------------------------------------------------------DEBUG/TEST
#print(italy_data['value'].unique())

# Imposta una soglia per rimuovere i valori anomali basata sull'osservazione dei dati avvenuta nel file data_analysis.py-----------IMPOSTAZIONE_SOGLIA_MANUALE
#threshold = 5000                                                                         
#italy_data_cleaned = italy_data[italy_data['value'] <= threshold]

# Verifica se la soglia è stata applicata correttamente----------------------------------------------------------------------------DEBUG/TEST
#print("Valori dopo il filtraggio:", italy_data_cleaned['value'])

# Calcola IQR per determinare una soglia ottimale----------------------------------------------------------------------------------IMPOSTAZIONE_SOGLIA_AUTOMATICA_(CONSIGLIATO)
# Utilizzo il metodo statistico IQR (Interquartile Range) per identificare i valori anomali e rimuoverli
q1 = italy_data['value'].quantile(0.25)
q3 = italy_data['value'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

italy_data_cleaned = italy_data[(italy_data['value'] >= lower_bound) & (italy_data['value'] <= upper_bound)]

# Funzione di formattazione dei tick per l'asse y----------------------------------------------------------------------------------
def format_y_ticks(value, _):
    return f'{value:,}'

# Visualizza le tendenze temporali per l'Italia------------------------------------------------------------------------------------
plt.figure(figsize=(14, 8))
plt.plot(italy_data_cleaned['year'], italy_data_cleaned['value'], marker='o', label='Italia')
plt.title('Consumi energetici annuali in Italia')
plt.xlabel('Anno')
plt.ylabel('Consumo energetico (KTOE, Kilotonnes of oil equivalent)')

# Migliora la leggibilità dell'asse y-----------------------------------------------------------------------------------------------
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

# Abilita la griglia----------------------------------------------------------------------------------------------------------------
plt.grid(True)

plt.legend()
plt.show()