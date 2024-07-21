import pandas as pd
import matplotlib.pyplot as plt

# In questo file analizzo i dati per identificare eventuali outliers e visualizzare la distribuzione dei dati tramite un grafico boxplot
# L'obiettivo è quello di rimuovere i valori anomali per una migliore visualizzazione delle tendenze temporali, nel file data_management.py

# Il grafico boxplot mostra la presenza di outliers oltre la soglia di 10000, che potrebbero influenzare l'analisi temporale,
# procedo quindi con la rimozione di tali valori anomali nel file data_management.py

# Carica il file TSV
file_path = '01_Analisi_temporali_consumi/estat_ten00129.tsv'
data = pd.read_csv(file_path, sep='\t')

# Rinomina la colonna per una migliore leggibilità
data.rename(columns={data.columns[0]: 'info'}, inplace=True)

# Separa le informazioni nella prima colonna
info_split = data['info'].str.split(',', expand=True)
data['freq'] = info_split[0]
data['nrg_bal'] = info_split[1]
data['siec'] = info_split[2]
data['unit'] = info_split[3]
data['geo'] = info_split[4]

# Rimuovi la colonna originale
data.drop(columns=['info'], inplace=True)

# Trasforma i dati in formato long per facilitare l'analisi
data_long = data.melt(id_vars=['freq', 'nrg_bal', 'siec', 'unit', 'geo'], var_name='year', value_name='value')

# Converti l'anno in formato numerico
data_long['year'] = data_long['year'].str.strip().astype(int)

# Rimuovi eventuali spazi bianchi e converte i valori in numerico, gestendo i valori mancanti
data_long['value'] = data_long['value'].str.strip()
data_long['value'] = pd.to_numeric(data_long['value'].str.replace(':', ''), errors='coerce')

# Filtra i dati per un particolare paese (es. Italia)
italy_data = data_long[data_long['geo'] == 'IT']

# Visualizza la distribuzione dei dati per identificare eventuali outliers
print(italy_data['value'].describe())

# Crea un grafico boxplot per visualizzare la distribuzione dei dati
plt.figure(figsize=(10, 6))
plt.boxplot(italy_data['value'].dropna())
plt.title('Distribuzione dei consumi energetici annuali in Italia')
plt.ylabel('Consumo energetico (KTOE)')
plt.show()