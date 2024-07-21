Il codice del file IT_04_Tensorflow implementa un modello di previsione basato su una rete neurale ricorrente di tipo Long Short-Term Memory (LSTM) 
per prevedere i consumi energetici annuali in Italia. 
Questo approccio è utilizzato per catturare e modellare le tendenze temporali e le stagionalità dei dati storici per effettuare previsioni future.

--Passaggi Principali--------------------------------------------------------------------------------------------------------------------------------------------------------------

1. Caricamento e Pulizia dei Dati:

I dati vengono caricati da un file .tsv e riorganizzati in un formato utilizzabile.
Le colonne vengono rinominate e suddivise in variabili separate.
I dati vengono filtrati per includere solo quelli relativi all'Italia e vengono rimossi gli outlier per pulire i dati.

2. Creazione della Serie Temporale:

I dati puliti vengono organizzati in una serie temporale, con l'anno come indice e i valori di consumo energetico come valori della serie.
Preparazione dei Dati per la Rete Neurale:

I dati vengono normalizzati (scalati) per essere compresi tra 0 e 1, un passaggio comune per migliorare le prestazioni delle reti neurali.
I dati vengono suddivisi in un set di addestramento (80%) e un set di test (20%).
Viene creata una funzione per trasformare i dati in un formato adatto per l'input della rete LSTM, creando sequenze temporali con una finestra di osservazione (look_back).

3. Creazione e Addestramento del Modello LSTM:

Viene definito un modello sequenziale con due strati LSTM e uno strato denso (fully connected).
Il modello viene compilato con una funzione di perdita (mean squared error) e un ottimizzatore (Adam).
Il modello viene addestrato sui dati di addestramento per un certo numero di epoche.

4. Previsione con il Modello Addestrato:

Vengono effettuate previsioni sui dati di addestramento e di test.
Le previsioni vengono trasformate di nuovo nelle scale originali (inverso della normalizzazione).

5. Previsione Futura:

Viene utilizzato il modello addestrato per effettuare previsioni sui consumi energetici futuri per i prossimi 10 anni.
Le previsioni future vengono trasformate di nuovo nella scala originale.

6. Visualizzazione:

Viene generato un grafico per visualizzare i dati storici, le previsioni in-sample (sui dati di addestramento) e le previsioni future.
Viene migliorata la leggibilità dell'asse y e abilitata la griglia per una migliore interpretazione.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[TensorFlow e Keras]:

TensorFlow è una libreria open-source per il machine learning sviluppata da Google. Keras è un'API di alto livello per costruire e addestrare modelli di deep learning, integrata in TensorFlow.

[LSTM (Long Short-Term Memory)]:

Le LSTM sono un tipo di rete neurale ricorrente (RNN) particolarmente adatta per catturare le dipendenze a lungo termine nei dati sequenziali. 
Sono utilizzate per gestire problemi di vanishing gradient, tipici delle RNN standard, grazie alla loro capacità di mantenere informazioni per lunghi periodi di tempo.

[Sequential]:

Il modello Sequential in Keras/TensorFlow è una semplice pila lineare di strati. È adatto per costruire modelli layer-by-layer in maniera sequenziale. Ogni layer ha esattamente un input e un output.