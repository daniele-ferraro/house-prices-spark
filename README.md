# House Prices con Apache Spark

Progetto per l'esame di Big Data: previsione del prezzo di vendita delle abitazioni per la competizione Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), realizzato con **PySpark**, **Spark ML** e **XGBoost distribuito**.

L'obiettivo è sviluppare un modello che, a partire da un dataset con le caratteristiche di una collezione di abitazioni (Ames Housing dataset, 79 variabili per casa), sia capace di predirne il prezzo di vendita. Ho lavorato prima sui due dataset forniti dalla competizione, imputando i valori mancanti e facendo data cleaning, poi ho addestrato un regressore Random Forest e successivamente uno XGBoost, confrontandone i risultati.

Come richiesto dalla competizione, la metrica è la radice dell'errore quadratico medio tra il logaritmo del prezzo previsto e il logaritmo del prezzo reale (RMSE logaritmico).

## Tecnologie

- **Apache Spark** (PySpark, Spark SQL, Spark ML): caricamento e pulizia dei dati, pipeline, cross validation
- **XGBoost** con `SparkXGBRegressor`, addestramento distribuito su 4 worker
- **HDFS**: nella versione originale i dati venivano letti e scritti su HDFS in locale
- Jupyter Notebook

## Cosa contiene il notebook

1. **Analisi e pulizia dei dati.** Nel dataset il valore `NA` ha due significati: per alcune variabili indica un dato mancante, per altre l'assenza della caratteristica (ad esempio `PoolQC = NA` significa che la casa non ha una piscina). Ho separato i due casi, imputato i valori mancanti (media per quartiere per `LotFrontage`, valore più frequente o 0 negli altri casi) e corretto i tipi delle colonne che Spark aveva letto come stringhe.
2. **Preparazione delle feature.** `StringIndexer` addestrato sull'unione di train e test, così la codifica delle variabili categoriche è la stessa nei due dataset; `VectorAssembler` per costruire il vettore delle feature.
3. **Random Forest.** Un primo addestramento su tutte le feature per ricavarne l'importanza, poi la selezione delle feature combinando importanza e correlazione con `SalePrice`, la feature derivata `Total_square` (superficie totale) e il tuning degli iperparametri con `CrossValidator`.
4. **XGBoost.** Addestramento distribuito con le stesse feature, poi con una selezione basata sull'importanza calcolata da XGBoost stesso e un tuning più ampio (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`).

## Risultati

RMSE logaritmico sul set di validazione (20% del training set) e punteggio ottenuto su Kaggle con il test set della competizione:

| Modello | Validazione | Kaggle |
|---|---|---|
| Random Forest con feature selezionate e tuning | 0,1380 | 0,14719 |
| XGBoost con le feature del Random Forest | 0,1320 | 0,14229 |
| XGBoost con feature selezionate e tuning esteso | **0,1249** | **0,13642** |

Le predizioni inviate a Kaggle sono nella cartella [`output/`](output/).

## Come eseguirlo

1. Scaricare i dati dalla [pagina della competizione](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) e copiare `train.csv` e `test.csv` nella cartella `data/`. I dati non sono inclusi nel repository perché il regolamento di Kaggle non ne consente la ridistribuzione.
2. Installare le dipendenze (serve anche Java 17 o 21 per Spark):
   ```bash
   pip install -r requirements.txt
   ```
3. Aprire `house_prices_spark.ipynb` con Jupyter ed eseguire tutte le celle.

Per leggere i dati da HDFS, come nella versione originale, basta impostare `DATA_DIR` e `OUTPUT_DIR` nella seconda cella con un percorso del tipo `hdfs://localhost:9000/...`.

## Revisione del 2026

Il progetto è stato sviluppato a gennaio 2025 con Spark 3.5. Rileggendolo per pubblicarlo ho corretto alcuni errori e rieseguito tutto il notebook con Spark 4:

- **Importanza delle feature in XGBoost.** `get_score()` restituisce chiavi del tipo `f0, f1, …` e omette le feature mai usate negli split: abbinarle ai nomi delle colonne con `zip()` le associava ai valori sbagliati. Ora il nome viene ricavato dall'indice contenuto nella chiave.
- **Imputazione del test set.** Le medie di `LotFrontage` per quartiere sono calcolate solo sul training set e riusate per il test.
- **Variabile `col` sovrascritta.** Il ciclo sulle correlazioni usava `col` come variabile, nascondendo la funzione `col()` di PySpark.
- Parametri di log di XGBoost passati in modo errato, seed fissato in tutti i modelli, descrizione di `Total_square` corretta (seminterrato, piano terra e primo piano, in piedi quadrati).
