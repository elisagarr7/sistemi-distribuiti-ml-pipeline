# Serverless ML Pipeline

Progetto sviluppato per il corso di **Sistemi Distribuiti e Cloud Computing**.

L'obiettivo è realizzare una pipeline di Machine Learning basata su un'architettura serverless/FaaS per la classificazione automatica di prodotti a partire dal loro titolo.

Il progetto utilizza il **PriceRunner Product Classification and Clustering Dataset**.

## Tecnologie utilizzate

- Python
- Docker e Docker Compose
- MinIO
- Nuclio
- scikit-learn
- pandas
- WSL2
- Joblib

## Funzionamento generale

Il dataset originale viene caricato nel bucket MinIO `raw-data`.

Il caricamento genera automaticamente un evento che attiva la funzione di preprocessing. Il dataset elaborato viene quindi salvato nel bucket `processed-data`.

La creazione del dataset processato attiva automaticamente la funzione di training, che addestra il modello e salva modello e metriche nel bucket `models`.

Sono inoltre disponibili delle funzioni HTTP per:

- caricare il dataset;
- controllare lo stato della pipeline;
- effettuare predizioni utilizzando il modello addestrato.

## Funzioni Nuclio

Il progetto utilizza cinque funzioni FaaS.

### `preprocessing-faas`

Esegue il preprocessing del dataset originale.

Le principali operazioni effettuate sono:

- controllo dello schema;
- normalizzazione dei nomi delle colonne;
- controllo dei valori mancanti;
- rimozione di eventuali duplicati;
- normalizzazione di `Product Title`;
- selezione delle colonne utilizzate dal modello.

Il risultato viene salvato in:

```text
processed-data/products_clean.csv
```

### `training-faas`

Viene attivata automaticamente dopo la creazione del dataset processato.

Il modello utilizza:

```text
TF-IDF
+
Logistic Regression
```

Lo split train/test viene effettuato a livello di `Cluster ID` per evitare che offerte appartenenti allo stesso prodotto siano presenti contemporaneamente nei due insiemi.

Gli output vengono salvati in:

```text
models/product_classifier.joblib
models/training_metrics.json
```

### `inference-faas`

Espone un endpoint HTTP utilizzato per classificare nuovi prodotti.

La richiesta contiene:

```json
{
  "product_title": "apple iphone 13 128gb"
}
```

La risposta contiene la categoria prevista e la confidence associata.

La funzione è esposta localmente sulla porta:

```text
8085
```

### `upload-faas`

Permette di caricare il dataset attraverso una richiesta HTTP.

Il file ricevuto viene inserito nel bucket:

```text
raw-data
```

Il caricamento genera l'evento MinIO che avvia automaticamente il preprocessing.

La funzione è esposta sulla porta:

```text
8084
```

### `status-faas`

Permette di controllare lo stato della pipeline.

La funzione verifica la presenza e l'aggiornamento di:

```text
raw-data/pricerunner_aggregate.csv
processed-data/products_clean.csv
models/product_classifier.joblib
models/training_metrics.json
```

e restituisce lo stato delle fasi:

```text
upload
preprocessing
training
```

Restituisce inoltre le metriche del modello quando il training è completato.

La funzione è esposta sulla porta:

```text
8086
```

## Dataset

Il dataset utilizzato contiene:

```text
35.311 righe
7 colonne
10 categorie
13.233 cluster
```

Le colonne originali sono:

```text
Product ID
Product Title
Merchant ID
Cluster ID
Cluster Label
Category ID
Category Label
```

Dopo il preprocessing vengono mantenute:

```text
product_id
product_title
cluster_id
category_label
```

## Risultati del modello

Nel test effettuato sono stati ottenuti:

```text
Accuracy:     circa 95.53%
Macro F1:     circa 95.63%

Train rows:   28.355
Test rows:     6.956

Train cluster: 10.586
Test cluster:   2.647
Cluster overlap: 0
```

Il valore `cluster_overlap = 0` conferma che non sono presenti cluster contemporaneamente nel training set e nel test set.

## Struttura principale del progetto

```text
sistemi-distribuiti-ml-pipeline/
|
|-- .env.example
|-- .gitignore
|-- compose.yaml
|-- requirements.txt
|-- requirements-dev.txt
|-- README.md
|
|-- data/
|   |-- raw/
|   |   `-- pricerunner_aggregate.csv
|   |
|   |-- work/
|   `-- processed/
|
|-- notebooks/
|   `-- 01_eda_dataset.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- storage_client.py
|   |-- preprocessing.py
|   `-- training.py
|
|-- scripts/
|   |-- test_minio_connection.py
|   |-- test_minio_upload.py
|   |-- run_preprocessing_local.py
|   |-- run_training_local.py
|   |-- setup_minio.py
|   `-- deploy_functions.sh
|
|-- functions/
|   |-- preprocessing/
|   |   |-- main.py
|   |   `-- function.yaml
|   |
|   |-- training/
|   |   |-- main.py
|   |   `-- function.yaml
|   |
|   |-- status/
|   |   |-- main.py
|   |   `-- function.yaml
|   |
|   |-- upload/
|   |   |-- main.py
|   |   `-- function.yaml
|   |
|   `-- inference/
|       |-- main.py
|       `-- function.yaml
|
|-- infra/
|   `-- minio/
|       `-- Dockerfile
|
|-- nuclio/
|   `-- platform.yaml
|
|-- tools/
|   |-- mc/
|   `-- nuctl/
|
`-- artifacts/
```

## Configurazione

Creare un file `.env` partendo da `.env.example`.

Esempio:

```env
MINIO_ROOT_USER=your_minio_user
MINIO_ROOT_PASSWORD=your_minio_password
MINIO_ENDPOINT=localhost:19000
```

Il file `.env` contiene le credenziali locali e non deve essere caricato su Git.

## Installazione delle dipendenze Python

Creare un ambiente virtuale:

```powershell
python -m venv .venv
```

Attivarlo:

```powershell
.\.venv\Scripts\Activate.ps1
```

Installare le dipendenze:

```powershell
python -m pip install -r requirements.txt
```

Per l'ambiente di sviluppo:

```powershell
python -m pip install -r requirements-dev.txt
```

## Avvio dell'ambiente Docker

Dalla directory principale del progetto:

```powershell
docker compose up -d
```

Per verificare i container:

```powershell
docker compose ps
```

I servizi principali sono disponibili su:

```text
MinIO API:       http://localhost:19000
MinIO Console:   http://localhost:19001
Nuclio Dashboard: http://localhost:8070
```

## Configurazione iniziale di MinIO

Per creare i bucket e configurare le notifiche:

```powershell
python scripts/setup_minio.py
```

Vengono utilizzati tre bucket:

```text
raw-data
processed-data
models
```

## Deployment delle funzioni

Il deployment delle funzioni Nuclio viene eseguito da WSL.

Dalla directory del progetto:

```bash
./scripts/deploy_functions.sh
```

Per controllarne lo stato:

```bash
~/bin/nuctl get functions --platform local
```

Le funzioni previste sono:

```text
preprocessing-faas
training-faas
inference-faas
upload-faas
status-faas
```

## Avvio della pipeline

La pipeline può essere avviata caricando il dataset in `raw-data`.

Il flusso è:

```text
Upload dataset
→ preprocessing
→ salvataggio dataset processato
→ training
→ salvataggio modello e metriche
```

L'upload può essere effettuato dalla Console MinIO oppure attraverso `upload-faas`.

## Test inference

Esempio PowerShell:

```powershell
$body = @{
    product_title = "apple iphone 13 128gb"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://localhost:8085" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

## Test stato pipeline

```powershell
Invoke-RestMethod `
    -Uri "http://localhost:8086" `
    -Method Get
```

## Frontend

È disponibile un frontend React separato che permette di:

- caricare il dataset;
- controllare l'avanzamento della pipeline;
- visualizzare le metriche del training;
- effettuare predizioni.

Per utilizzare correttamente il frontend è necessario che Docker e le funzioni Nuclio del backend siano attivi.