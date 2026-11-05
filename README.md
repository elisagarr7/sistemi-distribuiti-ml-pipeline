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

La creazione del dataset processato attiva automaticamente la funzione di training. Durante questa fase vengono addestrati e confrontati più modelli di classificazione; il modello migliore viene selezionato automaticamente, serializzato e salvato nel bucket `models`.

Ogni training genera una nuova versione del modello. Il file `current.json` identifica la versione attualmente utilizzata per l'inference.

Sono inoltre disponibili funzioni HTTP per:

- caricare il dataset;
- controllare lo stato della pipeline;
- effettuare predizioni utilizzando il modello attivo.

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

La classificazione utilizza una pipeline basata su:

```text
TF-IDF
+
selezione automatica del classificatore
```

Vengono confrontati tre algoritmi:

```text
Logistic Regression
Multinomial Naive Bayes
Linear SVM
```

Per `Linear SVM` viene utilizzata una calibrazione delle probabilità, in modo da poter ottenere anche la confidence durante l'inference.

Lo split principale train/test viene effettuato a livello di `Cluster ID`, così da evitare che offerte appartenenti allo stesso prodotto siano presenti contemporaneamente nei due insiemi.

Il train set viene successivamente suddiviso in un insieme utilizzato per addestrare i modelli candidati e un validation set. I tre algoritmi vengono confrontati sul validation set utilizzando come criterio principale il **Macro F1** e, in caso di parità, l'accuracy.

Dopo la selezione, il modello vincitore viene riaddestrato sull'intero train set e valutato una sola volta sul test set.

### Model versioning

Ogni esecuzione del training genera un identificativo di versione basato sul timestamp, ad esempio:

```text
20261104T042133Z
```

Gli artefatti sono organizzati nel bucket `models` nel seguente modo:

```text
models/
├── current.json
└── versions/
    └── <version_id>/
        ├── product_classifier.joblib
        └── training_metrics.json
```

`current.json` contiene il riferimento alla versione attiva, all'algoritmo selezionato e ai relativi file di modello e metriche.

Le versioni precedenti vengono mantenute nel bucket e non vengono sovrascritte.

### `inference-faas`

Espone un endpoint HTTP utilizzato per classificare nuovi prodotti.

La richiesta contiene:

```json
{
  "product_title": "apple iphone 13 128gb"
}
```

La risposta contiene:

```text
categoria prevista
confidence
algoritmo utilizzato
versione del modello
```

La funzione utilizza `current.json` per individuare il modello attivo.

È inoltre presente una cache in memoria: dopo il primo caricamento, il modello viene riutilizzato nelle successive richieste finché la versione attiva non cambia. Quando un nuovo training aggiorna `current.json`, la cache viene invalidata automaticamente e viene caricato il nuovo modello.

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
models/current.json
modello della versione attiva
metriche della versione attiva
```

e restituisce lo stato delle fasi:

```text
upload
preprocessing
training
```

Restituisce inoltre:

- metriche finali del modello;
- risultati del confronto tra i modelli candidati;
- algoritmo selezionato;
- versione attiva.

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

## Metriche

Le principali metriche utilizzate sono:

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
```

Per la selezione automatica viene utilizzato principalmente il **Macro F1 sul validation set**.

Le metriche finali mostrate dall'applicazione sono invece calcolate sul **test set** dopo il riaddestramento del modello selezionato.

Vengono inoltre registrati:

```text
Train rows
Test rows
Train clusters
Test clusters
Cluster overlap
```

Il valore `cluster_overlap = 0` verifica l'assenza di cluster condivisi tra train e test.

La baseline iniziale basata su Logistic Regression aveva ottenuto circa:

```text
Accuracy: 95.53%
Macro F1: 95.63%
```

Nella versione finale i risultati possono variare in base al modello selezionato automaticamente e sono salvati nel file `training_metrics.json` associato a ogni versione.

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
|   |-- training/
|   |   |-- main.py
|   |   `-- function.yaml
|   |-- status/
|   |   |-- main.py
|   |   `-- function.yaml
|   |-- upload/
|   |   |-- main.py
|   |   `-- function.yaml
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
MinIO API:        http://localhost:19000
MinIO Console:    http://localhost:19001
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
→ training dei modelli candidati
→ selezione automatica
→ riaddestramento del modello migliore
→ creazione della nuova versione
→ aggiornamento di current.json
```

L'upload può essere effettuato dalla Console MinIO oppure attraverso `upload-faas`.

Il frontend utilizza `upload-faas` per avviare l'intera pipeline dalla dashboard.

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

Tra le informazioni restituite sono presenti anche:

```text
model_version
algorithm
cache_hit
model_load_ms
```

`cache_hit = false` indica che il modello è stato caricato da MinIO, mentre `cache_hit = true` indica che è stato riutilizzato dalla cache in memoria.

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
- visualizzare le metriche finali;
- confrontare i modelli candidati;
- visualizzare il modello e la versione attivi;
- effettuare predizioni.

Per utilizzare correttamente il frontend è necessario che Docker e le funzioni Nuclio del backend siano attivi.