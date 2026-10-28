# Serverless ML Pipeline for Product Classification

Progetto sviluppato per il corso di **Sistemi Distribuiti e Cloud Computing**.

L'obiettivo è realizzare una pipeline di Machine Learning automatizzata basata su architettura **serverless / Function as a Service (FaaS)**, container Docker e object storage.

La pipeline utilizza il **PriceRunner Product Classification and Clustering Dataset** e permette di:

1. caricare il dataset originale in un object storage;
2. avviare automaticamente il preprocessing tramite evento;
3. avviare automaticamente il training del modello sul dataset processato;
4. salvare il modello addestrato nello storage;
5. esporre un endpoint HTTP per effettuare predizioni in tempo reale.

---

## Architettura

La pipeline segue il seguente flusso:

```text
                Upload CSV
                    |
                    v
             +--------------+
             |   raw-data   |
             |    MinIO     |
             +------+-------+
                    |
               ObjectCreated
                    |
                    v
        +-------------------------+
        |   preprocessing-faas    |
        |         Nuclio          |
        +------------+------------+
                     |
                     v
             +----------------+
             | processed-data |
             |     MinIO      |
             +--------+-------+
                      |
                 ObjectCreated
                      |
                      v
             +----------------+
             | training-faas  |
             |     Nuclio     |
             +--------+-------+
                      |
                      v
               +------------+
               |   models   |
               |   MinIO    |
               +------+-----+
                      |
                      v
               +--------------+
 HTTP request ->| inference-faas |
               |    Nuclio     |
               +------+--------+
                      |
                      v
             Predicted category
```

### Componenti principali

- **Docker Engine / Docker Desktop**: runtime dei container.
- **WSL2 + Ubuntu**: ambiente Linux utilizzato per il deployment Nuclio tramite `nuctl`.
- **MinIO**: object storage compatibile con API S3.
- **Nuclio**: piattaforma FaaS.
- **Python**: preprocessing, training e inference.
- **scikit-learn**: pipeline ML.
- **TF-IDF**: rappresentazione testuale dei titoli dei prodotti.
- **Logistic Regression**: classificatore baseline.

---

## Dataset

Il progetto utilizza il dataset:

```text
PriceRunner Product Classification and Clustering Dataset
```

Il dataset originale contiene:

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

Il file processato contiene quindi:

```text
35.311 righe
4 colonne
```

---

## Pipeline di preprocessing

La funzione:

```text
preprocessing-faas
```

viene attivata automaticamente quando viene caricato un file CSV nel bucket:

```text
raw-data
```

Il preprocessing effettua:

- normalizzazione dei nomi delle colonne;
- validazione dello schema;
- controllo dei valori mancanti;
- controllo delle stringhe vuote;
- rimozione di eventuali righe completamente duplicate;
- conversione e validazione degli identificativi numerici;
- normalizzazione minima di `Product Title`;
- conversione in lowercase;
- rimozione degli spazi iniziali/finali;
- normalizzazione degli spazi multipli;
- selezione delle colonne necessarie al training;
- ridenominazione delle colonne in `snake_case`.

Il risultato viene salvato automaticamente in:

```text
processed-data/products_clean.csv
```

---

## Training

La creazione di:

```text
processed-data/products_clean.csv
```

genera un secondo evento MinIO che attiva:

```text
training-faas
```

Il modello utilizza una pipeline scikit-learn composta da:

```text
Product Title
      |
      v
TF-IDF Vectorizer
      |
      v
Logistic Regression
```

### Split train/test

Lo split non viene effettuato direttamente sulle righe.

Viene effettuato a livello di:

```text
cluster_id
```

in modo che tutte le offerte relative allo stesso prodotto reale appartengano esclusivamente al training set oppure al test set.

Questo permette di evitare **data leakage**.

Lo split è approssimativamente:

```text
80% cluster training
20% cluster test
```

ed è stratificato per categoria.

Risultati ottenuti:

```text
Train rows:       28.355
Test rows:         6.956

Train clusters:   10.586
Test clusters:     2.647

Cluster overlap:       0
```

### Metriche ottenute

```text
Accuracy:  0.9552903968
Macro F1:  0.9563341360
```

Il valore di Macro F1 viene utilizzato insieme all'accuracy perché il dataset presenta un moderato sbilanciamento tra le categorie.

Il modello completo viene serializzato tramite Joblib:

```text
models/product_classifier.joblib
```

mentre le metriche vengono salvate in:

```text
models/training_metrics.json
```

---

## Inference

La funzione:

```text
inference-faas
```

espone un endpoint HTTP.

Accetta richieste JSON nel formato:

```json
{
  "product_title": "apple iphone 13 128gb"
}
```

La funzione:

1. normalizza il titolo;
2. recupera il modello dal bucket `models`;
3. esegue la pipeline TF-IDF + Logistic Regression;
4. restituisce categoria prevista e confidence.

Esempio di risposta:

```json
{
  "status": "success",
  "input": {
    "product_title": "apple iphone 13 128gb"
  },
  "normalized_title": "apple iphone 13 128gb",
  "predicted_category": "Mobile Phones",
  "confidence": 0.98
}
```

---

# Struttura del repository

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

---

# Prerequisiti

Ambiente utilizzato per lo sviluppo:

- Windows 10
- Docker Desktop
- Docker Compose
- WSL2
- Ubuntu
- Python 3
- Nuclio 1.16.9
- MinIO Community `RELEASE.2025-10-15T17-29-55Z`

Il deployment delle funzioni Nuclio in modalità locale viene eseguito da Ubuntu/WSL.

---

# Configurazione iniziale

## 1. Creare il file `.env`

Copiare:

```text
.env.example
```

in:

```text
.env
```

e configurare:

```env
MINIO_ROOT_USER=your_minio_user
MINIO_ROOT_PASSWORD=your_minio_password
MINIO_ENDPOINT=localhost:19000
```

Il file `.env` non deve essere versionato.

---

## 2. Creare l'ambiente Python

Da PowerShell:

```powershell
python -m venv .venv
```

Attivazione:

```powershell
.\.venv\Scripts\Activate.ps1
```

Installazione delle dipendenze:

```powershell
python -m pip install -r requirements.txt
```

Per l'ambiente di sviluppo e i notebook:

```powershell
python -m pip install -r requirements-dev.txt
```

---

# Avvio dell'infrastruttura

## 1. Build dei container

Dalla root del progetto:

```powershell
docker compose build
```

L'immagine MinIO viene costruita dal Dockerfile presente in:

```text
infra/minio/Dockerfile
```

---

## 2. Avvio

```powershell
docker compose up -d
```

Verifica:

```powershell
docker compose ps
```

MinIO è disponibile agli endpoint:

```text
API:     http://localhost:19000
Console: http://localhost:19001
```

Nuclio Dashboard:

```text
http://localhost:8070
```

---

# Deployment delle funzioni Nuclio

Il deployment locale tramite `nuctl` viene effettuato da Ubuntu/WSL.

Entrare nella directory del progetto, ad esempio:

```bash
cd /mnt/c/Users/<USER>/Documents/sistemi-distribuiti-ml-pipeline
```

Eseguire:

```bash
./scripts/deploy_functions.sh
```

Lo script distribuisce automaticamente:

```text
preprocessing-faas
training-faas
inference-faas
```

utilizzando i rispettivi:

```text
functions/*/main.py
functions/*/function.yaml
```

e recuperando le credenziali MinIO dal file `.env`.

Per controllare lo stato:

```bash
~/bin/nuctl get functions --platform local
```

Le funzioni devono risultare operative.

---

# Configurazione MinIO

Dopo il primo avvio dell'ambiente o quando viene utilizzato un volume MinIO nuovo:

```powershell
python scripts/setup_minio.py
```

Lo script crea, se non esistono:

```text
raw-data
processed-data
models
```

e configura le notifiche:

```text
raw-data/*.csv
        |
        v
preprocessing-faas
```

e:

```text
processed-data/products_clean.csv
        |
        v
training-faas
```

Lo script può essere rieseguito senza dover creare manualmente i bucket.

---

# Avvio della pipeline

Una volta completato il setup, caricare:

```text
pricerunner_aggregate.csv
```

nel bucket:

```text
raw-data
```

tramite la Console MinIO.

Non è necessario invocare manualmente le funzioni.

Il caricamento genera automaticamente:

```text
raw-data/pricerunner_aggregate.csv
        |
        v
preprocessing-faas
        |
        v
processed-data/products_clean.csv
        |
        v
training-faas
        |
        v
models/product_classifier.joblib

models/training_metrics.json
```

---

# Test dell'endpoint di inference

Recuperare il container della funzione:

```powershell
$infer = docker ps `
  --filter "label=nuclio.io/function-name=inference-faas" `
  --format "{{.Names}}"
```

Recuperare la porta pubblicata:

```powershell
docker port $infer 8080/tcp
```

Ad esempio:

```text
0.0.0.0:61332
```

Creare il body:

```powershell
$body = @{
    product_title = "apple iphone 13 128gb"
} | ConvertTo-Json
```

Effettuare la richiesta:

```powershell
Invoke-RestMethod `
    -Uri "http://localhost:61332" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Sostituire `61332` con la porta effettivamente assegnata alla funzione.

---

# Ripresa del progetto dopo un riavvio

Dopo il riavvio di Docker Desktop:

```powershell
docker compose up -d
```

Verificare:

```powershell
docker compose ps
```

Successivamente, da WSL:

```bash
~/bin/nuctl get functions --platform local
```

Se le funzioni risultano operative, non è necessario ridistribuirle.

Se risultano assenti, unhealthy o non operative:

```bash
./scripts/deploy_functions.sh
```

Normalmente non è necessario rieseguire:

```powershell
python scripts/setup_minio.py
```

perché bucket e notifiche vengono mantenuti nel volume Docker persistente `minio_data`.

---

# Persistenza

MinIO utilizza un Docker named volume:

```text
minio_data
```

I dati quindi sopravvivono a:

```powershell
docker compose stop
docker compose start
```

e:

```powershell
docker compose down
```

Non utilizzare:

```powershell
docker compose down -v
```

se si desidera mantenere i bucket e gli oggetti memorizzati, perché l'opzione `-v` elimina anche i volumi.

---

# Script locali

Sono disponibili anche script per testare le singole fasi indipendentemente da Nuclio.

### Test connessione MinIO

```powershell
python scripts/test_minio_connection.py
```

### Test upload MinIO

```powershell
python scripts/test_minio_upload.py
```

### Preprocessing locale

```powershell
python scripts/run_preprocessing_local.py
```

### Training locale

```powershell
python scripts/run_training_local.py
```

Questi script sono utili per debugging e permettono di verificare la logica applicativa indipendentemente dall'infrastruttura FaaS.

---

# Scelte progettuali

## Object Storage

MinIO è stato scelto come object storage compatibile con API S3 e containerizzabile tramite Docker.

Sono stati separati tre bucket:

```text
raw-data
processed-data
models
```

per separare chiaramente i diversi stadi della pipeline.

## Function as a Service

Nuclio viene utilizzato come piattaforma FaaS locale sopra Docker.

Le funzioni sono:

```text
preprocessing-faas
training-faas
inference-faas
```

Le prime due sono integrate in una pipeline event-driven tramite bucket notification MinIO.

La terza espone un endpoint HTTP per inference real-time.

## Data leakage

Il dataset contiene più offerte commerciali relative allo stesso prodotto reale.

Per questo motivo lo split train/test viene effettuato per:

```text
cluster_id
```

e non direttamente sulle singole righe.

Questo garantisce:

```text
Train Cluster IDs ∩ Test Cluster IDs = ∅
```

e impedisce che offerte dello stesso prodotto compaiano contemporaneamente nei due insiemi.

## Riproducibilità

Per rendere il progetto riproducibile:

- le dipendenze Python sono versionate;
- il server MinIO viene costruito tramite Dockerfile;
- le funzioni Nuclio sono definite tramite `main.py` e `function.yaml`;
- il deployment è automatizzato tramite `deploy_functions.sh`;
- bucket e notifiche sono configurabili tramite `setup_minio.py`;
- le credenziali sono separate dal codice tramite `.env`;
- `.env.example` documenta le variabili necessarie.

---

# Sicurezza

Le credenziali MinIO non vengono salvate nel repository.

Il file:

```text
.env
```

è escluso tramite `.gitignore`.

Il progetto è configurato per una demo e un ambiente di sviluppo locale. Il Docker socket utilizzato da Nuclio e le credenziali root MinIO non rappresentano una configurazione destinata a un ambiente production hardened.

---

# Stato del progetto

Le quattro fasi principali sono state implementate e validate:

```text
Fase 1 - Dataset analysis / EDA        COMPLETATA
Fase 2 - Serverless preprocessing      COMPLETATA
Fase 3 - Serverless ML training        COMPLETATA
Fase 4 - Real-time HTTP inference      COMPLETATA
```

Il caricamento del dataset originale genera automaticamente il dataset processato e il modello addestrato; l'endpoint HTTP di inference utilizza il modello salvato in MinIO per classificare nuovi titoli di prodotto.