#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

NUCTL="${HOME}/bin/nuctl"

if [ ! -x "$NUCTL" ]; then
    echo "Errore: nuctl non trovato in $NUCTL"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "Errore: file .env non trovato"
    exit 1
fi


# Caricamento delle sole credenziali necessarie
while IFS='=' read -r key value || [ -n "$key" ]; do

    key="${key#$'\xEF\xBB\xBF'}"
    key="${key//$'\r'/}"
    value="${value//$'\r'/}"

    [[ -z "$key" || "$key" == \#* ]] && continue

    case "$key" in
        MINIO_ROOT_USER|MINIO_ROOT_PASSWORD)
            export "$key=$value"
            ;;
    esac

done < .env


if [ -z "${MINIO_ROOT_USER:-}" ] || [ -z "${MINIO_ROOT_PASSWORD:-}" ]; then
    echo "Errore: credenziali MinIO mancanti nel file .env"
    exit 1
fi


deploy_function() {

    local path="$1"
    local name="$2"

    echo
    echo "======================================"
    echo "Deploy: $name"
    echo "======================================"

    "$NUCTL" deploy \
        --path "$path" \
        --platform local \
        --project-name sdcc-ml-pipeline \
        --env "MINIO_ACCESS_KEY=$MINIO_ROOT_USER" \
        --env "MINIO_SECRET_KEY=$MINIO_ROOT_PASSWORD"

}


deploy_function \
    "./functions/preprocessing" \
    "preprocessing-faas"


deploy_function \
    "./functions/training" \
    "training-faas"


deploy_function \
    "./functions/inference" \
    "inference-faas"

deploy_function \
    "./functions/upload" \
    "upload-faas"

deploy_function \
    "./functions/status" \
    "status-faas"


echo
echo "======================================"
echo "Funzioni distribuite"
echo "======================================"

"$NUCTL" get functions --platform local

echo
echo "Deployment completato."
