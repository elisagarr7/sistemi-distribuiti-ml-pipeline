import json
import os

from minio import Minio
from minio.error import S3Error


RAW_BUCKET = "raw-data"
RAW_OBJECT = "pricerunner_aggregate.csv"

PROCESSED_BUCKET = "processed-data"
PROCESSED_OBJECT = "products_clean.csv"

MODELS_BUCKET = "models"
MODEL_OBJECT = "product_classifier.joblib"
METRICS_OBJECT = "training_metrics.json"


def json_response(context, body, status_code=200):
    return context.Response(
        body=json.dumps(body),
        headers={},
        content_type="application/json",
        status_code=status_code,
    )


def get_object_info(client, bucket_name, object_name):
    """
    Restituisce i metadati dell'oggetto se esiste.
    Restituisce None se l'oggetto non esiste.
    """
    try:
        return client.stat_object(bucket_name, object_name)

    except S3Error as exc:
        if exc.code in (
            "NoSuchKey",
            "NoSuchObject",
            "NoSuchBucket",
        ):
            return None

        raise


def read_json_object(client, bucket_name, object_name):
    """
    Legge un file JSON da MinIO e lo converte
    in un dizionario Python.
    """
    response = None

    try:
        response = client.get_object(
            bucket_name,
            object_name,
        )

        content = response.read().decode("utf-8")

        return json.loads(content)

    finally:
        if response is not None:
            response.close()
            response.release_conn()


def handler(context, event):
    try:
        endpoint = os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")

        if not endpoint or not access_key or not secret_key:
            return json_response(
                context,
                {
                    "status": "error",
                    "message": "Configurazione MinIO incompleta",
                },
                500,
            )

        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )

        raw_info = get_object_info(
            client,
            RAW_BUCKET,
            RAW_OBJECT,
        )

        processed_info = get_object_info(
            client,
            PROCESSED_BUCKET,
            PROCESSED_OBJECT,
        )

        model_info = get_object_info(
            client,
            MODELS_BUCKET,
            MODEL_OBJECT,
        )

        metrics_info = get_object_info(
            client,
            MODELS_BUCKET,
            METRICS_OBJECT,
        )

        # -------------------------------
        # Verifica della "freschezza"
        # -------------------------------

        processed_is_fresh = (
            raw_info is not None
            and processed_info is not None
            and processed_info.last_modified
            >= raw_info.last_modified
        )

        model_is_fresh = (
            processed_is_fresh
            and model_info is not None
            and model_info.last_modified
            >= processed_info.last_modified
        )

        metrics_are_fresh = (
            processed_is_fresh
            and metrics_info is not None
            and metrics_info.last_modified
            >= processed_info.last_modified
        )

        training_complete = (
            model_is_fresh
            and metrics_are_fresh
        )

        # -------------------------------
        # Stato delle singole fasi
        # -------------------------------

        upload_status = (
            "success"
            if raw_info is not None
            else "idle"
        )

        if raw_info is None:
            preprocessing_status = "idle"

        elif processed_is_fresh:
            preprocessing_status = "success"

        else:
            preprocessing_status = "running"

        if not processed_is_fresh:
            training_status = "idle"

        elif training_complete:
            training_status = "success"

        else:
            training_status = "running"

        # -------------------------------
        # Metriche
        # -------------------------------

        metrics = None

        if metrics_are_fresh:
            metrics = read_json_object(
                client,
                MODELS_BUCKET,
                METRICS_OBJECT,
            )

        result = {
            "status": "success",

            "pipeline_complete": training_complete,

            "stages": {
                "upload": upload_status,
                "preprocessing": preprocessing_status,
                "training": training_status,
            },

            "objects": {
                "raw_dataset": {
                    "exists": raw_info is not None,
                    "last_modified": (
                        raw_info.last_modified.isoformat()
                        if raw_info
                        else None
                    ),
                },

                "processed_dataset": {
                    "exists": processed_info is not None,
                    "fresh": processed_is_fresh,
                    "last_modified": (
                        processed_info.last_modified.isoformat()
                        if processed_info
                        else None
                    ),
                },

                "model": {
                    "exists": model_info is not None,
                    "fresh": model_is_fresh,
                    "last_modified": (
                        model_info.last_modified.isoformat()
                        if model_info
                        else None
                    ),
                },

                "metrics": {
                    "exists": metrics_info is not None,
                    "fresh": metrics_are_fresh,
                    "last_modified": (
                        metrics_info.last_modified.isoformat()
                        if metrics_info
                        else None
                    ),
                },
            },

            "metrics": metrics,
        }

        return json_response(
            context,
            result,
            200,
        )

    except Exception as exc:
        context.logger.error_with(
            "Pipeline status failed",
            error=str(exc),
        )

        return json_response(
            context,
            {
                "status": "error",
                "message": str(exc),
            },
            500,
        )