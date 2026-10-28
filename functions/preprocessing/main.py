import io
import json
import os

import pandas as pd
from minio import Minio
import nuclio_sdk
from urllib.parse import unquote_plus

EXPECTED_COLUMNS = {
    "Product ID",
    "Product Title",
    "Merchant ID",
    "Cluster ID",
    "Cluster Label",
    "Category ID",
    "Category Label",
}

REQUIRED_COLUMNS = [
    "Product ID",
    "Product Title",
    "Cluster ID",
    "Category Label",
]


def validate_schema(df):
    missing_columns = EXPECTED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset non valido. Colonne mancanti: "
            f"{sorted(missing_columns)}"
        )


def preprocess_dataframe(df):
    df = df.copy()

    stats = {
        "input_rows": len(df)
    }

    # 1. Normalizzazione delle intestazioni
    df.columns = df.columns.str.strip()

    # 2. Validazione dello schema
    validate_schema(df)

    # 3. Rimozione duplicati completi
    duplicate_rows = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    stats["duplicate_rows_removed"] = duplicate_rows

    # 4. Conversione degli identificativi
    df["Product ID"] = pd.to_numeric(
        df["Product ID"],
        errors="coerce"
    )

    df["Cluster ID"] = pd.to_numeric(
        df["Cluster ID"],
        errors="coerce"
    )

    # 5. Conversione colonne testuali
    df["Product Title"] = df["Product Title"].astype("string")
    df["Category Label"] = df["Category Label"].astype("string")

    # 6. Ricerca record non validi
    missing_required = df[REQUIRED_COLUMNS].isna().any(axis=1)

    empty_title = (
        df["Product Title"]
        .str.strip()
        .eq("")
        .fillna(True)
    )

    empty_category = (
        df["Category Label"]
        .str.strip()
        .eq("")
        .fillna(True)
    )

    invalid_rows_mask = (
        missing_required
        | empty_title
        | empty_category
    )

    invalid_rows = int(invalid_rows_mask.sum())

    df = df.loc[~invalid_rows_mask].copy()

    stats["invalid_rows_removed"] = invalid_rows

    # 7. Normalizzazione minima del titolo
    df["Product Title"] = (
        df["Product Title"]
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["Category Label"] = (
        df["Category Label"]
        .str.strip()
    )

    # 8. Conversione definitiva degli ID
    df["Product ID"] = df["Product ID"].astype("int64")
    df["Cluster ID"] = df["Cluster ID"].astype("int64")

    # 9. Selezione delle colonne utili
    processed_df = df[
        [
            "Product ID",
            "Product Title",
            "Cluster ID",
            "Category Label",
        ]
    ].copy()

    # 10. Ridenominazione
    processed_df = processed_df.rename(
        columns={
            "Product ID": "product_id",
            "Product Title": "product_title",
            "Cluster ID": "cluster_id",
            "Category Label": "category_label",
        }
    )

    stats["output_rows"] = len(processed_df)

    return processed_df, stats


def get_minio_client():
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    if not endpoint or not access_key or not secret_key:
        raise ValueError(
            "Configurazione MinIO incompleta"
        )

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )


def parse_request_body(event):
    body = event.body

    if not body:
        return {}

    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))

    if isinstance(body, str):
        return json.loads(body)

    if isinstance(body, dict):
        return body

    return {}


def extract_input_object(body):
    """
    Estrae bucket e object name.

    Supporta:
    - eventi automatici MinIO
    - chiamate HTTP manuali usate durante i test
    """

    # Evento automatico proveniente da MinIO
    if "Records" in body and body["Records"]:
        record = body["Records"][0]

        try:
            bucket = record["s3"]["bucket"]["name"]
            object_name = record["s3"]["object"]["key"]
        except KeyError as exc:
            raise ValueError(
                "Evento MinIO non valido: bucket/object mancanti"
            ) from exc

        # I nomi degli oggetti negli eventi possono essere URL-encoded
        object_name = unquote_plus(object_name)

        event_name = record.get(
            "eventName",
            "unknown"
        )

        return bucket, object_name, event_name

    # Manteniamo compatibilità con il vecchio test manuale
    bucket = body.get(
        "bucket",
        "raw-data"
    )

    object_name = body.get(
        "object",
        "pricerunner_aggregate.csv"
    )

    return bucket, object_name, "manual-http"


def handler(context: nuclio_sdk.Context, event: nuclio_sdk.Event):
    try:
        body = parse_request_body(event)

        raw_bucket, raw_object, source_event = extract_input_object(body)

        processed_bucket = "processed-data"
        processed_object = "products_clean.csv"

        context.logger.info_with(
            "Preprocessing started",
            bucket=raw_bucket,
            object=raw_object,
            source_event=source_event
        )

        client = get_minio_client()

        # Download dell'oggetto da MinIO
        response = client.get_object(
            raw_bucket,
            raw_object
        )

        try:
            raw_bytes = response.read()
        finally:
            response.close()
            response.release_conn()

        # Lettura CSV
        df = pd.read_csv(
            io.BytesIO(raw_bytes)
        )

        # Preprocessing
        processed_df, stats = preprocess_dataframe(df)

        # Conversione del risultato in CSV in memoria
        output_buffer = io.BytesIO()

        processed_df.to_csv(
            output_buffer,
            index=False
        )

        processed_bytes = output_buffer.getvalue()

        # Upload del risultato in MinIO
        client.put_object(
            bucket_name=processed_bucket,
            object_name=processed_object,
            data=io.BytesIO(processed_bytes),
            length=len(processed_bytes),
            content_type="text/csv"
        )

        context.logger.info_with(
            "Preprocessing completed",
            input_rows=stats["input_rows"],
            output_rows=stats["output_rows"],
            duplicates_removed=stats["duplicate_rows_removed"],
            invalid_rows_removed=stats["invalid_rows_removed"]
        )

        return context.Response(
            body=json.dumps(
                {
                    "status": "success",
                    "input": f"{raw_bucket}/{raw_object}",
                    "output": (
                        f"{processed_bucket}/"
                        f"{processed_object}"
                    ),
                    "statistics": stats
                }
            ),
            headers={},
            content_type="application/json",
            status_code=200
        )

    except Exception as exc:
        context.logger.error_with(
            "Preprocessing failed",
            error=str(exc)
        )

        return context.Response(
            body=json.dumps(
                {
                    "status": "error",
                    "message": str(exc)
                }
            ),
            headers={},
            content_type="application/json",
            status_code=500
        )