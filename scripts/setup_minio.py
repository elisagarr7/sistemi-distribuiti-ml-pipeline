import os
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio

from minio.notificationconfig import (
    NotificationConfig,
    QueueConfig,
    SuffixFilterRule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


BUCKETS = [
    "raw-data",
    "processed-data",
    "models",
]


PREPROCESSING_ARN = (
    "arn:minio:sqs::PREPROCESSING:webhook"
)

TRAINING_ARN = (
    "arn:minio:sqs::TRAINING:webhook"
)


def create_client():
    load_dotenv(ENV_PATH)

    endpoint = os.getenv(
        "MINIO_ENDPOINT",
        "localhost:19000"
    )

    access_key = os.getenv(
        "MINIO_ROOT_USER"
    )

    secret_key = os.getenv(
        "MINIO_ROOT_PASSWORD"
    )

    if not access_key or not secret_key:
        raise ValueError(
            "Credenziali MinIO mancanti nel file .env"
        )

    # Il client MinIO vuole host:porta, non http://...
    endpoint = (
        endpoint
        .replace("http://", "")
        .replace("https://", "")
    )

    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )


def ensure_buckets(client):
    print("\n=== CREAZIONE BUCKET ===")

    for bucket in BUCKETS:

        if client.bucket_exists(
            bucket_name=bucket
        ):
            print(
                f"[OK] Bucket già presente: {bucket}"
            )

        else:
            client.make_bucket(
                bucket_name=bucket
            )

            print(
                f"[CREATO] Bucket: {bucket}"
            )


def configure_raw_notification(client):
    print(
        "\n=== NOTIFICA RAW-DATA ==="
    )

    config = NotificationConfig(
        queue_config_list=[
            QueueConfig(
                queue=PREPROCESSING_ARN,
                events=["s3:ObjectCreated:*"],
                config_id="preprocessing-trigger",
                suffix_filter_rule=SuffixFilterRule(".csv"),
            )
        ]
    )



    client.set_bucket_notification(
        bucket_name="raw-data",
        config=config
    )

    print(
        "[OK] raw-data/*.csv "
        "-> preprocessing-faas"
    )


def configure_processed_notification(client):
    print(
        "\n=== NOTIFICA PROCESSED-DATA ==="
    )

    config = NotificationConfig(
        queue_config_list=[
            QueueConfig(
                queue=TRAINING_ARN,
                events=["s3:ObjectCreated:*"],
                config_id="training-trigger",
                suffix_filter_rule=SuffixFilterRule(
                    "products_clean.csv"
                ),
            )
        ]
    )

    client.set_bucket_notification(
        bucket_name="processed-data",
        config=config
    )

    print(
        "[OK] processed-data/"
        "products_clean.csv "
        "-> training-faas"
    )


def verify_notifications(client):
    print("\n=== VERIFICA NOTIFICHE ===")

    for bucket in [
        "raw-data",
        "processed-data",
    ]:
        config = client.get_bucket_notification(
            bucket_name=bucket
        )

        print(f"\nBucket: {bucket}")

        if not config.queue_config_list:
            print("  Nessuna notifica configurata")
            continue

        for rule in config.queue_config_list:
            print(
                "  ARN:",
                rule.queue
            )

            print(
                "  Eventi:",
                ", ".join(rule.events)
            )

            if rule.suffix_filter_rule:
                print(
                    "  Suffisso:",
                    rule.suffix_filter_rule.value
                )


def main():
    print(
        "================================"
    )

    print(
        "   SETUP MINIO - SDCC PIPELINE"
    )

    print(
        "================================"
    )

    client = create_client()

    # Verifica implicita della connessione
    client.list_buckets()

    print(
        "\n[OK] Connessione a MinIO"
    )

    ensure_buckets(client)

    configure_raw_notification(client)

    configure_processed_notification(client)

    verify_notifications(client)

    print(
        "\n================================"
    )

    print(
        "Setup MinIO completato."
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()