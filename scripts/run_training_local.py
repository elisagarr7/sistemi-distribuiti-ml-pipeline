import json
import sys
from pathlib import Path

import joblib
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


from src.storage_client import create_minio_client
from src.training import train_and_evaluate


PROCESSED_BUCKET = "processed-data"
PROCESSED_OBJECT = "products_clean.csv"

MODELS_BUCKET = "models"
MODEL_OBJECT = "product_classifier.joblib"
METRICS_OBJECT = "training_metrics.json"


def main():

    client = create_minio_client()

    work_directory = project_root / "data" / "work"
    artifacts_directory = project_root / "artifacts"

    work_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    artifacts_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_file_path = (
        work_directory
        / "products_clean_from_minio.csv"
    )

    model_file_path = (
        artifacts_directory
        / MODEL_OBJECT
    )

    metrics_file_path = (
        artifacts_directory
        / METRICS_OBJECT
    )

    print("=== TRAINING PIPELINE ===")

    # 1. Download del dataset processato
    print("\n1. Download dataset processato...")

    client.fget_object(
        bucket_name=PROCESSED_BUCKET,
        object_name=PROCESSED_OBJECT,
        file_path=str(processed_file_path)
    )

    print("Download completato.")

    # 2. Lettura del dataset
    print("\n2. Lettura dataset...")

    df = pd.read_csv(
        processed_file_path
    )

    print(
        f"Dataset letto: "
        f"{df.shape[0]} righe, "
        f"{df.shape[1]} colonne."
    )

    # 3. Training
    print("\n3. Training del modello...")

    model, metrics, report = (
        train_and_evaluate(df)
    )

    print("Training completato.")

    # 4. Metriche
    print("\n4. Metriche:")

    print(
        "Train rows:",
        metrics["train_rows"]
    )

    print(
        "Test rows:",
        metrics["test_rows"]
    )

    print(
        "Train clusters:",
        metrics["train_clusters"]
    )

    print(
        "Test clusters:",
        metrics["test_clusters"]
    )

    print(
        "Cluster overlap:",
        metrics["cluster_overlap"]
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    # Classification report completo
    print("\nClassification report:")

    report_df = (
        pd.DataFrame(report)
        .transpose()
    )

    print(report_df)

    # 5. Serializzazione del modello
    print("\n5. Salvataggio modello...")

    joblib.dump(
        model,
        model_file_path
    )

    print(
        "Modello salvato in:",
        model_file_path
    )

    # 6. Salvataggio metriche
    with open(
        metrics_file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # 7. Upload modello
    print("\n6. Upload modello su MinIO...")

    client.fput_object(
        bucket_name=MODELS_BUCKET,
        object_name=MODEL_OBJECT,
        file_path=str(model_file_path),
        content_type="application/octet-stream"
    )

    # 8. Upload metriche
    client.fput_object(
        bucket_name=MODELS_BUCKET,
        object_name=METRICS_OBJECT,
        file_path=str(metrics_file_path),
        content_type="application/json"
    )

    print("Upload completato.")

    print("\n=== TRAINING COMPLETATO ===")


if __name__ == "__main__":
    main()