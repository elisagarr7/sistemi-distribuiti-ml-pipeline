import sys
from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.preprocessing import preprocess_dataframe
from src.storage_client import create_minio_client


RAW_BUCKET = "raw-data"
RAW_OBJECT = "pricerunner_aggregate.csv"

PROCESSED_BUCKET = "processed-data"
PROCESSED_OBJECT = "products_clean.csv"


def main():

    client = create_minio_client()

    # Cartelle locali utilizzate durante il test
    work_directory = project_root / "data" / "work"
    processed_directory = project_root / "data" / "processed"

    work_directory.mkdir(parents=True, exist_ok=True)
    processed_directory.mkdir(parents=True, exist_ok=True)

    raw_file_path = (
        work_directory
        / "pricerunner_aggregate_from_minio.csv"
    )

    processed_file_path = (
        processed_directory
        / PROCESSED_OBJECT
    )

    print("=== PREPROCESSING PIPELINE ===")

    # 1. Download del dataset raw da MinIO
    print("\n1. Download dataset da MinIO...")

    client.fget_object(
        bucket_name=RAW_BUCKET,
        object_name=RAW_OBJECT,
        file_path=str(raw_file_path)
    )

    print("Download completato.")

    # 2. Lettura con Pandas
    print("\n2. Lettura dataset...")

    df = pd.read_csv(raw_file_path)

    print(
        f"Dataset letto: {df.shape[0]} righe, "
        f"{df.shape[1]} colonne."
    )

    # 3. Preprocessing
    print("\n3. Esecuzione preprocessing...")

    processed_df, stats = preprocess_dataframe(df)


    #temp
    print("\nColonne dataset processato:")
    print(processed_df.columns.tolist())

    print("\nPrime 5 righe:")
    print(processed_df.head())

    print("\nDimensioni:")
    print(processed_df.shape)




    # 4. Visualizzazione statistiche
    print("\nStatistiche preprocessing:")

    print(
        "Record iniziali:",
        stats["input_rows"]
    )

    print(
        "Duplicati rimossi:",
        stats["duplicate_rows_removed"]
    )

    print(
        "Record invalidi rimossi:",
        stats["invalid_rows_removed"]
    )

    print(
        "Record finali:",
        stats["output_rows"]
    )

    # 5. Salvataggio locale
    print("\n4. Salvataggio dataset processato...")

    processed_df.to_csv(
        processed_file_path,
        index=False
    )

    print(
        "Dataset salvato localmente in:",
        processed_file_path
    )

    # 6. Upload su MinIO
    print("\n5. Upload in processed-data...")

    client.fput_object(
        bucket_name=PROCESSED_BUCKET,
        object_name=PROCESSED_OBJECT,
        file_path=str(processed_file_path),
        content_type="text/csv"
    )

    print("Upload completato.")

    # 7. Verifica dell'oggetto
    object_info = client.stat_object(
        bucket_name=PROCESSED_BUCKET,
        object_name=PROCESSED_OBJECT
    )

    print(
        f"\nOggetto presente in MinIO: "
        f"{object_info.object_name}"
    )

    print(
        f"Dimensione: {object_info.size} byte"
    )

    print("\n=== PREPROCESSING COMPLETATO ===")






if __name__ == "__main__":
    main()


