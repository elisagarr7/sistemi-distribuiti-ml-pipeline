import pandas as pd


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


def validate_schema(df: pd.DataFrame) -> None:
    """
    Verifica che il dataset contenga tutte le colonne attese.
    """

    missing_columns = EXPECTED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset non valido. Colonne mancanti: "
            f"{sorted(missing_columns)}"
        )


def preprocess_dataframe(df: pd.DataFrame):
    """
    Esegue il preprocessing del dataset PriceRunner.

    Restituisce:
    - DataFrame processato
    - statistiche sulle trasformazioni effettuate
    """

    df = df.copy()

    stats = {
        "input_rows": len(df)
    }

    # 1. Normalizzazione dei nomi delle colonne
    df.columns = df.columns.str.strip()

    # 2. Validazione dello schema
    validate_schema(df)

    # 3. Rimozione di eventuali righe completamente duplicate
    duplicate_rows = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    stats["duplicate_rows_removed"] = duplicate_rows

    # 4. Conversione degli identificativi richiesti in formato numerico
    df["Product ID"] = pd.to_numeric(
        df["Product ID"],
        errors="coerce"
    )

    df["Cluster ID"] = pd.to_numeric(
        df["Cluster ID"],
        errors="coerce"
    )

    # 5. Conversione delle colonne testuali
    df["Product Title"] = df["Product Title"].astype("string")
    df["Category Label"] = df["Category Label"].astype("string")

    # 6. Individuazione delle righe non valide
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

    # Pulizia minima anche dell'etichetta
    df["Category Label"] = (
        df["Category Label"]
        .str.strip()
    )

    # 8. Conversione definitiva degli ID in interi
    df["Product ID"] = df["Product ID"].astype("int64")
    df["Cluster ID"] = df["Cluster ID"].astype("int64")

    # 9. Selezione delle colonne utili al progetto
    processed_df = df[
        [
            "Product ID",
            "Product Title",
            "Cluster ID",
            "Category Label",
        ]
    ].copy()

    # 10. Ridenominazione in snake_case
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



