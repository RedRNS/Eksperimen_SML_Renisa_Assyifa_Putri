"""
automate_NAMA_KAMU.py
======================
Script otomatis untuk preprocessing dataset Telco Customer Churn.
Tahapan preprocessing identik dengan notebook Eksperimen_NAMA_KAMU.ipynb.

CATATAN: Ganti setiap kemunculan "NAMA_KAMU" dengan username Dicoding Anda.
         Contoh: automate_johndoe.py

Cara menjalankan:
    python automate_NAMA_KAMU.py

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn

Output:
    Folder  telco_churn_preprocessing/ berisi:
        - X_train.csv
        - X_test.csv
        - y_train.csv
        - y_test.csv
    File    correlation_heatmap.png
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # backend non-interaktif (aman untuk script)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ================================================================
# KONFIGURASI GLOBAL
# ================================================================
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)
OUTPUT_DIR      = "telco_churn_preprocessing"
TARGET_COL      = "Churn"
NUMERIC_FEATS   = ["tenure", "MonthlyCharges", "TotalCharges"]
TEST_SIZE       = 0.20
RANDOM_STATE    = 42


# ================================================================
# FUNGSI 1 – LOAD DATA
# ================================================================
def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari filepath (path lokal atau URL).

    Parameters
    ----------
    filepath : str
        Path file CSV lokal atau URL langsung ke file CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame hasil pembacaan file CSV.
    """
    print(f"\n{'='*60}")
    print("[LOAD DATA] Memuat dataset ...")
    print(f"  Sumber  : {filepath}")

    df = pd.read_csv(filepath)

    print(f"  Shape   : {df.shape[0]:,} baris × {df.shape[1]} kolom")
    print(f"\n  Tipe data per kolom:")
    for col, dtype in df.dtypes.items():
        null_n = df[col].isnull().sum()
        print(f"    {col:<25s} {str(dtype):<12s}  null={null_n}")

    return df


# ================================================================
# FUNGSI 2 – PERFORM EDA
# ================================================================
def perform_eda(df: pd.DataFrame) -> None:
    """
    Melakukan EDA ringkas ke terminal dan menyimpan heatmap korelasi.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame asli (belum diproses sama sekali).

    Returns
    -------
    None
    """
    print(f"\n{'='*60}")
    print("[EDA] EXPLORATORY DATA ANALYSIS")

    # Buat salinan khusus EDA – konversi TotalCharges agar NaN terlihat
    df_eda = df.copy()
    df_eda["TotalCharges"] = pd.to_numeric(df_eda["TotalCharges"], errors="coerce")

    # ---- 1. Statistik Deskriptif ----
    print("\n[EDA] Statistik Deskriptif (fitur numerik):")
    print(df_eda.describe().to_string())

    # ---- 2. Missing Values ----
    print("\n[EDA] Missing Values per kolom:")
    mv = df_eda.isnull().sum()
    mv_pct = (mv / len(df_eda) * 100).round(2)
    mv_df = pd.DataFrame({"Jumlah": mv, "Persentase (%)": mv_pct})
    mv_pos = mv_df[mv_df["Jumlah"] > 0]
    if len(mv_pos):
        print(mv_pos.to_string())
    else:
        print("  Tidak ada missing values (kecuali TotalCharges setelah konversi).")
    tc_null = df_eda["TotalCharges"].isnull().sum()
    print(f"  → 'TotalCharges': {tc_null} NaN setelah konversi ke numerik")

    # ---- 3. Distribusi Target ----
    print(f"\n[EDA] Distribusi Target '{TARGET_COL}':")
    counts = df_eda[TARGET_COL].value_counts()
    percs  = (counts / len(df_eda) * 100).round(1)
    for k in counts.index:
        bar = "█" * int(percs[k] // 2)
        print(f"  {k:<5s}: {counts[k]:>5,}  ({percs[k]:>5.1f}%)  {bar}")

    # ---- 4. Simpan heatmap korelasi ----
    df_corr = df_eda.copy()
    df_corr[TARGET_COL] = df_corr[TARGET_COL].map({"No": 0, "Yes": 1})
    num_cols = df_corr.select_dtypes(include=[np.number]).columns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df_corr[num_cols].corr(),
        annot=True, fmt=".3f",
        cmap="coolwarm", center=0,
        square=True, linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "Heatmap Korelasi Fitur Numerik\n(Telco Customer Churn)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    heatmap_path = "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\n[EDA] ✓ Heatmap korelasi disimpan → {heatmap_path}")


# ================================================================
# FUNGSI 3 – PREPROCESS DATA
# ================================================================
def preprocess_data(df: pd.DataFrame):
    """
    Pipeline preprocessing lengkap, identik dengan notebook eksperimen.

    Langkah-langkah (sama persis dengan notebook):
        1. Hapus kolom 'customerID' (identifier, bukan fitur)
        2. Konversi 'TotalCharges' object → float
        3. Imputasi NaN pada TotalCharges dengan median
        4. Hapus baris duplikat
        5. Encode target: No → 0, Yes → 1
        6. LabelEncode seluruh fitur bertipe object
        7. Pisahkan X (fitur) dan y (target)
        8. Split train-test 80:20 (stratified, random_state=42)
        9. StandardScaler pada NUMERIC_FEATS
           (fit hanya pada training set → tidak ada data leakage)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame asli yang dikembalikan oleh load_data().

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) sebagai pd.DataFrame / pd.Series.
    """
    print(f"\n{'='*60}")
    print("[PREPROCESSING] DATA PREPROCESSING")

    df_proc = df.copy()

    # ------------------------------------------------------------------
    # Step 1: Hapus kolom customerID
    # ------------------------------------------------------------------
    df_proc.drop(columns=["customerID"], inplace=True)
    print(f"\n[STEP 1] ✓ Hapus 'customerID'  → shape: {df_proc.shape}")

    # ------------------------------------------------------------------
    # Step 2: Konversi TotalCharges ke float
    #         Beberapa baris baru (tenure=0) berisi string spasi → NaN
    # ------------------------------------------------------------------
    df_proc["TotalCharges"] = pd.to_numeric(
        df_proc["TotalCharges"], errors="coerce"
    )
    n_nan = df_proc["TotalCharges"].isnull().sum()
    print(f"[STEP 2] ✓ Konversi 'TotalCharges' ke float  → {n_nan} NaN ditemukan")

    # ------------------------------------------------------------------
    # Step 3: Imputasi missing values dengan median
    # ------------------------------------------------------------------
    total_mv_before = df_proc.isnull().sum().sum()
    for col in NUMERIC_FEATS:
        if df_proc[col].isnull().sum() > 0:
            med = df_proc[col].median()
            df_proc[col].fillna(med, inplace=True)
            print(f"[STEP 3]   Imputasi '{col}' dengan median = {med:.4f}")
    total_mv_after = df_proc.isnull().sum().sum()
    print(f"[STEP 3] ✓ Total missing values: {total_mv_before} → {total_mv_after}")

    # ------------------------------------------------------------------
    # Step 4: Hapus baris duplikat
    # ------------------------------------------------------------------
    n_dup = df_proc.duplicated().sum()
    df_proc.drop_duplicates(inplace=True)
    print(f"[STEP 4] ✓ Baris duplikat dihapus: {n_dup}  → shape: {df_proc.shape}")

    # ------------------------------------------------------------------
    # Step 5: Encode target variable
    # ------------------------------------------------------------------
    df_proc[TARGET_COL] = df_proc[TARGET_COL].map({"No": 0, "Yes": 1})
    vc = df_proc[TARGET_COL].value_counts()
    print(f"[STEP 5] ✓ Target '{TARGET_COL}' encoded  (0={vc.get(0,0):,}, 1={vc.get(1,0):,})")

    # ------------------------------------------------------------------
    # Step 6: LabelEncode semua fitur kategorikal (dtype=object)
    # ------------------------------------------------------------------
    cat_cols = df_proc.select_dtypes(include="object").columns.tolist()
    le_store: dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col])
        le_store[col] = le
    print(f"[STEP 6] ✓ LabelEncoded {len(cat_cols)} kolom: {cat_cols}")

    # ------------------------------------------------------------------
    # Step 7: Pisahkan X dan y
    # ------------------------------------------------------------------
    X = df_proc.drop(columns=[TARGET_COL])
    y = df_proc[TARGET_COL]
    print(f"[STEP 7] ✓ X: {X.shape}  |  y: {y.shape}")
    print(f"         Kolom fitur ({X.shape[1]}): {list(X.columns)}")

    # ------------------------------------------------------------------
    # Step 8: Split train-test 80:20 (stratified)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[STEP 8] ✓ Split selesai  →  train: {X_train.shape}  |  test: {X_test.shape}")
    print(f"         Distribusi y_train: {dict(y_train.value_counts().sort_index())}")
    print(f"         Distribusi y_test : {dict(y_test.value_counts().sort_index())}")

    # ------------------------------------------------------------------
    # Step 9: StandardScaler – FIT hanya pada training set
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[NUMERIC_FEATS] = scaler.fit_transform(X_train[NUMERIC_FEATS])
    X_test[NUMERIC_FEATS]  = scaler.transform(X_test[NUMERIC_FEATS])
    print(f"[STEP 9] ✓ StandardScaler diterapkan pada: {NUMERIC_FEATS}")
    print(f"         (fit only on train – tidak ada data leakage)")

    # Ringkasan akhir
    print(f"\n[PREPROCESSING] Ringkasan output:")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  y_test  : {y_test.shape}")

    return X_train, X_test, y_train, y_test


# ================================================================
# FUNGSI 4 – SAVE PREPROCESSED DATA
# ================================================================
def save_preprocessed_data(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    output_dir: str,
) -> None:
    """
    Menyimpan X_train, X_test, y_train, y_test ke file CSV.

    Parameters
    ----------
    X_train    : pd.DataFrame – fitur set pelatihan
    X_test     : pd.DataFrame – fitur set pengujian
    y_train    : pd.Series    – label set pelatihan
    y_test     : pd.Series    – label set pengujian
    output_dir : str          – direktori tujuan penyimpanan

    Returns
    -------
    None
    """
    print(f"\n{'='*60}")
    print(f"[SAVE] Menyimpan ke: ./{output_dir}/")

    os.makedirs(output_dir, exist_ok=True)

    files = {
        "X_train.csv": X_train,
        "X_test.csv":  X_test,
        "y_train.csv": y_train.to_frame(),
        "y_test.csv":  y_test.to_frame(),
    }

    for fname, data in files.items():
        fpath = os.path.join(output_dir, fname)
        data.to_csv(fpath, index=False)
        print(f"[SAVE] ✓  {fpath:<50s}  [{data.shape[0]:,} baris × {data.shape[1]} kolom]")


# ================================================================
# FUNGSI 5 – MAIN
# ================================================================
def main() -> None:
    """
    Fungsi utama yang mengorkestrasikan seluruh pipeline preprocessing.
    Panggil fungsi ini saat script dijalankan langsung.
    """
    print("=" * 60)
    print("  AUTOMATE PREPROCESSING – TELCO CUSTOMER CHURN DATASET")
    print("  (Kriteria 1 Skilled – MSML Dicoding)")
    print("=" * 60)

    # 1. Load data
    df = load_data(DATASET_URL)

    # 2. EDA
    perform_eda(df)

    # 3. Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 4. Simpan hasil
    save_preprocessed_data(X_train, X_test, y_train, y_test, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("[SUKSES] Seluruh pipeline preprocessing selesai tanpa error!")
    print(f"[SUKSES] Output  → ./{OUTPUT_DIR}/")
    print(f"[SUKSES] Heatmap → ./correlation_heatmap.png")
    print("=" * 60)


# ================================================================
if __name__ == "__main__":
    main()
