import pandas as pd
import numpy as np
import pickle
import os
import warnings
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# -------------------------------------------------
# 1. DATA PREPROCESSING (Direct Replica)
# -------------------------------------------------
def load_data(path):
    print(f"[START] Loading Data from {path}...")
    if path.endswith('.tsv'):
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, sep=",", low_memory=False)

def clean_target(df):
    df = df.copy()
    
    col_map = {c.strip().lower(): c for c in df.columns}
    def get_col(name):
        return col_map.get(name.lower())

    status_col = get_col("commit_status")
    
    # Normalize commit_status safely just like user logic
    df[status_col] = (
        df[status_col]
        .astype(str)
        .str.strip()
        .str.replace("_", " ", regex=False)
        .str.upper()
    )

    mapping = {
        "ONTIME": 0,
        "COMMIT FAIL": 1,
        "POD COMMIT FAIL": 2
    }

    df["target"] = df[status_col].map(mapping)
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)

    return df

def drop_irrelevant_columns(df):
    df = df.copy()
    header_map = {c.lower().strip(): c for c in df.columns}
    
    cols_to_drop = [
        "Prime Trk Nos", "Emp Nos", "Consignee Comp", "Consignee Name", "Map",
        "prime trk nos", "emp nos", "consignee comp", "consignee name", "map"
    ]
    
    actual_drops = [header_map.get(c.lower()) for c in cols_to_drop if c.lower() in header_map]
    return df.drop(columns=[c for c in actual_drops if c], errors="ignore")

# -------------------------------------------------
# 2. FEATURE ENGINEERING (Direct Replica)
# -------------------------------------------------
def create_time_features(df):
    df = df.copy()
    
    header_map = {c.lower().strip(): c for c in df.columns}
    ist_col = header_map.get("ist_svc_commit_tmstp")

    if ist_col and ist_col in df.columns:
        df["IST_svc_commit_tmstp"] = pd.to_datetime(
            df[ist_col],
            errors="coerce"
        )
        if ist_col != "IST_svc_commit_tmstp":
            df = df.drop(columns=[ist_col])

        df["commit_hour"] = df["IST_svc_commit_tmstp"].dt.hour
        df["commit_day"] = df["IST_svc_commit_tmstp"].dt.day
        df["commit_month"] = df["IST_svc_commit_tmstp"].dt.month
        df["commit_weekday"] = df["IST_svc_commit_tmstp"].dt.weekday

        # Weekend indicator
        df["is_weekend"] = df["commit_weekday"].apply(
            lambda x: 1 if x in [5, 6] else 0
        )

        def hour_bucket(h):
            if pd.isna(h):
                return "Unknown"
            elif 6 <= h < 12:
                return "Morning"
            elif 12 <= h < 18:
                return "Afternoon"
            elif 18 <= h < 24:
                return "Evening"
            else:
                return "Night"

        df["commit_hour_bucket"] = df["commit_hour"].apply(hour_bucket)

    # -------------------------------------------------------------
    # EXPANDED FEATURES: Geographical Vectors for Variance Handling
    # -------------------------------------------------------------
    bso_col = header_map.get('bso_cd', 'bso_cd')
    if bso_col in df.columns and "commit_weekday" in df.columns:
        df["Station_Day_Cross"] = df[bso_col].astype(str) + "_" + df["commit_weekday"].astype(str)
        
    pstl_col = header_map.get('recp_pstl_cd', 'recp_pstl_cd')
    if pstl_col in df.columns:
        df["postal_zone"] = df[pstl_col].astype(str).str[:3]
        
    qty_col = header_map.get('shp_pce_qty', 'shp_pce_qty')
    if qty_col in df.columns:
        qty_s = pd.to_numeric(df[qty_col], errors="coerce").fillna(1.0)
        df["qty_bins"] = pd.cut(
            qty_s,
            bins=[0, 1, 3, 10, 50, 999999],
            labels=['1', '2-3', '4-10', '11-50', '50+'],
            right=True
        ).astype(str)

    scan_col = header_map.get('last_scan', 'last_scan')
    if scan_col in df.columns:
        def group_scan(s):
            s = str(s).lower()
            if 'clearance' in s or 'customs' in s: return 'Customs/Clearance'
            if 'delay' in s or 'exception' in s: return 'Delay/Exception'
            if 'departed' in s or 'arrived' in s or 'transit' in s: return 'In Transit'
            if 'delivery' in s or 'delivered' in s: return 'Delivery Phase'
            if 'facility' in s or 'hub' in s: return 'Facility Operations'
            return 'Other/Unknown'
            
        df['last_scan_category'] = df[scan_col].apply(group_scan)

    return df

# -------------------------------------------------
# 3. Sliding Time Split (Recent-Focused)
# -------------------------------------------------
def sliding_time_split(df):
    if "IST_svc_commit_tmstp" not in df.columns:
        df["IST_svc_commit_tmstp"] = pd.to_datetime(df.get("ist_svc_commit_tmstp"), errors="coerce")
        
    sort_key = pd.to_datetime(df["IST_svc_commit_tmstp"], errors="coerce").fillna(pd.Timestamp.min)
    df = df.iloc[sort_key.argsort()]

    n = len(df)
    train_start = int(n * 0.20)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    train_df = df.iloc[train_start:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df

# -------------------------------------------------
# 4. Threshold Optimization
# -------------------------------------------------
def optimize_threshold(y_true, probs):
    best_thresh = 0.50
    best_score = -999

    from sklearn.metrics import accuracy_score, recall_score

    for thresh in np.arange(0.20, 0.75, 0.01):
        preds = (probs > thresh).astype(int)
        
        acc = accuracy_score(y_true, preds)
        rec = recall_score(y_true, preds)
        
        # Option 2: High Safety/Recall Model 
        # Aggressively push Recall up (73%), strictly penalize Accuracy dropping below 69.5%
        penalty = 0 if acc >= 0.695 else (0.695 - acc) * 20
        
        score = rec - penalty

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    
    # USING THE MERGED DATASET AS REQUESTED
    df = load_data("final_merged_dataset.csv")

    print("[DONE] Cleaning Target...")
    df = clean_target(df)

    print("\n[INFO] Binary Target Distribution (0=OnTime, 1=Breached):")
    df["target_binary"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

    print("[DONE] Dropping Irrelevant Columns...")
    df = drop_irrelevant_columns(df)

    print("[DONE] Creating Time Features...")
    df = create_time_features(df)

    print("[DONE] Handling Missing Values...")
    df = df.fillna("Unknown")

    from sklearn.model_selection import train_test_split

    print("\n[INFO] Dataset Shape:", df.shape)

    # -------------------------------------------------
    # Randomized Stratified Split (Train 80%, Val 10%, Test 10%)
    # -------------------------------------------------
    train_temp, test_df = train_test_split(df, test_size=0.10, stratify=df["target_binary"], random_state=42)
    train_df, val_df = train_test_split(train_temp, test_size=0.1111, stratify=train_temp["target_binary"], random_state=42)

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    # Encode Categoricals
    categorical_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()

    header_map = {c.lower().strip(): c for c in df.columns}
    status_col = header_map.get("commit_status")
    if status_col in categorical_cols:
        categorical_cols.remove(status_col)

    if "IST_svc_commit_tmstp" in categorical_cols:
         categorical_cols.remove("IST_svc_commit_tmstp")
         
    for c in ["Trk Nos", "trk nos"]:
        if c in categorical_cols: categorical_cols.remove(c)

    target_encoders = {}

    for col in categorical_cols:
        global_mean = train_df["target_binary"].mean()
        
        # Smoothed Target Encoding Mathematics
        agg = train_df.groupby(col)["target_binary"].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        
        # Smoothing weight of 20 
        smooth = 20
        smoothed = (counts * means + smooth * global_mean) / (counts + smooth)
        
        encoding_dict = smoothed.to_dict()
        
        # Overwrite Strings with Continuous Historical Probabilities
        train_df[col] = train_df[col].map(encoding_dict).fillna(global_mean)
        val_df[col] = val_df[col].map(encoding_dict).fillna(global_mean)
        test_df[col] = test_df[col].map(encoding_dict).fillna(global_mean)

        target_encoders[col] = {
            'map': encoding_dict,
            'default': global_mean
        }

    # Strip and lower all test dataframe column names to prevent drop mismatches
    train_df.columns = [str(c).lower().strip() for c in train_df.columns]
    val_df.columns = [str(c).lower().strip() for c in val_df.columns]
    test_df.columns = [str(c).lower().strip() for c in test_df.columns]

    drop_cols = [
        status_col.lower() if status_col else "", "target", "target_binary", "ist_svc_commit_tmstp",
        "time_diff_hours", "last_scan", "time_remaining_commit", "last scan date time", "last scan loc", 
        "trk nos", "dest loc", "cntry_cd", "map", "emp nos"
    ]
    drop_cols = [c for c in drop_cols if c]

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    y_train = train_df["target_binary"]

    X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns], errors="ignore")
    y_val = val_df["target_binary"]

    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")
    y_test = test_df["target_binary"]
    
    # Save the exact expected column order for inference compatibility
    model_features = X_train.columns.tolist()

    # Class Weighting
    scale_pos_weight = (len(y_train[y_train == 0]) / len(y_train[y_train == 1]))

    # -------------------------------------------------
    # XGBoost Algorithm
    # -------------------------------------------------
    print("\n[START] Training Improved Binary XGBoost (Random Split)...")
    
    # Numeric constraint
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_val   = X_val.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test  = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

    print("\n[INFO] scale_pos_weight:", scale_pos_weight)
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=1500,
        max_depth=9,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5,
        reg_alpha=2,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # -------------------------------------------------
    # Validation Evaluation (Threshold Optimizer)
    # -------------------------------------------------
    val_probs = model.predict_proba(X_val)[:, 1]
    best_thresh, best_acc = optimize_threshold(y_val, val_probs)

    print("\n[RESULT] Best Threshold (Optimized on Validation):", best_thresh)
    
    val_preds = (val_probs > best_thresh).astype(int)
    print("\n[REPORT] Validation Performance:")
    print(classification_report(y_val, val_preds))

    # -------------------------------------------------
    # Final Test Set Evaluation
    # -------------------------------------------------
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs > best_thresh).astype(int)

    print("\n[REPORT] Test Performance:")
    print(classification_report(y_test, test_preds))

    # Calculate exact metrics dictionary for UI
    metrics = {
        "accuracy": accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds),
        "recall": recall_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds),
        "roc_auc": roc_auc_score(y_test, test_probs),
        "dataset_size": len(df)
    }

    # -------------------------------------------------
    # SINGLE-FILE COMPILATION
    # -------------------------------------------------
    print("\nPackaging Single-File Multi-Object Bundle...")
    
    model_bundle = {
        'model'          : model,
        'threshold'      : best_thresh,
        'target_encoders': target_encoders,
        'metrics'        : metrics,
        'expected_cols'  : model_features,
        'version'        : 'Merged_XGBoost_TargetEncoded'
    }

    path = 'shipment_model.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model_bundle, f)

    print("\n[DONE] Stable Binary Model Saved to:", path)


if __name__ == "__main__":
    main()
