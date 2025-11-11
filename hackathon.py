"""
Smart Product Pricing Pipeline (Modified Version)
Purpose:
 - Load training & test data
 - Parse text and derive product-level features
 - Extract TF-IDF + SVD features
 - (Optional) Dense embeddings using SentenceTransformer
 - (Optional) Image embeddings placeholder
 - Train LightGBM using log(price) with stratified folds
 - Produce test_out.csv and OOF predictions

Expected Inputs:
 dataset/train.csv
 dataset/test.csv

Outputs:
 oof_preds.npy
 test_out.csv
 model_fold_1.txt ... model_fold_K.txt
"""

import os
import re
import gc
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
import lightgbm as lgb

# Optional dense embedding model
try:
    from sentence_transformers import SentenceTransformer
    EMB_MODEL_AVAILABLE = True
except:
    EMB_MODEL_AVAILABLE = False


# -------------------- Utility Methods --------------------

def detect_pack_quantity(txt: str) -> Tuple[int, str]:
    if pd.isna(txt):
        return 1, ""
    content = txt.lower()

    match = re.search(r"(?:pack of|set of|pack|qty|x)\s*(\d{1,4})", content)
    if match:
        try:
            return int(match.group(1)), content
        except:
            pass

    nums = re.findall(r"\b(\d{1,3})\b", content)
    nums = [int(x) for x in nums if 1 <= int(x) <= 100]
    return (min(nums), content) if nums else (1, content)


def enrich_catalog(df: pd.DataFrame, text_col="catalog_content") -> pd.DataFrame:
    titles, descriptions, packs = [], [], []

    for entry in df[text_col].fillna("").astype(str):
        parts = re.split(r"\n| - | \| ", entry, maxsplit=1)
        title = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""

        pack, _ = detect_pack_quantity(entry)

        titles.append(title)
        descriptions.append(desc)
        packs.append(pack)

    df["title"] = titles
    df["description"] = descriptions
    df["pack_qty"] = packs
    df["title_length"] = df["title"].str.len()
    df["desc_length"] = df["description"].str.len()
    df["has_img"] = df["image_link"].notna() & (df["image_link"].astype(str) != "")

    return df


def placeholder_image_embeddings(df: pd.DataFrame, col="image_link"):
    # Placeholder for real embeddings (e.g., CLIP / ResNet)
    print("Image embeddings disabled (placeholder used).")
    return np.zeros((len(df), 256), dtype=np.float32)


def smape_score(actual, predicted, eps=1e-9):
    actual = np.array(actual, float)
    predicted = np.array(predicted, float)
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    return np.mean(np.abs(predicted - actual) / (denom + eps)) * 100


# -------------------- Main Training Flow --------------------

def run_pipeline():
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    print("Data loaded:", train.shape, test.shape)

    train = enrich_catalog(train)
    test = enrich_catalog(test)

    train["full_text"] = train["title"] + " " + train["description"]
    test["full_text"] = test["title"] + " " + test["description"]

    base_features = ["pack_qty", "title_length", "desc_length", "has_img"]

    # TF-IDF + SVD
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    tfidf.fit(pd.concat([train["full_text"], test["full_text"]]))

    svd = TruncatedSVD(n_components=128, random_state=42)
    X_train_tfidf = svd.fit_transform(tfidf.transform(train["full_text"]))
    X_test_tfidf = svd.transform(tfidf.transform(test["full_text"]))

    # Sentence Transformer Embeddings (optional)
    if EMB_MODEL_AVAILABLE:
        print("Encoding text with SentenceTransformer...")
        model = SentenceTransformer("all-mpnet-base-v2")
        emb_train = model.encode(train["full_text"].tolist(), show_progress_bar=True)
        emb_test = model.encode(test["full_text"].tolist(), show_progress_bar=True)

        pca = PCA(n_components=64, random_state=42)
        all_emb = np.vstack([emb_train, emb_test])
        reduced = pca.fit_transform(all_emb)

        emb_train, emb_test = reduced[: len(train)], reduced[len(train):]
    else:
        print("Dense text embeddings unavailable → using zeros.")
        emb_train = np.zeros((len(train), 64))
        emb_test = np.zeros((len(test), 64))

    # Image embeddings placeholder
    img_train = placeholder_image_embeddings(train)
    img_test = placeholder_image_embeddings(test)

    X_train = np.hstack([
        train[base_features].to_numpy(),
        X_train_tfidf,
        emb_train,
        img_train
    ])
    X_test = np.hstack([
        test[base_features].to_numpy(),
        X_test_tfidf,
        emb_test,
        img_test
    ])

    scaler = StandardScaler()
    scaler.fit(np.vstack([X_train, X_test]))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y = train["price"].fillna(0).clip(lower=0).values
    y_log = np.log1p(y)
    bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    test_predictions = np.zeros(len(test))

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "seed": 42,
    }

    for f, (train_idx, valid_idx) in enumerate(folds.split(X_train, bins)):
        print(f"Fold {f+1}")

        train_ds = lgb.Dataset(X_train[train_idx], label=y_log[train_idx])
        valid_ds = lgb.Dataset(X_train[valid_idx], label=y_log[valid_idx])

        model = lgb.train(
            params,
            train_ds,
            num_boost_round=5000,
            valid_sets=[train_ds, valid_ds],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(200)
            ]
        )

        oof[valid_idx] = model.predict(X_train[valid_idx], num_iteration=model.best_iteration)
        test_predictions += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

        model.save_model(f"model_fold_{f+1}.txt")
        gc.collect()

    oof_prices = np.expm1(oof)
    smape_val = smape_score(train["price"], oof_prices)
    print("OOF SMAPE (%):", smape_val)

    np.save("oof_preds.npy", oof)

    final_test_prices = np.expm1(test_predictions)
    floor = max(0.01, train["price"].clip(lower=0.01).min())
    final_test_prices = np.maximum(final_test_prices, floor)

    submission = pd.DataFrame({"sample_id": test["sample_id"], "price": final_test_prices})
    submission.to_csv("test_out.csv", index=False)
    print("Saved test_out.csv")


if _name_ == "_main_":
    run_pipeline()