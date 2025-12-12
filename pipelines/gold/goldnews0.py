# --------------------------------------------------------------
# Gold GPU Script 1 — GPU0
# Automatically processes EVEN-index silver files only
# --------------------------------------------------------------

from pathlib import Path
import re, gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# ---------------- PATHS ----------------
FILE_DIR = Path(__file__).resolve().parent
ROOT = FILE_DIR.parent

SILVER_DIR = ROOT / "silver" / "delta_news_silver"
GOLD_DIR = FILE_DIR / "gold_parquet_gpu0"
GOLD_DIR.mkdir(exist_ok=True)

print(">>> GPU0 script started")
print("Silver:", SILVER_DIR)
print("Output:", GOLD_DIR)


# ---------------- LOAD SILVER LIST ----------------
all_files = sorted([p for p in SILVER_DIR.rglob("*.parquet") if "_delta_log" not in str(p)])
silver_files = all_files[0::2]   # EVEN INDEX ONLY

print(f"GPU0 handling {len(silver_files)} silver files")


# ---------------- LM LEXICON ----------------
lm_path = ROOT / "gold" / "lexicons" / "Loughran-McDonald_MasterDictionary_1993-2024.csv"
lm = pd.read_csv(lm_path)
lm["Word"] = lm["Word"].str.lower()

LEX_LM_POS = set(lm[lm["Positive"] > 0]["Word"])
LEX_LM_NEG = set(lm[lm["Negative"] > 0]["Word"])
LEX_LM_UNC = set(lm[lm["Uncertainty"] > 0]["Word"])
LEX_LM_LIT = set(lm[lm["Litigious"] > 0]["Word"])
LEX_LM_CON = set(lm[lm["Constraining"] > 0]["Word"])


# ---------------- GPU0 SETUP ----------------
device = "cuda:0"
print("Using device:", device)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

ner_tok = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER").to(device)
ner_pipe = pipeline("ner", model=ner_model, tokenizer=ner_tok, device=0, aggregation_strategy="simple")


# ---------------- HELPERS ----------------
def tokenize(t):
    return re.findall(r"[A-Za-z]+", t.lower()) if isinstance(t, str) else []


def lexicon_features(t):
    s = set(tokenize(t))
    return (
        len(s & LEX_LM_POS),
        len(s & LEX_LM_NEG),
        len(s & LEX_LM_UNC),
        len(s & LEX_LM_LIT),
        len(s & LEX_LM_CON),
    )


def struct_features(t):
    w = len(t.split())
    return len(t), w, len(t) / max(w, 1)


def compute_emb(texts):
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def ner_counts(t):
    ents = ner_pipe(t[:1500])
    org = gpe = person = money = 0
    for e in ents:
        if e["entity_group"] == "ORG": org += 1
        elif e["entity_group"] == "LOC": gpe += 1
        elif e["entity_group"] == "PER": person += 1
    return org, gpe, person, money


# ---------------- PCA SAMPLE ----------------
sample_texts = []
for f in silver_files:
    try:
        df = pd.read_parquet(f, columns=["text", "len_text"])
        df = df[df["len_text"] > 50]
        sample_texts.extend(df["text"].tolist())
        if len(sample_texts) >= 1500:
            break
    except:
        continue

sample_emb = compute_emb(sample_texts[:1500])
pca = PCA(n_components=min(50, sample_emb.shape[0], sample_emb.shape[1]))
pca.fit(sample_emb)


# ---------------- BATCH GENERATOR ----------------
BASE_COLS = [
    "date","text","publication","author","url",
    "text_type","time_precision","date_trading","tz_hint",
    "dataset","dataset_source","source","anchor_policy",
    "source_file","len_text","silver_ingestion_ts"
]

def batches(batch_size=2500):
    buf = []
    for f in silver_files:
        try:
            df = pd.read_parquet(f, columns=BASE_COLS)
        except:
            print("Skipping corrupt:", f)
            continue

        df = df[df["len_text"] > 50]
        buf.append(df)
        merged = pd.concat(buf, ignore_index=True)

        while len(merged) >= batch_size:
            yield merged.iloc[:batch_size].copy()
            merged = merged.iloc[batch_size:].copy()

        buf = [merged]

    if buf[0].shape[0] > 0:
        yield buf[0]


# ---------------- MAIN LOOP ----------------
batch_idx = 0
print("Starting GPU0 half-pipeline...")

for pdf in tqdm(batches(), desc="GPU0 batches"):
    texts = pdf["text"].tolist()

    emb = compute_emb(texts)
    df_emb = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])

    reduced = pca.transform(emb)
    df_pca = pd.DataFrame(reduced, columns=[f"pca_{i}" for i in range(reduced.shape[1])])

    df_struct = pd.DataFrame([struct_features(t) for t in texts],
                             columns=["len_chars_gold","num_words","avg_word_len"])
  
    df_lex = pd.DataFrame([lexicon_features(t) for t in texts],
                           columns=["lm_pos","lm_neg","lm_unc","lm_lit","lm_con"])

    df_ner = pd.DataFrame([ner_counts(t) for t in texts],
                           columns=["ner_org","ner_gpe","ner_person","ner_money"])

    out = pd.concat([pdf.reset_index(drop=True), df_emb, df_pca, df_struct, df_lex, df_ner], axis=1)
    out.to_parquet(GOLD_DIR / f"gold_{batch_idx:05d}.parquet", index=False)

    batch_idx += 1
    torch.cuda.empty_cache()
    gc.collect()

print("GPU0 complete.")
