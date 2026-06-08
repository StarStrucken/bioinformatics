import numpy as np
import pandas as pd

def clean_vec(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    ok = np.isfinite(x)
    mean = np.where(ok, x, 0.0).sum(axis=0) / np.maximum(ok.sum(axis=0), 1)
    x = np.where(ok, x, mean)
    out = (x - mean) / (x.std(axis=0) + 1e-6)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def parse_seq_ids(x):
    if x is None or pd.isna(x):
        return ()

    s = str(x).strip()

    if not s:
        return ()

    return tuple(int(v) for v in s.split())

def seq_words(seq, word_size=2):
    seq = tuple(seq)
    word_size = max(1, int(word_size))

    if len(seq) < word_size:
        return ()

    return tuple(seq[i:i + word_size] for i in range(len(seq) - word_size + 1))

def edit_distance(a, b, insert_cost=1.0, delete_cost=1.0, substitute_cost=2.0):
    if a == b:
        return 0.0

    if not a:
        return len(b) * insert_cost

    if not b:
        return len(a) * delete_cost

    prev = np.arange(len(b) + 1, dtype=np.float32) * insert_cost

    for i, ca in enumerate(a, start=1):
        curr = np.empty(len(b) + 1, dtype=np.float32)
        curr[0] = i * delete_cost

        for j, cb in enumerate(b, start=1):
            sub = 0.0 if ca == cb else substitute_cost
            curr[j] = min(
                prev[j] + delete_cost,
                curr[j - 1] + insert_cost,
                prev[j - 1] + sub,
            )

        prev = curr

    return float(prev[-1])

def normalized_edit_distance(a, b):
    return edit_distance(a, b, 1.0, 1.0, 2.0) / max(len(a), len(b), 1)

def jaccard_distance(a, b):
    sa = set(a)
    sb = set(b)

    if not sa and not sb:
        return 0.0

    return 1.0 - len(sa & sb) / max(len(sa | sb), 1)

def local_alignment_score(a, b, match=2.0, mismatch=-1.0, gap=-1.0):
    if not a or not b:
        return 0.0

    prev = np.zeros(len(b) + 1, dtype=np.float32)
    best = 0.0

    for ca in a:
        curr = np.zeros(len(b) + 1, dtype=np.float32)

        for j, cb in enumerate(b, start=1):
            sub = match if ca == cb else mismatch
            v = max(
                0.0,
                float(prev[j] + gap),
                float(curr[j - 1] + gap),
                float(prev[j - 1] + sub),
            )
            curr[j] = v
            best = max(best, v)

        prev = curr

    return float(best)

def local_alignment_distance(a, b):
    score = local_alignment_score(a, b, 2.0, -1.0, -1.0)
    denom = max(2.0 * min(len(a), len(b)), 1.0)
    return float(np.clip(1.0 - score / denom, 0.0, 1.0))

def blast_like_distance(a, b, word_size=2):
    wa = set(seq_words(a, word_size))
    wb = set(seq_words(b, word_size))

    if not wa and not wb:
        word_dist = 0.0
    elif not wa or not wb:
        word_dist = 1.0
    else:
        word_dist = 1.0 - len(wa & wb) / max(len(wa | wb), 1)

    return float(np.clip(0.35 * word_dist + 0.65 * local_alignment_distance(a, b), 0.0, 1.0))

def sequence_distance(a, b, measurement):
    if measurement == "seq_local":
        return local_alignment_distance(a, b)

    if measurement == "seq_jaccard":
        return jaccard_distance(a, b)

    if measurement == "seq_blast":
        return blast_like_distance(a, b, 2)

    return normalized_edit_distance(a, b)
