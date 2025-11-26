"""
tdabball.py

Utilities for Topological Data Analysis of NBA & NCAA men's basketball
shot charts, for Math 412 TDA project (Evan & Ricardo).

Goals (from proposal + TDAsports example):
- Build point clouds from shot coordinates and game context.
- Compute persistent homology of shot distributions (Ripser / giotto-tda).
- Compare eras / shot-clock situations via bottleneck / Wasserstein distances.
- Build player feature vectors combining:
    * spatial shot-chart shape
    * shot-clock distribution
    * shot-type distribution (2s vs 3s, makes vs misses)
  then do PCA + clustering to obtain player archetypes.

Data assumptions
----------------
You have Parquet files created in R with hoopR, under:

    data/nba/nba_shots_YYYY.parquet
    data/mbb/mbb_shots_YYYY.parquet

with (at least) columns:

    coordinate_x, coordinate_y
    clock_seconds
    athlete_id, athlete_name
    team_id, season, league
    (optionally) shot_value or score_value, scoring_play, shot_result

The functions are written to be somewhat robust if some
of the optional columns are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Optional TDA / ML libs ---
try:
    from ripser import ripser
    from persim import bottleneck, wasserstein
except ImportError:
    ripser = None
    bottleneck = None
    wasserstein = None

try:
    from gtda.homology import VietorisRipsPersistence
except ImportError:
    VietorisRipsPersistence = None

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering


# ---------------------------------------------------------------------
# 0. Data loading
# ---------------------------------------------------------------------

@dataclass
class ShotData:
    nba_by_season: Dict[str, pd.DataFrame]
    mbb_by_season: Dict[str, pd.DataFrame]

    @property
    def nba_all(self) -> pd.DataFrame:
        return pd.concat(self.nba_by_season.values(), ignore_index=True)

    @property
    def mbb_all(self) -> pd.DataFrame:
        return pd.concat(self.mbb_by_season.values(), ignore_index=True)


def _season_from_filename(path: Path, prefix: str) -> str:
    stem = path.stem
    return stem.replace(prefix, "").strip("_")


def load_shot_data(
    data_root: str | Path = "data",
    nba_prefix: str = "nba_shots_",
    mbb_prefix: str = "mbb_shots_",
) -> ShotData:

    root = Path(data_root)

    nba_files = sorted((root / "nba").glob(f"{nba_prefix}*.parquet"))
    mbb_files = sorted((root / "mbb").glob(f"{mbb_prefix}*.parquet"))

    if not nba_files:
        raise FileNotFoundError(f"No NBA parquet files found under {root/'nba'}")
    if not mbb_files:
        raise FileNotFoundError(f"No MBB parquet files found under {root/'mbb'}")

    nba_by_season = {
        _season_from_filename(f, nba_prefix): pd.read_parquet(f)
        for f in nba_files
    }
    mbb_by_season = {
        _season_from_filename(f, mbb_prefix): pd.read_parquet(f)
        for f in mbb_files
    }

    return ShotData(nba_by_season=nba_by_season, mbb_by_season=mbb_by_season)


# ---------------------------------------------------------------------
# 1. Point clouds & filters
# ---------------------------------------------------------------------

def make_point_cloud(
    df: pd.DataFrame,
    x_col: str = "coordinate_x",
    y_col: str = "coordinate_y",
) -> np.ndarray:

    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Expected columns '{x_col}' and '{y_col}' in DataFrame")

    pts = df[[x_col, y_col]].to_numpy(dtype=float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    return pts


def filter_by_clock(
    df: pd.DataFrame,
    min_sec: Optional[float] = None,
    max_sec: Optional[float] = None,
    clock_col: str = "clock_seconds",
) -> pd.DataFrame:

    if clock_col not in df.columns:
        raise KeyError(f"Expected column '{clock_col}' in DataFrame")

    mask = np.ones(len(df), dtype=bool)
    if min_sec is not None:
        mask &= df[clock_col] >= min_sec
    if max_sec is not None:
        mask &= df[clock_col] <= max_sec
    return df.loc[mask]


def era_cloud(
    by_season: Dict[str, pd.DataFrame],
    seasons: Iterable[int | str],
    **kwargs,
) -> np.ndarray:

    dfs = []
    for s in seasons:
        key = str(s)
        if key not in by_season:
            raise KeyError(f"Season '{key}' not in data.")
        dfs.append(by_season[key])
    df_cat = pd.concat(dfs, ignore_index=True)
    return make_point_cloud(df_cat, **kwargs)


# ---------------------------------------------------------------------
# 2. Persistent homology
# ---------------------------------------------------------------------

def persistence_ripser(
    X: np.ndarray,
    maxdim: int = 1,
    metric: str = "euclidean",
) -> List[np.ndarray]:

    if ripser is None:
        raise ImportError("ripser not installed. Install with: pip install ripser")

    res = ripser(X, maxdim=maxdim, metric=metric)
    return res["dgms"]
    

def persistence_gtda(
    X: np.ndarray,
    maxdim: int = 1,
) -> List[np.ndarray]:
    """
    Persistent homology via giotto-tda's VietorisRipsPersistence.

    Returns a list of diagrams, one per homology dimension, each as a
    float array of shape (n_points, 2) (birth, death).
    """
    if VietorisRipsPersistence is None:
        raise ImportError("giotto-tda not installed. Install with: pip install giotto-tda")

    # giotto expects shape (n_samples, n_points, n_features)
    vr = VietorisRipsPersistence(homology_dimensions=list(range(maxdim + 1)))
    dgms_raw = vr.fit_transform(X[None, :, :])      # shape: (1, n_dims, ...)

    diagrams: List[np.ndarray] = []
    for k in range(maxdim + 1):
        dgm_k = dgms_raw[0, k]                      # usually an object array

        # Convert giotto's ragged/object format to a numeric (n, 2) array
        dgm_k = np.asarray(dgm_k, dtype=object)

        if dgm_k.ndim == 1:
            # common case: array of pairs stored as objects
            # e.g. array([ [b1,d1], [b2,d2], ... ], dtype=object )
            dgm_k = np.vstack(dgm_k)
        elif dgm_k.ndim > 2:
            # just in case giotto gives something weird, flatten all but last axis
            dgm_k = np.reshape(dgm_k, (-1, 2))

        dgm_k = dgm_k.astype(float)                 # ensure numeric
        diagrams.append(dgm_k)

    return diagrams


def _normalize_diagram(dgm: np.ndarray) -> np.ndarray:
    """
    Ensure a persistence diagram is a float array of shape (n_points, 2).

    Handles cases where the diagram is 1-D (flattened) or has weird shapes.
    """
    arr = np.asarray(dgm, dtype=float)

    if arr.ndim == 1:
        # Interpret as flattened [b0, d0, b1, d1, ...]
        if arr.size % 2 != 0:
            raise ValueError("Diagram array must contain an even number of entries.")
        arr = arr.reshape(-1, 2)
    elif arr.ndim > 2:
        # Giotto can sometimes return higher-dimensional arrays;
        # flatten everything except the last axis.
        arr = arr.reshape(-1, 2)

    # At this point we either have shape (n, 2) or (0,)
    if arr.size == 0:
        # Empty diagram: persim can handle it, but keep shape consistent
        arr = arr.reshape(0, 2)

    return arr


def diagram_distance(
    dgm1: np.ndarray,
    dgm2: np.ndarray,
    metric: str = "bottleneck",
    **kwargs,
) -> float:
    """
    Distance between two persistence diagrams (same homology dimension).

    Normalizes input diagrams to shape (n_points, 2) before calling persim.
    """
    dgm1 = _normalize_diagram(dgm1)
    dgm2 = _normalize_diagram(dgm2)

    if metric == "bottleneck":
        if bottleneck is None:
            raise ImportError("persim not installed. `pip install persim`")
        return float(bottleneck(dgm1, dgm2, **kwargs))
    elif metric == "wasserstein":
        if wasserstein is None:
            raise ImportError("persim not installed. `pip install persim`")
        return float(wasserstein(dgm1, dgm2, **kwargs))
    else:
        raise ValueError("metric must be 'bottleneck' or 'wasserstein'")


# ---------------------------------------------------------------------
# 3. Player feature matrix
# ---------------------------------------------------------------------

def _shot_histogram_2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins_x: int,
    bins_y: int,
    normalize: bool = True,
) -> np.ndarray:

    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]

    if xs.size == 0:
        return np.zeros(bins_x * bins_y, dtype=float)

    H, _, _ = np.histogram2d(xs, ys, bins=[bins_x, bins_y])
    v = H.flatten().astype(float)
    if normalize and v.sum() > 0:
        v /= v.sum()
    return v


def _clock_histogram(
    df: pd.DataFrame,
    clock_col: str = "clock_seconds",
    bins: Tuple[int, ...] = (0, 6, 15, 24),
    normalize: bool = True,
) -> np.ndarray:

    if clock_col not in df.columns:
        return np.zeros(len(bins) - 1, dtype=float)

    xs = df[clock_col].to_numpy(dtype=float)
    xs = xs[~np.isnan(xs)]

    if xs.size == 0:
        return np.zeros(len(bins) - 1, dtype=float)

    counts, _ = np.histogram(xs, bins=bins)
    v = counts.astype(float)
    if normalize and v.sum() > 0:
        v /= v.sum()
    return v


def _shot_type_features(
    df: pd.DataFrame,
    score_col_candidates: Tuple[str, ...] = ("shot_value", "score_value"),
    scoring_flag: str = "scoring_play",
    result_col: str = "shot_result",
    normalize: bool = True,
) -> Tuple[np.ndarray, List[str]]:

    score_col = None
    for c in score_col_candidates:
        if c in df.columns:
            score_col = c
            break

    if score_col is None:
        return np.zeros(4, dtype=float), ["p_2pa", "p_3pa", "p_2pm", "p_3pm"]

    vals = df[score_col].to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]

    two_mask = vals == 2
    three_mask = vals == 3
    n2a, n3a = two_mask.sum(), three_mask.sum()
    total_att = n2a + n3a

    if total_att == 0:
        two_att = three_att = 0.0
    else:
        two_att, three_att = n2a / total_att, n3a / total_att

    if scoring_flag in df.columns:
        makes = df[scoring_flag].astype(bool).to_numpy()
    elif result_col in df.columns:
        makes = df[result_col].astype(str).str.lower().eq("made").to_numpy()
    else:
        makes = np.zeros(len(df), dtype=bool)

    vals2 = df[score_col].to_numpy(dtype=float)
    n2m = ((vals2 == 2) & makes).sum()
    n3m = ((vals2 == 3) & makes).sum()
    total_makes = n2m + n3m

    if total_makes == 0:
        two_make_frac = three_make_frac = 0.0
    else:
        two_make_frac, three_make_frac = n2m / total_makes, n3m / total_makes

    vec = np.array([two_att, three_att, two_make_frac, three_make_frac], dtype=float)

    if normalize and vec.sum() > 0:
        vec /= vec.sum()

    names = ["p_2pa", "p_3pa", "p_2pm", "p_3pm"]
    return vec, names


def build_player_feature_matrix(
    df: pd.DataFrame,
    min_shots: int = 200,
    bins_x: int = 10,
    bins_y: int = 10,
    clock_bins: Tuple[int, ...] = (0, 6, 15, 24),
    id_col: str = "athlete_id_1",
    name_col: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:

    required = ["coordinate_x", "coordinate_y", id_col]
    if name_col is not None:
        required.append(name_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    groups = df.groupby(id_col)

    feat_list = []
    meta_rows = []
    feature_names = None

    for pid, g in groups:
        n_shots = len(g)
        if n_shots < min_shots:
            continue

        spatial = _shot_histogram_2d(
            g, "coordinate_x", "coordinate_y", bins_x, bins_y, normalize=True
        )
        spatial_names = [f"spatial_bin_{i}" for i in range(spatial.size)]

        clock = _clock_histogram(
            g, clock_col="clock_seconds", bins=clock_bins, normalize=True
        )
        clock_names = [f"clock_bin_{i}" for i in range(clock.size)]

        shot_type, type_names = _shot_type_features(g)

        vec = np.concatenate([spatial, clock, shot_type])

        if feature_names is None:
            feature_names = spatial_names + clock_names + type_names

        feat_list.append(vec)

        row = {
            "player_id": pid,
            "n_shots": n_shots,
        }
        if name_col is not None and name_col in g.columns:
            row["player_name"] = g[name_col].iloc[0]
        else:
            row["player_name"] = str(pid)

        meta_rows.append(row)

    if not feat_list:
        raise ValueError("No players met the min_shots threshold.")

    X = np.vstack(feat_list)
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)

    return X, meta, feature_names


# ---------------------------------------------------------------------
# 4. PCA + clustering
# ---------------------------------------------------------------------

@dataclass
class PlayerEmbedding:
    X: np.ndarray
    X_proc: np.ndarray
    embedding: np.ndarray
    meta: pd.DataFrame
    labels: np.ndarray
    feature_names: List[str]


def embed_and_cluster_players(
    X: np.ndarray,
    meta: pd.DataFrame,
    n_components: int = 2,
    n_clusters: int = 6,
    scale: bool = True,
    method: str = "kmeans",
    random_state: int = 0,
    feature_names: Optional[List[str]] = None,
) -> PlayerEmbedding:

    X_proc = X
    if scale:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)

    pca = PCA(n_components=n_components, random_state=random_state)
    emb = pca.fit_transform(X_proc)

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    elif method == "agg":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError("method must be 'kmeans' or 'agg'")

    labels = model.fit_predict(emb)

    if feature_names is None:
        feature_names = []

    return PlayerEmbedding(
        X=X,
        X_proc=X_proc,
        embedding=emb,
        meta=meta.reset_index(drop=True),
        labels=labels,
        feature_names=feature_names,
    )
