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

# --- TDA / ML libs (some are optional; used only when called) ---

try:
    from ripser import ripser
    from persim import bottleneck, wasserstein
except ImportError:  # pragma: no cover
    ripser = None
    bottleneck = None
    wasserstein = None

try:
    from gtda.homology import VietorisRipsPersistence
except ImportError:  # pragma: no cover
    VietorisRipsPersistence = None

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering


# ---------------------------------------------------------------------
# 0. Data loading
# ---------------------------------------------------------------------


@dataclass
class ShotData:
    """Container for all shot data (NBA + MBB)."""

    nba_by_season: Dict[str, pd.DataFrame]
    mbb_by_season: Dict[str, pd.DataFrame]

    @property
    def nba_all(self) -> pd.DataFrame:
        return pd.concat(self.nba_by_season.values(), ignore_index=True)

    @property
    def mbb_all(self) -> pd.DataFrame:
        return pd.concat(self.mbb_by_season.values(), ignore_index=True)


def _season_from_filename(path: Path, prefix: str) -> str:
    """Extract season from a filename like 'nba_shots_2015.parquet'."""
    stem = path.stem  # 'nba_shots_2015'
    return stem.replace(prefix, "").strip("_")


def load_shot_data(
    data_root: str | Path = "data",
    nba_prefix: str = "nba_shots_",
    mbb_prefix: str = "mbb_shots_",
) -> ShotData:
    """
    Load all NBA and MBB shot Parquet files into a ShotData object.
    """
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
# 1. Point clouds & filters (for persistent homology)
# ---------------------------------------------------------------------


def make_point_cloud(
    df: pd.DataFrame,
    x_col: str = "coordinate_x",
    y_col: str = "coordinate_y",
) -> np.ndarray:
    """Turn a shot DataFrame into a (n_shots, 2) numpy array of locations."""
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Expected columns '{x_col}' and '{y_col}' in DataFrame")

    pts = df[[x_col, y_col]].to_numpy(dtype=float)
    pts = pts[~np.isnan(pts).any(axis=1)]  # drop NaNs
    return pts


def filter_by_clock(
    df: pd.DataFrame,
    min_sec: Optional[float] = None,
    max_sec: Optional[float] = None,
    clock_col: str = "clock_seconds",
) -> pd.DataFrame:
    """Restrict shots by shot-clock window (inclusive bounds)."""
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
    **point_kwargs,
) -> np.ndarray:
    """
    Concatenate specified seasons and return a point cloud of shot locations.

    Example: early vs late analytics eras.
    """
    dfs: List[pd.DataFrame] = []
    for s in seasons:
        key = str(s)
        if key not in by_season:
            raise KeyError(f"Season '{key}' not in data.")
        dfs.append(by_season[key])
    df_cat = pd.concat(dfs, ignore_index=True)
    return make_point_cloud(df_cat, **point_kwargs)


# ---------------------------------------------------------------------
# 2. Persistent homology
# ---------------------------------------------------------------------


def persistence_ripser(
    X: np.ndarray,
    maxdim: int = 1,
    metric: str = "euclidean",
) -> List[np.ndarray]:
    """
    Compute persistence diagrams for a point cloud using Ripser.

    Returns list diagrams[d] = H_d diagram (n_features, 2).
    """
    if ripser is None:
        raise ImportError("ripser is not installed. `pip install ripser`")

    res = ripser(X, maxdim=maxdim, metric=metric)
    return res["dgms"]


def persistence_gtda(
    X: np.ndarray,
    maxdim: int = 1,
) -> List[np.ndarray]:
    """
    Compute persistence diagrams via giotto-tda's VietorisRipsPersistence.
    """
    if VietorisRipsPersistence is None:
        raise ImportError("giotto-tda is not installed. `pip install giotto-tda`")

    vr = VietorisRipsPersistence(homology_dimensions=list(range(maxdim + 1)))
    dgms = vr.fit_transform(X[None, :, :])  # (1, n_dims, n_features, 2)
    return [dgms[0][k] for k in range(maxdim + 1)]


def diagram_distance(
    dgm1: np.ndarray,
    dgm2: np.ndarray,
    metric: str = "bottleneck",
    **kwargs,
) -> float:
    """
    Distance between two persistence diagrams (same homology dimension).
    """
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
# 3. Player-level features (spatial + clock + shot type)
# ---------------------------------------------------------------------


def _shot_histogram_2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins_x: int,
    bins_y: int,
    normalize: bool = True,
) -> np.ndarray:
    """2D spatial histogram flattened to a vector."""
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
    """
    1D histogram over shot clock ranges.

    Default bins emulate:
        [0,6], (6,15], (15,24]
    """
    if clock_col not in df.columns:
        # If missing, return zeros (so code still runs).
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
    """
    Build simple shot-type distribution features:

        - fraction of 2PA, 3PA
        - fraction of 2PM, 3PM (if we can infer makes)
    """
    # Try to identify a "value" column (2 vs 3)
    score_col = None
    for c in score_col_candidates:
        if c in df.columns:
            score_col = c
            break

    if score_col is None:
        # Can't distinguish 2 vs 3, return zeros.
        return np.zeros(4, dtype=float), ["p_2pa", "p_3pa", "p_2pm", "p_3pm"]

    vals = df[score_col].to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]

    # attempts
    two_mask = vals == 2
    three_mask = vals == 3
    n2a, n3a = two_mask.sum(), three_mask.sum()
    total_att = n2a + n3a
    if total_att == 0:
        two_att, three_att = 0.0, 0.0
    else:
        two_att, three_att = n2a / total_att, n3a / total_att

    # makes (either via scoring flag or result column)
    if scoring_flag in df.columns:
        makes = df[scoring_flag].astype(bool).to_numpy()
    elif result_col in df.columns:
        makes = df[result_col].astype(str).str.lower().eq("made").to_numpy()
    else:
        makes = np.zeros(len(df), dtype=bool)

    vals2 = df[score_col].to_numpy(dtype=float)
    two_makes = (vals2 == 2) & makes
    three_makes = (vals2 == 3) & makes
    n2m, n3m = two_makes.sum(), three_makes.sum()
    total_makes = n2m + n3m

    if total_makes == 0:
        two_make_frac, three_make_frac = 0.0, 0.0
    else:
        two_make_frac, three_make_frac = n2m / total_makes, n3m / total_makes

    v = np.array([two_att, three_att, two_make_frac, three_make_frac], dtype=float)
    if normalize and v.sum() > 0:
        # not strictly necessary, but keeps scale similar to other features
        v /= v.sum()
    feat_names = ["p_2pa", "p_3pa", "p_2pm", "p_3pm"]
    return v, feat_names


def build_player_feature_matrix(
    df: pd.DataFrame,
    min_shots: int = 200,
    bins_x: int = 10,
    bins_y: int = 10,
    clock_bins: Tuple[int, ...] = (0, 6, 15, 24),
    id_col: str = "athlete_id",
    name_col: str = "athlete_name",
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Build feature matrix for players, combining:
        - spatial shot histogram
        - shot-clock distribution
        - shot-type distribution

    Returns
    -------
    X : np.ndarray   shape (n_players, n_features)
    meta : pd.DataFrame  (id, name, n_shots, etc.)
    feature_names : list[str]
    """
    required = [id_col, name_col, "coordinate_x", "coordinate_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    groups = df.groupby(id_col)
    feat_list: List[np.ndarray] = []
    meta_rows: List[dict] = []
    feature_names: Optional[List[str]] = None

    for pid, g in groups:
        n_shots = len(g)
        if n_shots < min_shots:
            continue

        # spatial
        spatial = _shot_histogram_2d(
            g, "coordinate_x", "coordinate_y", bins_x, bins_y, normalize=True
        )
        spatial_names = [
            f"spatial_bin_{i}" for i in range(spatial.size)
        ]

        # clock
        clock = _clock_histogram(
            g, clock_col="clock_seconds", bins=clock_bins, normalize=True
        )
        clock_names = [
            f"clock_bin_{i}"
            for i in range(clock.size)
        ]

        # type
        shot_type, type_names = _shot_type_features(g)

        feat_vec = np.concatenate([spatial, clock, shot_type])

        if feature_names is None:
            feature_names = spatial_names + clock_names + type_names

        feat_list.append(feat_vec)
        meta_rows.append(
            {
                id_col: pid,
                name_col: g[name_col].iloc[0],
                "n_shots": n_shots,
            }
        )

    if not feat_list:
        raise ValueError("No players met the min_shots threshold.")

    X = np.vstack(feat_list)
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)

    assert feature_names is not None
    return X, meta, feature_names


# ---------------------------------------------------------------------
# 4. PCA + clustering for archetypes
# ---------------------------------------------------------------------


@dataclass
class PlayerEmbedding:
    """Result of embedding + clustering player feature vectors."""

    X: np.ndarray               # original features
    X_proc: np.ndarray          # scaled features
    embedding: np.ndarray       # PCA embedding (n_players, n_components)
    meta: pd.DataFrame          # player metadata
    labels: np.ndarray          # cluster labels
    feature_names: List[str]


def embed_and_cluster_players(
    X: np.ndarray,
    meta: pd.DataFrame,
    n_components: int = 2,
    n_clusters: int = 6,
    scale: bool = True,
    method: str = "kmeans",
    random_state: int = 0,
) -> PlayerEmbedding:
    """
    Run PCA + clustering on player feature vectors to obtain archetypes.

    method ∈ {"kmeans", "agg"}.
    """
    X_proc = X
    if scale:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)

    pca = PCA(n_components=n_components, random_state=random_state)
    emb = pca.fit_transform(X_proc)

    if method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters, n_init=10, random_state=random_state
        )
    elif method == "agg":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError("method must be 'kmeans' or 'agg'")

    labels = model.fit_predict(emb)

    return PlayerEmbedding(
        X=X,
        X_proc=X_proc,
        embedding=emb,
        meta=meta.reset_index(drop=True),
        labels=labels,
        feature_names=[],  # can be filled if you want later
    )
