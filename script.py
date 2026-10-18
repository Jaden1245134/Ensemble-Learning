import os, time, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from joblib import Parallel, delayed          # process-level parallelism only
from scipy.stats import gaussian_kde          # used only to draw density plots

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

DATA_PATH    = "networkTraffic.csv"
RANDOM_STATE = 42            # fixed everywhere so results are reproducible
N_JOBS       = os.cpu_count()
K            = 10            

CLASS_NAMES = {0: "Normal", 1: "Reconnaissance", 2: "Backdoor", 3: "DoS", 4: "Exploits",
               5: "Analysis", 6: "Fuzzers", 7: "Worms", 8: "Shellcode", 9: "Generic"}
LABELS = list(range(K))
NAMES  = [CLASS_NAMES[i] for i in LABELS]

QUICK = os.environ.get("NB_QUICK", "0") == "1"

SUBSAMPLE   = 8_000  if QUICK else 60_000     
T_MAX       = 40     if QUICK else 200        
CHECKPOINTS = [1, 2, 3, 5, 10, 20, 40] if QUICK else \
              [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
TUNE_T      = 20     if QUICK else 100        
N_RUNS      = 2      if QUICK else int(os.environ.get("NB_RUNS", 30))  
CACHE_DIR   = "runs_cache_quick" if QUICK else "runs_cache"
print(f"QUICK={QUICK}  subsample={SUBSAMPLE:,}  jobs={N_JOBS}")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FAMILY_COLOR = {"Decision tree": "#2a78d6", "k-NN": "#eb6834", "Random forest": "#1baf7a"}

MODELS = ["Tree (baseline)", "Tree (balanced)",
          "k-NN (baseline)", "k-NN (prior-corrected)",
          "RF (unweighted)", "RF (weighted)"]
FAMILY   = {m: ("Decision tree" if m.startswith("Tree") else "k-NN" if m.startswith("k-NN") else "Random forest")
            for m in MODELS}
COL      = {m: FAMILY_COLOR[FAMILY[m]] for m in MODELS}
TREATED  = {m: m in ("Tree (balanced)", "k-NN (prior-corrected)", "RF (weighted)") for m in MODELS}
LSTYLE   = {m: "-" if TREATED[m] else "--" for m in MODELS}
HATCH    = {m: "////" if TREATED[m] else "" for m in MODELS}

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 8,
})
# one-hue sequential ramp (light -> dark blue) for heatmaps
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

def annotate_heatmap(ax, M, fmt="{:.2f}", vmin=0.0, vmax=None, min_show=0.0, fs=6.5):
    """Write the value in each cell: dark ink on light cells, light ink on dark cells (judged against the colour scale vmin..vmax)."""
    vmax = np.nanmax(M) if vmax is None else vmax
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v) or v < min_show:
                continue
            frac = (v - vmin) / max(vmax - vmin, 1e-12)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color="white" if frac > 0.5 else INK)

df = pd.read_csv(DATA_PATH, low_memory=False)
print("Shape:", df.shape)
df.head()

counts = df["attack_cat"].value_counts().sort_index()
summary = pd.DataFrame({"class": [CLASS_NAMES[i] for i in counts.index],
                        "n": counts.values,
                        "pct": (counts.values / len(df) * 100).round(2)})
summary.sort_values("n", ascending=False)

fig, ax = plt.subplots(figsize=(8, 3.6))
s = summary.sort_values("n", ascending=False)
ax.bar(s["class"], s["n"], color=FAMILY_COLOR["Decision tree"], width=0.62)
ax.set_yscale("log"); ax.set_ylabel("Records (log scale)")
ax.set_title("Class distribution (note the log axis)")
plt.xticks(rotation=35, ha="right"); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.show()

# 2.1 Disguised missing values: '?' placeholders
print("Conventional NaN in the dataframe:", int(df.isna().sum().sum()))
for col in df.columns:
    n = int((df[col].astype(str) == "?").sum())
    if n:
        print(f"  '?' in {col:10s} {n:>7,} rows ({n/len(df)*100:5.1f}%)")

# 2.2 Exact duplicate rows (id excluded: it is unique per row and would mask every duplicate)
dupes = int(df.drop(columns=["id"]).duplicated().sum())
print(f"Exact duplicate rows: {dupes:,} ({dupes/len(df)*100:.1f}%)")

# 2.3 Contradictory labels: identical features, different class
feat_cols = [c for c in df.columns if c not in ("id", "attack_cat")]
h = pd.util.hash_pandas_object(df[feat_cols], index=False)
nclass = pd.DataFrame({"h": h, "y": df["attack_cat"]}).groupby("h")["y"].transform("nunique")
print(f"Rows with a feature-identical twin carrying a different label: {int((nclass > 1).sum()):,}")

# 2.4 `id` is a unique descriptive feature and leaks class information
print("id unique per row:", df["id"].is_unique)
df.groupby("attack_cat")["id"].mean().rename(index=CLASS_NAMES).round(0).to_frame("mean id")

# 2.5 Cardinality of the nominal features
for col in ["proto", "state", "service"]:
    vc = df[col].value_counts()
    print(f"{col}: {df[col].nunique()} levels; top 5 {dict(vc.head(5))}; levels with < 10 rows: {(vc < 10).sum()}")

# 2.6 Skew, outliers and incomparable ranges
num = df.select_dtypes(include=[np.number]).drop(columns=["id", "attack_cat"])
skew_tbl = pd.DataFrame({"min": num.min(), "median": num.median(), "max": num.max(), "skew": num.skew(),
                         "max / p99": num.max() / num.quantile(0.99).replace(0, np.nan)}
                        ).sort_values("skew", ascending=False)
skew_tbl.head(8).round(2)

# 2.7 Correlated features
corr = num.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pairs = (upper.stack().sort_values(ascending=False).rename("abs_corr").reset_index()
              .rename(columns={"level_0": "feature_a", "level_1": "feature_b"}))
print(f"{(pairs.abs_corr > 0.95).sum()} feature pairs with |r| > 0.95")
pairs[pairs.abs_corr > 0.95].round(4)

# 2.8 Values that violate the data dictionary (features.csv declares both as Binary)
print("is_ftp_login:", df["is_ftp_login"].value_counts().to_dict())
print("ct_ftp_cmd  :", df["ct_ftp_cmd"].value_counts().to_dict())

data = df.copy()
data = data.drop(columns=["id"])                                  # leaking identifier
data["service"] = data["service"].replace("?", "none")            # '?' placeholder is a category
data["is_ftp_login"] = (data["is_ftp_login"] > 0).astype(int)     # clip out-of-domain binary
before = len(data)
data = data.drop_duplicates().reset_index(drop=True)              # remove exact duplicates BEFORE splitting
print(f"Removed {before - len(data):,} duplicates -> {len(data):,} rows remain")

# Collapse rare nominal levels (same levels kept as Assignment 2)
for col, keep_n in [("proto", 12), ("state", 6), ("service", 13)]:
    top = data[col].value_counts().nlargest(keep_n).index
    data[col] = np.where(data[col].isin(top), data[col], "other")
    print(f"{col:8s} -> {data[col].nunique()} levels")

# Label-noise ceiling: the best accuracy ANY deterministic classifier can reach on this data.
# Rows with identical features but different labels cannot all be classified correctly.
feat = [c for c in data.columns if c != "attack_cat"]
hh = pd.util.hash_pandas_object(data[feat], index=False)
g = pd.DataFrame({"h": hh, "y": data["attack_cat"]})
cnt = g.groupby(["h", "y"]).size()
ceiling = cnt.groupby(level=0).max().sum() / len(data)
n_conf = int((g.groupby("h")["y"].transform("nunique") > 1).sum())
print(f"Rows in feature-identical groups with conflicting labels: {n_conf:,} ({n_conf/len(data)*100:.1f}%)")
print(f"Accuracy ceiling for any deterministic classifier on these features: {ceiling:.4f}")

# Stratified subsample
if SUBSAMPLE and len(data) > SUBSAMPLE:
    data = (data.groupby("attack_cat", group_keys=False)
                .sample(frac=SUBSAMPLE / len(data), random_state=RANDOM_STATE)
                .reset_index(drop=True))
print("Working set:", data.shape)
after = data["attack_cat"].value_counts().sort_index()
pd.DataFrame({"class": NAMES, "n": after.values, "pct": (after.values / len(data) * 100).round(2)}
             ).sort_values("n", ascending=False)

y_all = data["attack_cat"].to_numpy().astype(np.int64)
X = data.drop(columns=["attack_cat"]).reset_index(drop=True)

NOMINAL = ["proto", "state", "service"]
NUMERIC = [c for c in X.columns if c not in NOMINAL]
print(f"{len(NUMERIC)} numeric features, {len(NOMINAL)} nominal features")

def stratified_split(y, test_size=0.30, seed=RANDOM_STATE):
    """Per-class permutation split. Returns sorted (train_idx, test_idx) positions."""
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_te = int(round(len(idx) * test_size))
        te.append(idx[:n_te]); tr.append(idx[n_te:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))

tr_idx, te_idx = stratified_split(y_all, 0.30, RANDOM_STATE)
y_train, y_test = y_all[tr_idx], y_all[te_idx]
print(f"Train: {len(tr_idx):,}   Test: {len(te_idx):,}")
pd.DataFrame({"train": np.bincount(y_train, minlength=K), "test": np.bincount(y_test, minlength=K)}, index=NAMES)

class TreePrep:
    """Ordinal-encode the nominal features and bin every feature to uint8. Fitted on TRAIN only."""
    feature_names = NUMERIC + NOMINAL

    def __init__(self, n_bins=64):
        self.n_bins = n_bins

    def _matrix(self, Xdf):
        cols = [Xdf[c].to_numpy(float) for c in NUMERIC]
        for c in NOMINAL:
            lv = self.levels[c]
            cols.append(Xdf[c].map(lv).fillna(len(lv)).to_numpy(float))   # unseen level -> its own code
        return np.column_stack(cols)

    def fit(self, Xdf):
        self.levels = {c: {v: i for i, v in enumerate(sorted(Xdf[c].unique()))} for c in NOMINAL}
        M = self._matrix(Xdf)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = []
        for j in range(M.shape[1]):
            u = np.unique(M[:, j])
            if len(u) <= self.n_bins:
                self.edges.append((u[:-1] + u[1:]) / 2)                  # one bin per distinct value (exact)
            else:
                self.edges.append(np.unique(np.quantile(M[:, j], qs)))   # quantile edges
        return self

    def transform(self, Xdf):
        M = self._matrix(Xdf)
        return np.column_stack([np.searchsorted(e, M[:, j], side="left")
                                for j, e in enumerate(self.edges)]).astype(np.uint8)


class KnnPrep:
    """log1p -> standardise -> one-hot, after dropping correlated numerics. Fitted on TRAIN only."""
    def fit(self, Xdf):
        c = Xdf[NUMERIC].corr().abs()
        u = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
        self.num_cols = [f for f in NUMERIC if not (u[f] > 0.95).any()]      
        L = np.log1p(Xdf[self.num_cols].to_numpy(float))
        self.mu, self.sd = L.mean(0), L.std(0)
        self.sd[self.sd == 0] = 1.0
        self.levels = {col: sorted(Xdf[col].unique()) for col in NOMINAL}
        return self

    def transform(self, Xdf):
        L = (np.log1p(Xdf[self.num_cols].to_numpy(float)) - self.mu) / self.sd
        oh = [(Xdf[col].to_numpy()[:, None] == np.array(self.levels[col])[None, :]).astype(float)
              for col in NOMINAL]                                              # unseen level -> all zeros
        return np.hstack([L] + oh).astype(np.float32)

assert (X[NUMERIC].min() >= 0).all(), "log1p assumes non-negative numerics"
tprep = TreePrep().fit(X.iloc[tr_idx]);  kprep = KnnPrep().fit(X.iloc[tr_idx])
Btr, Bte = tprep.transform(X.iloc[tr_idx]), tprep.transform(X.iloc[te_idx])
Ztr, Zte = kprep.transform(X.iloc[tr_idx]), kprep.transform(X.iloc[te_idx])
P_FEATURES = Btr.shape[1]
print(f"Tree / forest input: {Btr.shape}  (uint8 bin indices, {len(tprep.feature_names)} features)")
print(f"k-NN input         : {Ztr.shape}  ({len(kprep.num_cols)} numeric after pruning {len(NUMERIC) - len(kprep.num_cols)} correlated + one-hot)")
print("Dropped for k-NN only:", [c for c in NUMERIC if c not in kprep.num_cols])

NB = 64   # histogram bins per feature (uint8 bin indices 0..63)

def confusion(y, p):
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)

def scores(y, p):
    """Return (metric dict, per-class F1, confusion matrix). Averages run over classes present in y."""
    cm = confusion(y, p).astype(float)
    tp, sup, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    rec  = np.divide(tp, sup,  out=np.zeros(K), where=sup > 0)
    prec = np.divide(tp, pred, out=np.zeros(K), where=pred > 0)
    f1   = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    present = sup > 0
    return ({"accuracy": tp.sum() / cm.sum(),
             "balanced_accuracy": rec[present].mean(),
             "f1_macro": f1[present].mean(),
             "f1_weighted": (f1 * sup).sum() / sup.sum()}, f1, cm)

def class_weights(y, gamma=1.0):
    """w_c = (n / (K n_c)) ** gamma.  gamma=0: none.  gamma=1: 'balanced'.  0<gamma<1: partial."""
    c = np.bincount(y, minlength=K).astype(float)
    return (len(y) / (K * np.maximum(c, 1))) ** gamma


class HistTree:
    """CART classification tree (Gini) on binned features, with a random feature subset per node
    and optional per-class weights. Grown iteratively with an explicit stack."""

    def __init__(self, max_depth=None, min_leaf=1, mtry=None, class_weight=None, seed=0):
        self.max_depth, self.min_leaf, self.mtry = max_depth, max(1, int(min_leaf)), mtry
        self.class_weight, self.seed = class_weight, seed

    def fit(self, Xb, y):
        n, p = Xb.shape
        m = p if self.mtry is None else min(int(self.mtry), p)
        max_depth = 60 if self.max_depth is None else self.max_depth
        rng = np.random.RandomState(self.seed)
        w = None if self.class_weight is None else self.class_weight[y]
        offs = (np.arange(m) * NB)[None, :]
        feat, thr, left, right, value = [], [], [], [], []

        def new_node(idx):
            c = np.bincount(y[idx], weights=None if w is None else w[idx], minlength=K)
            feat.append(-1); thr.append(0); left.append(-1); right.append(-1)
            value.append(c / c.sum())
            return len(feat) - 1

        root_idx = np.arange(n)
        stack = [(new_node(root_idx), root_idx, 0)]
        while stack:
            nid, idx, depth = stack.pop()
            if depth >= max_depth or len(idx) < 2 * self.min_leaf:
                continue
            yy = y[idx]
            if yy.min() == yy.max():                     
                continue
            fs = rng.permutation(p)[:m] if m < p else np.arange(p)
            flat = ((offs + Xb[np.ix_(idx, fs)]) * K + yy[:, None]).ravel()      # (feature, bin, class) cell per entry
            hr = np.bincount(flat, minlength=m * NB * K).reshape(m, NB, K)       # raw class histograms
            cl = np.cumsum(hr.sum(2), axis=1)                                    # rows with bin <= b
            ok = (cl >= self.min_leaf) & (cl[:, -1:] - cl >= self.min_leaf)
            if w is None:
                hw = hr.astype(np.float64)
            else:
                hw = np.bincount(flat, weights=np.repeat(w[idx], m), minlength=m * NB * K).reshape(m, NB, K)
            L = np.cumsum(hw, axis=1)                                            # class counts left of every threshold
            R = L[:, -1:, :] - L
            with np.errstate(divide="ignore", invalid="ignore"):
                score = (L * L).sum(2) / L.sum(2) + (R * R).sum(2) / R.sum(2)    # maximise == minimise Gini
            score = np.where(ok & np.isfinite(score), score, -np.inf)
            fi, bi = divmod(int(np.argmax(score)), NB)
            if score[fi, bi] == -np.inf:
                continue
            f = fs[fi]
            go_left = Xb[idx, f] <= bi
            li, ri = idx[go_left], idx[~go_left]
            ln, rn = new_node(li), new_node(ri)
            feat[nid], thr[nid], left[nid], right[nid] = int(f), int(bi), ln, rn
            stack.append((ln, li, depth + 1)); stack.append((rn, ri, depth + 1))

        self.feat  = np.array(feat,  dtype=np.int16);  self.thr   = np.array(thr,  dtype=np.uint8)
        self.left  = np.array(left,  dtype=np.int32);  self.right = np.array(right, dtype=np.int32)
        self.value = np.array(value, dtype=np.float32)
        return self

    def predict_proba(self, Xb):
        node = np.zeros(len(Xb), dtype=np.int32)
        active = np.arange(len(Xb))
        while active.size:
            nd = node[active]; f = self.feat[nd]
            keep = f >= 0
            active, nd, f = active[keep], nd[keep], f[keep]
            if not active.size:
                break
            node[active] = np.where(Xb[active, f] <= self.thr[nd], self.left[nd], self.right[nd])
        return self.value[node]

    @property
    def n_leaves(self):
        return int((self.feat < 0).sum())

    # plain-array (de)serialisation so worker processes never need to pickle the class itself
    def state(self):
        return dict(feat=self.feat, thr=self.thr, left=self.left, right=self.right, value=self.value)

    @classmethod
    def from_state(cls, s):
        t = cls(); t.feat, t.thr, t.left, t.right, t.value = s["feat"], s["thr"], s["left"], s["right"], s["value"]
        return t


def _grow_tree(Xb, y, seed, params, class_weight, bootstrap):
    """Grow one tree (runs in a worker process). Returns its arrays and its out-of-bag predictions."""
    rng = np.random.RandomState(seed)
    n = len(y)
    tree_seed = int(rng.randint(2**31 - 1))
    if not bootstrap:
        t = HistTree(seed=tree_seed, class_weight=class_weight, **params).fit(Xb, y)
        return t.state(), np.empty(0, np.int64), np.empty((0, K), np.float32)
    i = rng.randint(0, n, n)                               # bootstrap: n draws with replacement
    mask = np.ones(n, bool); mask[i] = False
    oob = np.flatnonzero(mask)                             # rows this tree never saw
    t = HistTree(seed=tree_seed, class_weight=class_weight, **params).fit(Xb[i], y[i])
    return t.state(), oob, t.predict_proba(Xb[oob])


class RandomForest:
    def __init__(self, n_trees=100, mtry=None, max_depth=None, min_leaf=1,
                 class_weight=None, bootstrap=True, seed=0, n_jobs=N_JOBS):
        self.n_trees, self.mtry, self.max_depth, self.min_leaf = n_trees, mtry, max_depth, min_leaf
        self.class_weight, self.bootstrap, self.seed, self.n_jobs = class_weight, bootstrap, seed, n_jobs

    def fit(self, Xb, y, checkpoints=()):
        seeds = np.random.RandomState(self.seed).randint(0, 2**31 - 1, self.n_trees)
        params = dict(max_depth=self.max_depth, min_leaf=self.min_leaf, mtry=self.mtry)
        res = Parallel(n_jobs=self.n_jobs)(
            delayed(_grow_tree)(Xb, y, int(s), params, self.class_weight, self.bootstrap) for s in seeds)
        self.trees = [HistTree.from_state(r[0]) for r in res]
        if self.bootstrap:
            self._oob(res, y, set(checkpoints))
        return self

    def _oob(self, res, y, checkpoints):
        """Accumulate each row's votes from the trees that did NOT see it; snapshot the score at checkpoints."""
        n = len(y); acc = np.zeros((n, K)); cnt = np.zeros(n)
        self.oob_curve, self.tree_oob_f1, frac = {}, [], []
        for t, (_, oob, pr) in enumerate(res, 1):
            acc[oob] += pr; cnt[oob] += 1; frac.append(len(oob) / n)
            self.tree_oob_f1.append(scores(y[oob], pr.argmax(1))[0]["f1_macro"])      # this tree alone, on its own OOB rows
            if t in checkpoints:
                seen = cnt > 0
                self.oob_curve[t] = scores(y[seen], acc[seen].argmax(1))[0]
        seen = cnt > 0
        self.oob_scores_, self.oob_f1_per_class_, self.oob_cm_ = scores(y[seen], acc[seen].argmax(1))
        self.oob_fraction_, self.oob_coverage_ = float(np.mean(frac)), float(seen.mean())

    def predict_proba(self, Xb, n_trees=None):
        ts = self.trees[:n_trees]
        return sum(t.predict_proba(Xb) for t in ts) / len(ts)

    def predict(self, Xb):
        return self.predict_proba(Xb).argmax(1)


def knn_votes(Ztr, ytr, Zte, k=9, chunk=1000, return_ties=False):
    """Exact brute-force k-NN. Returns the (n_test, K) matrix of neighbour votes per class.
    ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2; the ||b||^2 term is constant per test row, so it is dropped.
    return_ties=True also returns a flag per test row: are the k-th and (k+1)-th neighbours equidistant?
    (then the neighbour set, and hence the vote, depends on arbitrary tie-breaking)"""
    sq = (Ztr.astype(np.float32) ** 2).sum(1)
    votes = np.zeros((len(Zte), K), np.int32)
    ties = np.zeros(len(Zte), bool)
    for s in range(0, len(Zte), chunk):
        d = sq[None, :] - 2.0 * (Zte[s:s + chunk] @ Ztr.T)
        nn = np.argpartition(d, k - 1, axis=1)[:, :k]
        votes[s:s + chunk] = (ytr[nn][:, :, None] == np.arange(K)).sum(1)
        if return_ties:
            part = np.partition(d, [k - 1, k], axis=1)
            ties[s:s + chunk] = (part[:, k] - part[:, k - 1]) <= 1e-4
    return (votes, ties) if return_ties else votes

def knn_predict(votes, prior=None):
    """prior=None: plain majority vote. prior=class frequencies: divide votes by the prior (Assignment 2's correction)."""
    return (votes if prior is None else votes / prior).argmax(1)

def resubstitution_ceiling(B, y):
    """Best accuracy any deterministic classifier can reach on the rows (B, y): the majority label inside each group of identical binned rows."""
    g = pd.DataFrame({"h": pd.util.hash_pandas_object(pd.DataFrame(B), index=False).to_numpy(), "y": y})
    return g.groupby(["h", "y"]).size().groupby(level=0).max().sum() / len(y)

t0 = time.time()
tree_full = HistTree(max_depth=None, min_leaf=1, mtry=None, seed=0).fit(Btr, y_train)
train_acc = (tree_full.predict_proba(Btr).argmax(1) == y_train).mean()
ceil_bin = resubstitution_ceiling(Btr, y_train)
print(f"[1] unrestricted tree: {tree_full.n_leaves:,} leaves, fit {time.time()-t0:.1f}s")
print(f"    train accuracy {train_acc:.4f} vs resubstitution ceiling of the binned training rows {ceil_bin:.4f} -> "
      f"{'PASS (it memorises everything that can be memorised)' if abs(train_acc - ceil_bin) < 1e-9 else 'MISMATCH'}")

pr = tree_full.predict_proba(Bte)
print(f"[2] leaf distributions sum to 1: {np.allclose(pr.sum(1), 1, atol=1e-5)}")

rf_t = RandomForest(n_trees=20, mtry=6, seed=1).fit(Btr, y_train)
print(f"[3] OOB fraction per tree {rf_t.oob_fraction_:.3f} (theory e^-1 = {np.exp(-1):.3f}); "
      f"rows covered by at least one OOB tree: {rf_t.oob_coverage_:.3f}")

RUN_ORACLE_CHECK = not QUICK
if RUN_ORACLE_CHECK:
    try:
        from sklearn.tree import DecisionTreeClassifier as _SKTree          # test oracle ONLY
        from sklearn.neighbors import KNeighborsClassifier as _SKKnn        # test oracle ONLY
    except ImportError:
        print("scikit-learn not installed: oracle check skipped (not needed for the rest of the notebook)")
    else:
        rows = []
        # exact-threshold tree on the un-binned ordinal matrix, same depth limit as the Assignment 2 tree
        Mtr, Mte = tprep._matrix(X.iloc[tr_idx]), tprep._matrix(X.iloc[te_idx])
        sk = _SKTree(max_depth=15, criterion="gini", random_state=0).fit(Mtr, y_train)
        own = HistTree(max_depth=15, seed=0).fit(Btr, y_train)
        for name, p in [("own tree, binned (64 bins)", own.predict_proba(Bte).argmax(1)),
                        ("exact tree (reference)", sk.predict(Mte))]:
            m = scores(y_test, p)[0]; rows.append({"model": name, **m})
        # exact k-NN (k=9, uniform votes)
        v, ties = knn_votes(Ztr, y_train, Zte, k=9, return_ties=True)
        own_knn = knn_predict(v)
        sk_knn = _SKKnn(n_neighbors=9, n_jobs=-1).fit(Ztr, y_train).predict(Zte)
        for name, p in [("own k-NN", own_knn), ("k-NN (reference)", sk_knn)]:
            m = scores(y_test, p)[0]; rows.append({"model": name, **m})
        print(pd.DataFrame(rows).set_index("model").round(4).to_string())
        agree = own_knn == sk_knn
        print(f"\nk-NN prediction agreement with the reference: {agree.mean():.4f}")
        print(f"rows whose 9th and 10th neighbours are equidistant (neighbour set ambiguous): {ties.mean():.3%}")
        print(f"agreement on the remaining rows: {agree[~ties].mean():.4f};  on the ambiguous rows: {agree[ties].mean():.4f}")

MTRY_SQRT = int(np.sqrt(P_FEATURES))
print(f"p = {P_FEATURES} features; sqrt rule gives mtry = {MTRY_SQRT}; tuning on {len(y_train):,} training rows; jobs = {N_JOBS}")

def oob_row(rf, **extra):
    s = rf.oob_scores_
    return {**extra, "oob_acc": s["accuracy"], "oob_bal_acc": s["balanced_accuracy"], "oob_f1_macro": s["f1_macro"],
            "tree_oob_f1": float(np.mean(rf.tree_oob_f1))}

def pick(df_, simpler_first, ascending=True):
    """Best OOB macro F1; among settings within 0.002 of it (the tie rule), take the simplest."""
    tied = df_[df_["oob_f1_macro"] >= df_["oob_f1_macro"].max() - 0.002]
    return tied.sort_values(simpler_first, ascending=ascending).iloc[0]

t0 = time.time()
rf_q1 = RandomForest(n_trees=T_MAX, mtry=MTRY_SQRT, seed=11).fit(Btr, y_train, checkpoints=CHECKPOINTS)
print(f"fitted {T_MAX} trees in {time.time()-t0:.0f}s")
q1 = pd.DataFrame([{"trees": t, **{k: v for k, v in s.items()}} for t, s in rf_q1.oob_curve.items()]).set_index("trees")
single = float(np.mean(rf_q1.tree_oob_f1))
T_STAR = int(q1[q1["f1_macro"] >= q1["f1_macro"].max() - 0.002].index.min())
print(f"average single-tree OOB macro F1: {single:.4f}")
print(f"T* = {T_STAR}  (smallest checkpoint within 0.002 of the best, {q1['f1_macro'].max():.4f})")
q1.round(4)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(q1.index, q1["f1_macro"], "-o", ms=4, lw=2, color=FAMILY_COLOR["Random forest"], label="Forest (OOB)")
ax.axhline(single, color=FAMILY_COLOR["Decision tree"], ls="--", lw=1.6, label="Average single tree (OOB)")
ax.axvline(T_STAR, color=INK2, ls=":", lw=1)
ax.annotate(f"T* = {T_STAR}", (T_STAR, q1["f1_macro"].max()), xytext=(-6, -14), textcoords="offset points", ha="right", color=INK2, fontsize=8)
ax.set_xscale("log"); ax.minorticks_off()
tick = [t for t in [1, 2, 5, 10, 20, 50, 100, 200] if t <= T_MAX]
ax.set_xticks(tick); ax.set_xticklabels(tick)
ax.set_xlabel("Number of trees (log scale)"); ax.set_ylabel("OOB macro F1")
ax.set_title("Q1: how the OOB score grows with the number of trees")
ax.legend(loc="center right"); plt.tight_layout(); plt.show()

MTRY_GRID = sorted({1, 2, 4, MTRY_SQRT, 10, P_FEATURES // 2, P_FEATURES})
if QUICK: MTRY_GRID = [2, MTRY_SQRT, P_FEATURES]
q2_rows = []
for mt in MTRY_GRID:
    t0 = time.time()
    rf = RandomForest(n_trees=TUNE_T, mtry=mt, seed=21).fit(Btr, y_train)
    q2_rows.append(oob_row(rf, mtry=mt, fit_s=time.time() - t0, leaves=np.mean([t.n_leaves for t in rf.trees])))
    print(f"mtry={mt:3d}  OOB macro F1 {q2_rows[-1]['oob_f1_macro']:.4f}  single tree {q2_rows[-1]['tree_oob_f1']:.4f}  ({q2_rows[-1]['fit_s']:.0f}s)")
q2 = pd.DataFrame(q2_rows)
q2["ensemble_gain"] = q2["oob_f1_macro"] - q2["tree_oob_f1"]       # what averaging adds over the average single tree
MTRY_STAR = int(pick(q2, "mtry")["mtry"])
print(f"\nmtry* = {MTRY_STAR}")
q2.round(4)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(q2["mtry"], q2["oob_f1_macro"], "-o", ms=4, lw=2, color=FAMILY_COLOR["Random forest"], label="Forest (OOB)")
ax.plot(q2["mtry"], q2["tree_oob_f1"], "--o", ms=4, lw=1.6, color=FAMILY_COLOR["Decision tree"], label="Average single tree (OOB)")
ax.axvline(MTRY_STAR, color=INK2, ls=":", lw=1)
ax.set_xscale("log"); ax.set_xticks(q2["mtry"]); ax.set_xticklabels(q2["mtry"]); ax.minorticks_off()
ax.set_xlabel("mtry: features examined per split (mtry = p is plain bagging)"); ax.set_ylabel("OOB macro F1")
ax.set_title("Q2: strength of the individual tree against decorrelation of the ensemble")
ax.legend(loc="lower right"); plt.tight_layout(); plt.show()

DEPTH_GRID = [8, 16] if QUICK else [8, 12, 16, 24, None]
LEAF_GRID  = [1, 20] if QUICK else [1, 5, 20]
q3_rows = []
for md in DEPTH_GRID:
    for ml in LEAF_GRID:
        t0 = time.time()
        rf = RandomForest(n_trees=TUNE_T, mtry=MTRY_STAR, max_depth=md, min_leaf=ml, seed=31).fit(Btr, y_train)
        q3_rows.append(oob_row(rf, max_depth=(60 if md is None else md), depth_label=("none" if md is None else md),
                               min_leaf=ml, fit_s=time.time() - t0, leaves=np.mean([t.n_leaves for t in rf.trees])))
        print(f"depth={str(md):>4s} leaf={ml:3d}  OOB macro F1 {q3_rows[-1]['oob_f1_macro']:.4f}  ({q3_rows[-1]['fit_s']:.0f}s)")
q3 = pd.DataFrame(q3_rows)
# simplest = shallowest, then largest leaf
_t = pick(q3, ["max_depth", "min_leaf"], ascending=[True, False])
DEPTH_STAR = None if _t["max_depth"] == 60 else int(_t["max_depth"]); LEAF_STAR = int(_t["min_leaf"])
print(f"\nmax_depth* = {DEPTH_STAR}, min_leaf* = {LEAF_STAR}")

H = q3.pivot(index="depth_label", columns="min_leaf", values="oob_f1_macro")
H = H.loc[[("none" if d is None else d) for d in DEPTH_GRID]]
fig, ax = plt.subplots(figsize=(5.4, 3.8))
im = ax.imshow(H.values, cmap=SEQ, aspect="auto")
ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns); ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index)
ax.grid(False); annotate_heatmap(ax, H.values, fmt="{:.4f}", vmin=np.nanmin(H.values), vmax=np.nanmax(H.values), fs=8)
ax.set_xlabel("min_leaf (minimum rows per leaf)"); ax.set_ylabel("max_depth")
ax.set_title("Q3: OOB macro F1 (darker is better)"); plt.tight_layout(); plt.show()

GAMMAS = [0.0, 1.0] if QUICK else [0.0, 0.5, 1.0]
q4_rows, q4_pc = [], {}
for g_ in GAMMAS:
    t0 = time.time()
    rf = RandomForest(n_trees=TUNE_T, mtry=MTRY_STAR, max_depth=DEPTH_STAR, min_leaf=LEAF_STAR,
                      class_weight=(None if g_ == 0 else class_weights(y_train, g_)), seed=41).fit(Btr, y_train)
    q4_rows.append(oob_row(rf, gamma=g_, fit_s=time.time() - t0)); q4_pc[g_] = rf.oob_f1_per_class_
    print(f"gamma={g_:.1f}  acc {q4_rows[-1]['oob_acc']:.4f}  bal acc {q4_rows[-1]['oob_bal_acc']:.4f}  macro F1 {q4_rows[-1]['oob_f1_macro']:.4f}")
q4 = pd.DataFrame(q4_rows)
print("OOB F1 per class")
print(pd.DataFrame({f"gamma={g_:g}": q4_pc[g_] for g_ in GAMMAS}, index=NAMES).round(3).to_string())
_t4 = pick(q4, "gamma")
GAMMA_STAR = float(_t4["gamma"])
print(f"\ngamma* = {GAMMA_STAR}")
q4.round(4)

sup_tr = np.bincount(y_train, minlength=K)
order = np.argsort(-sup_tr)
ramp = {0.0: "#9ec5f4", 0.5: "#3987e5", 1.0: "#184f95"}          # ordered single-hue ramp: more weight = darker
fig, ax = plt.subplots(figsize=(9, 3.8))
w_ = 0.8 / len(GAMMAS)
for i, g_ in enumerate(GAMMAS):
    ax.bar(np.arange(K) + (i - (len(GAMMAS) - 1) / 2) * w_, q4_pc[g_][order], w_ * 0.92, color=ramp.get(g_, "#3987e5"),
           label=f"gamma = {g_:g}" + ("  (no weights)" if g_ == 0 else "  (balanced)" if g_ == 1 else ""))
ax.set_xticks(range(K)); ax.set_xticklabels([f"{NAMES[c]}\n(n={sup_tr[c]:,})" for c in order], rotation=30, ha="right", fontsize=7.5)
ax.set_ylabel("OOB F1"); ax.set_title("Q4: per-class OOB F1 by class-weight exponent (classes ordered by training frequency)")
ax.grid(axis="x", visible=False); ax.legend(); plt.tight_layout(); plt.show()

GAMMA_FINAL = GAMMA_STAR if GAMMA_STAR > 0 else 1.0      # the "weighted" variant must differ from the unweighted one
FOREST = dict(n_trees=T_STAR, mtry=MTRY_STAR, max_depth=DEPTH_STAR, min_leaf=LEAF_STAR)
print("Final forest:", FOREST)
print(f"Unweighted variant: gamma = 0     Weighted variant: gamma = {GAMMA_FINAL:g}")

TREE_A2 = dict(max_depth=15, min_leaf=1)
KNN_K   = 9
os.makedirs(CACHE_DIR, exist_ok=True)

def run_once(seed):
    """Split with `seed`, fit all six models on the training part, evaluate on the test part. Cached per seed."""
    cfg = dict(forest=FOREST, gamma=GAMMA_FINAL, subsample=SUBSAMPLE, tree=TREE_A2, k=KNN_K)
    path = os.path.join(CACHE_DIR, f"run_{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if cached["cfg"] == cfg:
            return cached["out"]
    tr, te = stratified_split(y_all, 0.30, seed)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y_all[tr], y_all[te]
    tp = TreePrep().fit(Xtr);  Bt, Be = tp.transform(Xtr), tp.transform(Xte)
    kp = KnnPrep().fit(Xtr);   Zt, Ze = kp.transform(Xtr), kp.transform(Xte)       # correlation pruning refitted on THIS train part
    out = {}

    def record(name, pred, fit_s, pred_s):
        m, f1, cm = scores(yte, pred)
        out[name] = dict(metrics=m, f1=f1, cm=cm, fit_s=fit_s, predict_s=pred_s)

    for name, cw in [("Tree (baseline)", None), ("Tree (balanced)", class_weights(ytr, 1.0))]:
        t0 = time.time(); t = HistTree(class_weight=cw, seed=seed, **TREE_A2).fit(Bt, ytr); fs = time.time() - t0
        t0 = time.time(); p = t.predict_proba(Be).argmax(1);                                ps = time.time() - t0
        record(name, p, fs, ps)

    t0 = time.time(); votes = knn_votes(Zt, ytr, Ze, KNN_K); ps = time.time() - t0          # both k-NN variants share the same neighbours
    prior = np.bincount(ytr, minlength=K) / len(ytr)
    record("k-NN (baseline)", knn_predict(votes), 0.0, ps)
    record("k-NN (prior-corrected)", knn_predict(votes, prior), 0.0, ps)

    for name, g_ in [("RF (unweighted)", 0.0), ("RF (weighted)", GAMMA_FINAL)]:
        t0 = time.time()
        rf = RandomForest(class_weight=(None if g_ == 0 else class_weights(ytr, g_)),
                          seed=seed * 100 + int(g_ * 10), **FOREST).fit(Bt, ytr)
        fs = time.time() - t0
        t0 = time.time(); p = rf.predict(Be); ps = time.time() - t0
        record(name, p, fs, ps)

    with open(path, "wb") as f:
        pickle.dump(dict(cfg=cfg, out=out), f)
    return out

t0 = time.time()
single_run = run_once(RANDOM_STATE)
print(f"seed {RANDOM_STATE}: all six models fitted and evaluated in {time.time()-t0:.0f}s")

def report_from_cm(cm):
    tp, sup, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    prec = tp / np.maximum(pred, 1); rec = tp / np.maximum(sup, 1)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    return pd.DataFrame({"precision": prec, "recall": rec, "f1": f1, "support": sup.astype(int)}, index=NAMES)

UNTREATED = [m for m in MODELS if not TREATED[m]]
pd.DataFrame({m: single_run[m]["metrics"] for m in UNTREATED}).T.round(4)

print("Random forest (unweighted), classification report on the test partition")
report_from_cm(single_run["RF (unweighted)"]["cm"]).round(3)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, m in zip(axes, UNTREATED):
    cm = single_run[m]["cm"]; cmn = cm / cm.sum(1, keepdims=True)         # diagonal is per-class recall
    ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1); ax.grid(False)
    annotate_heatmap(ax, cmn, vmax=1.0, min_show=0.005)
    ax.set_xticks(range(K)); ax.set_xticklabels(NAMES, rotation=50, ha="right", fontsize=7.5)
    ax.set_yticks(range(K)); ax.set_yticklabels(NAMES, fontsize=7.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{m} (row-normalised)")
plt.tight_layout(); plt.show()

sup_te = np.bincount(y_test, minlength=K)
order_te = np.argsort(-sup_te)

def grouped_class_bars(ax, tables, models, order, sd=None):
    """Grouped bars: classes on the x axis, one bar per model. Colour = family, hatch = imbalance treatment."""
    n = len(models); w = 0.84 / n
    for i, m in enumerate(models):
        ax.bar(np.arange(K) + (i - (n - 1) / 2) * w, tables[m][order], w * 0.94,
               yerr=None if sd is None else sd[m][order], capsize=1.5, error_kw=dict(lw=0.7, ecolor="#555"),
               color=COL[m], hatch=HATCH[m], edgecolor="white", linewidth=0, label=m)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"{NAMES[c]}\n(n={sup_te[c]:,})" for c in order], rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("F1 score"); ax.set_ylim(0, 1.02); ax.grid(axis="x", visible=False)

fig, ax = plt.subplots(figsize=(11, 4.2))
grouped_class_bars(ax, {m: single_run[m]["f1"] for m in UNTREATED}, UNTREATED, order_te)
ax.set_title("Per-class F1 on the seed-42 split, classes ordered by test-set frequency"); ax.legend(ncol=3)
plt.tight_layout(); plt.show()

pd.DataFrame({m: single_run[m]["metrics"] for m in MODELS}).T.round(4)

runs = {}
t_all = time.time()
for r in range(N_RUNS):
    seed = RANDOM_STATE + r
    t0 = time.time()
    runs[seed] = run_once(seed)
    print(f"run {r + 1:2d}/{N_RUNS} (seed {seed}) {time.time() - t0:5.0f}s | " +
          " | ".join(f"{m.split(' (')[0]}{' T' if TREATED[m] else ''} {runs[seed][m]['metrics']['f1_macro']:.3f}" for m in MODELS))
print(f"\nTotal {time.time() - t_all:.0f}s")

rows = []
for seed, out in runs.items():
    for m in MODELS:
        rows.append({"run": seed - RANDOM_STATE, "model": m, **out[m]["metrics"],
                     "fit_s": out[m]["fit_s"], "predict_s": out[m]["predict_s"]})
results = pd.DataFrame(rows)
per_class_runs = {m: np.array([runs[s][m]["f1"] for s in runs]) for m in MODELS}                    
cm_runs = {m: np.array([runs[s][m]["cm"] / runs[s][m]["cm"].sum(1, keepdims=True) for s in runs]) for m in MODELS}
print("Collected", len(results), "model-run results")

METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
METRIC_LABELS = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                 ("f1_macro", "Macro F1"), ("f1_weighted", "Weighted F1")]

readable = pd.DataFrame(
    {lbl: [f"{results[results.model == m][k].mean():.4f} ± {results[results.model == m][k].std(ddof=1):.4f}" for m in MODELS]
     for k, lbl in METRIC_LABELS}, index=MODELS)
print(f"mean ± sd over {N_RUNS} runs")
readable

timing = results.groupby("model")[["fit_s", "predict_s"]].agg(["mean", "std"]).reindex(MODELS).round(3)
print(timing.to_string())

def compare(metric, a, b):
    piv = results.pivot(index="run", columns="model", values=metric)
    d = piv[a] - piv[b]
    return {"metric": metric, "comparison": f"{a}  vs  {b}", "mean diff": round(d.mean(), 4), "sd diff": round(d.std(ddof=1), 4),
            "min diff": round(d.min(), 4), "max diff": round(d.max(), 4), f"wins (of {len(d)})": int((d > 0).sum())}

PAIRS = [("RF (unweighted)", "Tree (baseline)"),          # the controlled comparison: ensembling vs one tree
         ("RF (weighted)",   "Tree (balanced)"),
         ("RF (weighted)",   "RF (unweighted)"),          # effect of class weighting inside the forest
         ("RF (unweighted)", "k-NN (baseline)"),
         ("RF (weighted)",   "k-NN (prior-corrected)")]
pd.DataFrame([compare(k, a, b) for k in ["f1_macro", "balanced_accuracy", "accuracy"] for a, b in PAIRS])

piv = results.pivot(index="run", columns="model", values="f1_macro")
for m in MODELS:
    print(f"{m:24s} macro F1 range [{piv[m].min():.4f}, {piv[m].max():.4f}]")
print("\nRanges overlap (RF unweighted vs Tree baseline):", not (piv['RF (unweighted)'].min() > piv['Tree (baseline)'].max()))
print("Ranges overlap (RF weighted vs Tree balanced)  :", not (piv['RF (weighted)'].min() > piv['Tree (balanced)'].max()))

supp_rows = []
for seed in runs:
    tr, te = stratified_split(y_all, 0.30, seed)
    Xtr_, Xte_, ytr_, yte_ = X.iloc[tr], X.iloc[te], y_all[tr], y_all[te]
    tp_ = TreePrep().fit(Xtr_)
    t_ = HistTree(class_weight=class_weights(ytr_, GAMMA_FINAL), seed=seed, **TREE_A2).fit(tp_.transform(Xtr_), ytr_)
    m_, f1_, cm_ = scores(yte_, t_.predict_proba(tp_.transform(Xte_)).argmax(1))
    supp_rows.append({"run": seed - RANDOM_STATE, **m_})
supp = pd.DataFrame(supp_rows).set_index("run")
GL = f"{GAMMA_FINAL:g}"
print(f"Tree weighted with gamma = {GL}, mean ± sd over {len(supp)} runs")
print(pd.Series({lbl: f"{supp[k].mean():.4f} ± {supp[k].std(ddof=1):.4f}" for k, lbl in METRIC_LABELS}).to_string())
rows_ = []
for k in ["f1_macro", "balanced_accuracy", "accuracy"]:
    d = results.pivot(index="run", columns="model", values=k)["RF (weighted)"] - supp[k]
    rows_.append({"metric": k, "comparison": f"RF (weighted)  vs  Tree (gamma={GL})", "mean diff": round(d.mean(), 4),
                  "sd diff": round(d.std(ddof=1), 4), "min diff": round(d.min(), 4), "max diff": round(d.max(), 4),
                  f"wins (of {len(d)})": int((d > 0).sum())})
pd.DataFrame(rows_)

G1_NAME = "RF (gamma=1)"
g1_runs = {}
for seed in runs:
    path = os.path.join(CACHE_DIR, f"g1_{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            g1_runs[seed] = pickle.load(f)
        continue
    tr, te = stratified_split(y_all, 0.30, seed)
    Xtr_, Xte_, ytr_, yte_ = X.iloc[tr], X.iloc[te], y_all[tr], y_all[te]
    tp_ = TreePrep().fit(Xtr_); Bt_, Be_ = tp_.transform(Xtr_), tp_.transform(Xte_)
    t0 = time.time()
    rf_ = RandomForest(class_weight=class_weights(ytr_, 1.0), seed=seed * 100 + 10, **FOREST).fit(Bt_, ytr_)
    fs_ = time.time() - t0
    m_, f1_, cm_ = scores(yte_, rf_.predict(Be_))
    g1_runs[seed] = dict(metrics=m_, f1=f1_, cm=cm_, fit_s=fs_, oob=rf_.oob_scores_)
    with open(path, "wb") as f:
        pickle.dump(g1_runs[seed], f)

g1 = pd.DataFrame([{"run": s - RANDOM_STATE, **g1_runs[s]["metrics"]} for s in g1_runs]).set_index("run")
print(f"Forest with gamma = 1, mean ± sd over {len(g1)} runs   (mean OOB macro F1 {np.mean([g1_runs[s]['oob']['f1_macro'] for s in g1_runs]):.4f})")
print(pd.Series({lbl: f"{g1[k].mean():.4f} ± {g1[k].std(ddof=1):.4f}" for k, lbl in METRIC_LABELS}).to_string())

rows_ = []
for other in ["RF (weighted)", "RF (unweighted)", "Tree (balanced)"]:
    for k in ["f1_macro", "balanced_accuracy", "accuracy"]:
        d = g1[k] - results.pivot(index="run", columns="model", values=k)[other]
        rows_.append({"metric": k, "comparison": f"RF (gamma=1)  vs  {other}", "mean diff": round(d.mean(), 4),
                      "sd diff": round(d.std(ddof=1), 4), "min diff": round(d.min(), 4), "max diff": round(d.max(), 4),
                      f"wins (of {len(d)})": int((d > 0).sum())})
print()
print(pd.DataFrame(rows_).to_string(index=False))

pc_g1 = np.array([g1_runs[s]["f1"] for s in g1_runs]).mean(0)
print("\nMean per-class F1 on the test partitions")
print(pd.DataFrame({"RF (unweighted)": pc_mean["RF (unweighted)"], "RF (weighted, gamma=0.5)": pc_mean["RF (weighted)"],
                    "RF (gamma=1)": pc_g1}, index=NAMES).loc[[NAMES[c] for c in order_te]].round(3).to_string())

fig, ax = plt.subplots(figsize=(11, 4.4))
n = len(MODELS); w = 0.84 / n; x = np.arange(len(METRICS))
for i, m in enumerate(MODELS):
    sub = results[results.model == m]
    mu = [sub[k].mean() for k in METRICS]; sd = [sub[k].std(ddof=1) for k in METRICS]
    ax.bar(x + (i - (n - 1) / 2) * w, mu, w * 0.94, yerr=sd, capsize=2, error_kw=dict(lw=0.8, ecolor="#444"),
           color=COL[m], hatch=HATCH[m], edgecolor="white", linewidth=0, label=m)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in METRIC_LABELS]); ax.set_ylim(0, 1.16); ax.set_ylabel("Score")
ax.grid(axis="x", visible=False); ax.legend(ncol=6, loc="upper center", fontsize=7.5)
ax.set_title(f"Performance over {N_RUNS} independent runs (mean ± 1 s.d.; hatched = class-balanced)")
plt.tight_layout(); plt.show()

pc_mean = {m: per_class_runs[m].mean(0) for m in MODELS}
pc_sd   = {m: per_class_runs[m].std(0, ddof=1) if len(per_class_runs[m]) > 1 else np.zeros(K) for m in MODELS}
fig, ax = plt.subplots(figsize=(12, 4.6))
grouped_class_bars(ax, pc_mean, MODELS, order_te, sd=pc_sd)
ax.set_title(f"Per-class F1, mean ± s.d. over {N_RUNS} runs (classes ordered by test-set frequency; hatched = class-balanced)")
ax.legend(ncol=3); plt.tight_layout(); plt.show()

pd.DataFrame({m: pc_mean[m] for m in MODELS}, index=NAMES).loc[[NAMES[c] for c in order_te]].round(3)

fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
for ax, m in zip(axes.ravel(), MODELS):
    cm = cm_runs[m].mean(0)
    ax.imshow(cm, cmap=SEQ, vmin=0, vmax=1); ax.grid(False); annotate_heatmap(ax, cm, vmax=1.0, min_show=0.005)
    ax.set_xticks(range(K)); ax.set_xticklabels(NAMES, rotation=50, ha="right", fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels(NAMES, fontsize=7); ax.set_title(m)
plt.suptitle(f"Row-normalised confusion matrices averaged over {N_RUNS} runs (diagonal = recall)", x=0.01, ha="left", fontsize=11)
plt.tight_layout(); plt.show()

def kde_curve(v, pad=0.25, n=400):
    v = np.asarray(v, float); lo, hi = v.min(), v.max(); span = max(hi - lo, 1e-6)
    grid = np.linspace(lo - pad * span - 0.005, hi + pad * span + 0.005, n)
    return grid, gaussian_kde(v)(grid)

def draw_density(ax, metric, models, label=None):
    for m in models:
        v = results.loc[results.model == m, metric].to_numpy()
        if len(v) < 3 or np.ptp(v) < 1e-9:
            continue
        grid, dens = kde_curve(v)
        ax.plot(grid, dens, LSTYLE[m], color=COL[m], lw=1.8, label=m)
        ax.fill_between(grid, dens, color=COL[m], alpha=0.10)
        ax.plot(v, np.full_like(v, -0.02 * dens.max()), "|", color=COL[m], ms=6, mew=1.0, alpha=0.7)
    ax.set_xlabel(label or metric); ax.set_ylabel("Density"); ax.margins(y=0.12)

if N_RUNS >= 3:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (k, lbl) in zip(axes.ravel(), METRIC_LABELS):
        draw_density(ax, k, MODELS, lbl); ax.set_title(lbl)
    axes[0, 0].legend(fontsize=7.5, loc="upper left")
    plt.suptitle(f"Metric distributions over {N_RUNS} runs (dashed = untreated, solid = class-balanced)", x=0.01, ha="left", fontsize=11)
    plt.tight_layout(); plt.show()
else:
    print("Density plots need at least 3 runs.")

if N_RUNS >= 3:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (k, lbl) in zip(axes, [("f1_macro", "Macro F1"), ("balanced_accuracy", "Balanced accuracy")]):
        draw_density(ax, k, ["Tree (baseline)", "Tree (balanced)", "RF (unweighted)", "RF (weighted)"], lbl); ax.set_title(lbl)
    axes[0].legend(fontsize=8)
    plt.suptitle("The controlled comparison: one tree against a forest of the same trees", x=0.01, ha="left", fontsize=11)
    plt.tight_layout(); plt.show()

if N_RUNS >= 3:
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for row, (fam, members) in enumerate([("Tree", ["Tree (baseline)", "Tree (balanced)"]),
                                          ("k-NN", ["k-NN (baseline)", "k-NN (prior-corrected)"]),
                                          ("Random forest", ["RF (unweighted)", "RF (weighted)"])]):
        for col, (k, lbl) in enumerate([("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy")]):
            draw_density(axes[row, col], k, members, lbl); axes[row, col].set_title(f"{fam}: {lbl}")
            for m in members:
                axes[row, col].axvline(results.loc[results.model == m, k].mean(), color=COL[m], ls=":", lw=1)
            axes[row, col].legend(fontsize=7.5)
    plt.suptitle("The imbalance trade-off per family: accuracy falls (left) as balanced accuracy rises (right)", x=0.01, ha="left", fontsize=11)
    plt.tight_layout(); plt.show()

print(f"n = {N_RUNS} runs per model\n")
for k, lbl in METRIC_LABELS:
    p_ = results.pivot(index="run", columns="model", values=k)
    print(f"--- {lbl} ---")
    for m in MODELS:
        v = p_[m]; print(f"  {m:24s} mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  range [{v.min():.4f}, {v.max():.4f}]")
    print()