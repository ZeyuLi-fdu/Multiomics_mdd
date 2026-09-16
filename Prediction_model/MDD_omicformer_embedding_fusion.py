"""MDD multi-omics risk prediction via OmicFormer embedding-concatenation fusion."""
import os, sys, random, warnings, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lifelines.utils import concordance_index

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'models'))
from model_factory import OmicFormerBranch, CovarMLP

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--dim', type=int, required=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--out_root', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
args = parser.parse_args()

DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
OUT_ROOT = args.out_root or os.path.join(HERE, 'results_omicformer_embedding_fusion')
OUT_DIR = os.path.join(OUT_ROOT, f'lr{args.lr:g}_dim{args.dim}')
DATA_DIR = args.data_dir or os.path.join(HERE, 'data', 'Feature_FDRSig')
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DIR = os.path.join(DATA_DIR, 'Test')

SEED = 2024
EPOCHS = 20
PATIENCE = 7
BATCH_SIZE = 256
AUX_WEIGHT = 0.3
KERNEL_SIZES = (3, 5, 8, 13)
COV_CAT_COLS = ['sex', 'smoking', 'drinking', 'education']
N_BOOTSTRAP = 1000
DIM, DEPTH, HEADS = args.dim, 3, 4
LR, WD, DROPOUT = args.lr, 1e-5, 0.1
FOCAL_GAMMA = 2.0
EMBED_DIM = 32
FUSION_HIDDEN = 64
OMIC_LIST = ['covar', 'prs', 'cli', 'met', 'pro']

os.makedirs(OUT_DIR, exist_ok=True)
print(f'Device: {DEVICE} | lr={LR} | dim={DIM} | omics={OMIC_LIST}', flush=True)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_tsv(d, fname):
    return pd.read_csv(os.path.join(d, fname), sep='\t')


def preprocess_covariate(cov_df, cat_cols):
    actual_cat = [c for c in cat_cols if c in cov_df.columns]
    cont_cols = [c for c in cov_df.columns if c not in actual_cat + ['eid']]
    df_cont = cov_df[['eid'] + cont_cols].set_index('eid').fillna(cov_df[cont_cols].median())
    parts = [df_cont]
    for c in actual_cat:
        dummies = pd.get_dummies(cov_df[c].fillna(-1).astype(int), prefix=c, drop_first=True)
        dummies.index = cov_df['eid'].values
        parts.append(dummies)
    return pd.concat(parts, axis=1).fillna(0).astype(np.float32)


def preprocess_omics(df):
    feat_cols = [c for c in df.columns if c != 'eid']
    df_num = df[feat_cols].apply(pd.to_numeric, errors='coerce')
    return df[['eid']].join(df_num).set_index('eid').fillna(df_num.median()).astype(np.float32)


def filter_incident(label_df):
    label_df = label_df[~((label_df['status'] == 1) & (label_df['date_baseline'] <= 0))]
    label_df = label_df[~((label_df['status'] == 0) & (label_df['date_baseline'] < 0))]
    return label_df.reset_index(drop=True)


class PlattCalibrator:
    def __init__(self):
        self.a, self.b = 1.0, 0.0

    def fit(self, logits, labels):
        lr = LogisticRegression(C=1000.0, solver='lbfgs', max_iter=1000)
        lr.fit(np.array(logits).reshape(-1, 1), labels)
        self.a, self.b = lr.coef_[0, 0], lr.intercept_[0]
        return self

    def calibrate(self, logits):
        return 1.0 / (1.0 + np.exp(-(self.a * np.array(logits) + self.b)))


def compute_ece(labels, probs, n_bins=10):
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= i / n_bins) & (probs < (i + 1) / n_bins)
        if mask.sum() > 0:
            ece += (mask.sum() / len(probs)) * abs(labels[mask].mean() - probs[mask].mean())
    return ece


def bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap=N_BOOTSTRAP, alpha=0.05):
    n = len(y_true)
    rng = np.random.RandomState(42)
    metrics = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            metrics.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    metrics = np.array(metrics)
    return (np.mean(metrics), np.percentile(metrics, 100 * alpha / 2),
            np.percentile(metrics, 100 * (1 - alpha / 2)))


def bootstrap_ci_cindex(years, probs, labels, n_bootstrap=N_BOOTSTRAP):
    n = len(labels)
    rng = np.random.RandomState(42)
    metrics = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            metrics.append(concordance_index(years[idx], -probs[idx], labels[idx].astype(bool)))
        except Exception:
            pass
    metrics = np.array(metrics)
    return (np.mean(metrics), np.percentile(metrics, 2.5), np.percentile(metrics, 97.5))


class LateFusionOmicEmbedding(nn.Module):
    """Each omic branch -> dim features -> 32-dim embedding -> concat -> FC+ReLU -> score."""

    def __init__(self, dim, dim_out, depth, heads, omics, input_dims,
                 attn_dropout=0., ff_dropout=0., dropout=0.,
                 embed_dim=EMBED_DIM, fusion_hidden=FUSION_HIDDEN, kernel_sizes=KERNEL_SIZES):
        super().__init__()
        self.omics = list(omics)
        self.branches = nn.ModuleDict()
        for o in self.omics:
            # 'covar' uses a lightweight MLP; all other omics (prs / cli /
            # met / pro) uniformly go through OmicFormerBranch, consistent
            # with the paper's description that "genomic/proteomic/
            # metabolomic/clinical features [are] processed through
            # modality-specific OmicFormer branches".
            if o == 'covar':
                self.branches[o] = CovarMLP(input_dim=input_dims[o], dim=dim, dim_out=dim_out, dropout=dropout)
            else:
                self.branches[o] = OmicFormerBranch(
                    num_features=input_dims[o], dim=dim, dim_out=dim_out, depth=depth, heads=heads,
                    attn_dropout=attn_dropout, ff_dropout=ff_dropout, kernel_sizes=kernel_sizes)
        self.embed_proj = nn.ModuleDict({o: nn.Linear(dim, embed_dim) for o in self.omics})
        self.fusion = nn.Sequential(
            nn.Linear(len(self.omics) * embed_dim, fusion_hidden),
            nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden, dim_out))

    def forward(self, x_dict):
        embeds, aux_logits = [], {}
        for o in self.omics:
            logit_o, feat, _ = self.branches[o](x_dict[o])
            aux_logits[o] = logit_o
            embeds.append(self.embed_proj[o](feat))
        concat = torch.cat(embeds, dim=-1)
        return self.fusion(concat), concat, aux_logits, None

    def fit_channels(self, x_dict_np, y_np):
        for o in self.omics:
            branch = self.branches[o]
            if isinstance(branch, OmicFormerBranch):
                branch.fit_channels(x_dict_np[o], y_np)


def predict(model, loader, omics):
    model.eval()
    logits, probs, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            *feats, y = batch
            x = {o: f.to(DEVICE) for o, f in zip(omics, feats)}
            lg = model(x)[0].squeeze(1)
            logits.append(lg.cpu().numpy())
            probs.append(torch.sigmoid(lg).cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(logits), np.concatenate(probs), np.concatenate(labels)


def train_fold(tr_tensors, y_tr, va_tensors, y_va, omics, input_dims, fold_id, out_dir, pos_weight,
               it_tensors=None, y_it=None, te_tensors=None, y_te=None):
    tr_loader = DataLoader(TensorDataset(*tr_tensors, y_tr), BATCH_SIZE, shuffle=True, pin_memory=True)
    va_loader = DataLoader(TensorDataset(*va_tensors, y_va), BATCH_SIZE, shuffle=False, pin_memory=True)

    model = LateFusionOmicEmbedding(
        dim=DIM, dim_out=1, depth=DEPTH, heads=HEADS, omics=omics, input_dims=input_dims,
        attn_dropout=DROPOUT, ff_dropout=DROPOUT, dropout=DROPOUT,
        embed_dim=EMBED_DIM, fusion_hidden=FUSION_HIDDEN, kernel_sizes=KERNEL_SIZES).to(DEVICE)
    model.fit_channels({o: t.numpy() for o, t in zip(omics, tr_tensors)}, y_tr.numpy())

    pw = torch.tensor([pos_weight], device=DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=8)
    best_auc, best_state, wait = -1e9, None, 0

    for _ in range(EPOCHS):
        model.train()
        for batch in tr_loader:
            *feats, y = batch
            x = {o: f.to(DEVICE) for o, f in zip(omics, feats)}
            y = y.to(DEVICE)
            opt.zero_grad()
            logits, _, aux, _ = model(x)
            logits = logits.squeeze(1)
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction='none')
            loss = ((1 - torch.exp(-bce)) ** FOCAL_GAMMA * bce).mean()
            if aux and AUX_WEIGHT > 0 and len(omics) > 1:
                aux_loss = sum(F.binary_cross_entropy_with_logits(
                    aux[o].squeeze(1), y, pos_weight=pw) for o in omics) / len(omics)
                loss = loss + AUX_WEIGHT * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        _, va_p, va_l = predict(model, va_loader, omics)
        va_auc = roc_auc_score(va_l, va_p) if len(np.unique(va_l)) > 1 else 0.5
        sched.step(va_auc)
        if va_auc > best_auc:
            best_auc, best_state, wait = va_auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    va_logits, va_probs, va_labels = predict(model, va_loader, omics)
    cal = PlattCalibrator().fit(va_logits, va_labels)

    result = {'fold': fold_id, 'inner_val_logits': va_logits, 'inner_val_probs_raw': va_probs,
              'inner_val_probs_cal': cal.calibrate(va_logits), 'inner_val_labels': va_labels}
    for key, tensors, labels in (('val', it_tensors, y_it), ('test', te_tensors, y_te)):
        if tensors is not None:
            loader = DataLoader(TensorDataset(*tensors, labels), BATCH_SIZE, shuffle=False)
            lg, pr, lb = predict(model, loader, omics)
            result[f'{key}_logits'], result[f'{key}_probs_raw'] = lg, pr
            result[f'{key}_probs_cal'], result[f'{key}_labels'] = cal.calibrate(lg), lb
        else:
            for k in ('logits', 'probs_raw', 'probs_cal', 'labels'):
                result[f'{key}_{k}'] = None

    torch.save(best_state, os.path.join(out_dir, f'best_model_fold{fold_id}.pt'))
    return result


# ---- load data (TSV files under data/Feature_FDRSig/{Training,Test}/) ----
# each split has: Label.txt, Covariate_v2.txt, Clinical.txt, Metabolome.txt, Proteome.txt, PRS.txt
print('Loading data...', flush=True)
label_df = filter_incident(read_tsv(TRAIN_DIR, 'Label.txt'))
test_label_df = filter_incident(read_tsv(TEST_DIR, 'Label.txt'))

TRAIN_OMIC = {
    'covar': preprocess_covariate(read_tsv(TRAIN_DIR, 'Covariate_v2.txt'), COV_CAT_COLS),
    'prs': preprocess_omics(read_tsv(TRAIN_DIR, 'PRS.txt')),
    'cli': preprocess_omics(read_tsv(TRAIN_DIR, 'Clinical.txt')),
    'met': preprocess_omics(read_tsv(TRAIN_DIR, 'Metabolome.txt')),
    'pro': preprocess_omics(read_tsv(TRAIN_DIR, 'Proteome.txt')),
}
TEST_OMIC = {
    'covar': preprocess_covariate(read_tsv(TEST_DIR, 'Covariate_v2.txt'), COV_CAT_COLS),
    'prs': preprocess_omics(read_tsv(TEST_DIR, 'PRS.txt')),
    'cli': preprocess_omics(read_tsv(TEST_DIR, 'Clinical.txt')),
    'met': preprocess_omics(read_tsv(TEST_DIR, 'Metabolome.txt')),
    'pro': preprocess_omics(read_tsv(TEST_DIR, 'Proteome.txt')),
}

train_common = sorted(set.intersection(set(label_df['eid']),
                                       *[set(TRAIN_OMIC[k].index) for k in OMIC_LIST]))
train_label_sub = label_df[label_df['eid'].isin(train_common)].reset_index(drop=True)
train_eids = train_label_sub['eid'].values
train_y = train_label_sub['status'].values.astype(int)
train_years = train_label_sub['date_baseline'].values

test_common = sorted(set.intersection(set(test_label_df['eid']),
                                      *[set(TEST_OMIC[k].index) for k in OMIC_LIST]))
test_label_sub = test_label_df[test_label_df['eid'].isin(test_common)].reset_index(drop=True)
test_eids = test_label_sub['eid'].values
test_y = test_label_sub['status'].values.astype(int)
test_years = test_label_sub['date_baseline'].values

print(f'Train N={len(train_common)} pos={train_y.sum()}, Test N={len(test_common)} pos={test_y.sum()}', flush=True)

omic_data_train = {o: TRAIN_OMIC[o].loc[[e for e in train_eids if e in TRAIN_OMIC[o].index]].values.astype(np.float32) for o in OMIC_LIST}
omic_data_test = {o: TEST_OMIC[o].loc[[e for e in test_eids if e in TEST_OMIC[o].index]].values.astype(np.float32) for o in OMIC_LIST}

data_train = {o: omic_data_train[o] for o in OMIC_LIST}
data_test = {o: omic_data_test[o] for o in OMIC_LIST}
input_dims = {o: data_train[o].shape[1] for o in OMIC_LIST}
print(f'Feature dims: {input_dims}', flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_results, itest_indices = [], []
for fold_id, (tr_idx, itest_idx) in enumerate(skf.split(train_eids, train_y)):
    print(f'\n===== Fold {fold_id} | train pos={train_y[tr_idx].sum()} =====', flush=True)
    seed_everything(SEED + fold_id)
    itest_indices.append(itest_idx)
    inner_tr_idx, inner_val_idx = train_test_split(
        tr_idx, test_size=0.2, stratify=train_y[tr_idx], random_state=SEED + fold_id)

    tr_l, va_l, it_l, te_l = [], [], [], []
    for o in OMIC_LIST:
        s = StandardScaler().fit(data_train[o][inner_tr_idx])
        tr_l.append(torch.tensor(s.transform(data_train[o][inner_tr_idx]), dtype=torch.float32))
        va_l.append(torch.tensor(s.transform(data_train[o][inner_val_idx]), dtype=torch.float32))
        it_l.append(torch.tensor(s.transform(data_train[o][itest_idx]), dtype=torch.float32))
        te_l.append(torch.tensor(s.transform(data_test[o]), dtype=torch.float32))
    yt = torch.tensor(train_y[inner_tr_idx].astype(np.float32))
    yva = torch.tensor(train_y[inner_val_idx].astype(np.float32))
    yit = torch.tensor(train_y[itest_idx].astype(np.float32))
    ye = torch.tensor(test_y.astype(np.float32))
    pw = (yt == 0).sum().item() / max(yt.sum().item(), 1)

    fold_results.append(train_fold(tr_l, yt, va_l, yva, OMIC_LIST, input_dims,
                                   fold_id, OUT_DIR, pw, it_l, yit, te_l, ye))
    print(f'  Fold {fold_id} done', flush=True)

np.savez(os.path.join(OUT_DIR, 'fold_predictions.npz'),
         val_logits=np.array([r['val_logits'] for r in fold_results], dtype=object),
         val_probs_raw=np.array([r['val_probs_raw'] for r in fold_results], dtype=object),
         val_probs_cal=np.array([r['val_probs_cal'] for r in fold_results], dtype=object),
         val_labels=np.array([r['val_labels'] for r in fold_results], dtype=object),
         val_indices=np.array(itest_indices, dtype=object),
         inner_val_logits=np.array([r['inner_val_logits'] for r in fold_results], dtype=object),
         inner_val_probs_raw=np.array([r['inner_val_probs_raw'] for r in fold_results], dtype=object),
         inner_val_probs_cal=np.array([r['inner_val_probs_cal'] for r in fold_results], dtype=object),
         inner_val_labels=np.array([r['inner_val_labels'] for r in fold_results], dtype=object),
         test_logits=np.stack([r['test_logits'] for r in fold_results]),
         test_probs_raw=np.stack([r['test_probs_raw'] for r in fold_results]),
         test_probs_cal=np.stack([r['test_probs_cal'] for r in fold_results]),
         test_labels=np.stack([r['test_labels'] for r in fold_results]),
         train_eids=train_eids, test_eids=test_eids,
         train_years=train_years, test_years=test_years)

tp_raw = np.mean([r['test_probs_raw'] for r in fold_results], axis=0)
tp_cal = np.mean([r['test_probs_cal'] for r in fold_results], axis=0)
tl = fold_results[0]['test_labels']

test_aucs = [roc_auc_score(fold_results[i]['test_labels'], fold_results[i]['test_probs_raw']) for i in range(5)]
cv_aucs = [roc_auc_score(fold_results[i]['val_labels'], fold_results[i]['val_probs_raw']) for i in range(5)]

auc_mean, auc_lo, auc_hi = bootstrap_ci(roc_auc_score, tl, tp_raw)
brier, brier_lo, brier_hi = bootstrap_ci(brier_score_loss, tl, tp_cal)
ece_val, ece_lo, ece_hi = bootstrap_ci(compute_ece, tl, tp_cal)
cidx_mean, cidx_lo, cidx_hi = bootstrap_ci_cindex(test_years, tp_cal, tl)

result = {
    'lr': LR, 'dim': DIM, 'N_features': sum(input_dims.values()),
    'CV_AUC_mean': float(np.mean(cv_aucs)), 'CV_AUC_std': float(np.std(cv_aucs)),
    'Test_AUC_mean': float(np.mean(test_aucs)), 'Test_AUC_std': float(np.std(test_aucs)),
    'AUC_CI_low': auc_lo, 'AUC_CI_high': auc_hi,
    'C-index': cidx_mean, 'C-index_CI_low': cidx_lo, 'C-index_CI_high': cidx_hi,
    'Brier_cal': brier, 'Brier_CI_low': brier_lo, 'Brier_CI_high': brier_hi,
    'ECE_cal': ece_val, 'ECE_CI_low': ece_lo, 'ECE_CI_high': ece_hi,
}
pd.DataFrame([result]).to_csv(os.path.join(OUT_DIR, 'result.csv'), index=False)
with open(os.path.join(OUT_DIR, 'result.json'), 'w') as f:
    json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in result.items()}, f, indent=2)

print(f'\n  lr={LR:g} dim={DIM}: AUC={result["Test_AUC_mean"]:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]  '
      f'C-index={cidx_mean:.4f}  Brier={brier:.4f}  ECE={ece_val:.4f}  '
      f'CV_AUC={result["CV_AUC_mean"]:.4f}', flush=True)
print(f'Saved to: {OUT_DIR}')
print('Done.')
