"""Stage-4b recovery: subject embedding -> mixture weights (regression);
session-conditioned embedding -> per-session family (classification), with the
subject-only embedding as the dissociation control. Runs INSIDE the wrapper
container (Beaker) where W&B artifacts + the loader are reachable.

Usage: python stage4b_recovery.py <inventory.json> <outdir>
inventory.json: {run_id: {enc, emb, state, data_cfg:{...}}}
Writes: <outdir>/stage4b_recovery.json  (all metrics)
        <outdir>/emb_<run>.csv           (per-subject embedding + mixweights, for figures)
"""
import os, sys, json, numpy as np, pandas as pd
# code root = parent of this script's dir (.../code/analysis/ -> .../code)
_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE_ROOT)
sys.path.insert(0, "/workspace/aind-disrnn-wrapper/code")
import wandb
api = wandb.Api(); ENT, PROJ = "AIND-disRNN", "embedding_recovery"
from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents
from utils.multisubject import compute_session_conditioned_context_dataframe, extract_subject_embeddings_from_params
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold, KFold
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

inv = json.load(open(sys.argv[1])); outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)

def regen_gt(cfg):
    ld = HierarchicalCognitiveAgents(
        task=cfg["task"], agent=cfg["agent"], num_trials=cfg["num_trials"],
        num_subjects=cfg["num_subjects"], num_sessions_per_subject=cfg["num_sessions_per_subject"],
        eval_every_n=cfg.get("eval_every_n", 2), batch_size=cfg.get("batch_size"),
        batch_mode=cfg.get("batch_mode", "random"),
        subject_seed_stride=cfg.get("subject_seed_stride", 100000),
        generation_workers=1, seed=cfg.get("seed", 42),
        heldout_session_mode=cfg.get("heldout_session_mode", "tail"),
        heldout_frac=cfg.get("heldout_frac", 0.2))
    return ld.groundtruth_table()

def fetch(run, fn, dest, mtype="gru"):
    a = api.artifact(f"{ENT}/{PROJ}/{mtype}-output-{run}:latest", type="training-output")
    return a.get_entry(fn).download(root=dest)

results = {}
gt_cache = {}
for run, meta in inv.items():
    if meta["state"] != "finished":
        continue
    enc, emb = meta["enc"], meta["emb"]
    rd = os.path.join(outdir, run); os.makedirs(rd, exist_ok=True)
    mtype = meta.get("model", "gru")
    cfg_fn = "disrnn_config.json" if mtype == "disrnn" else "gru_config.json"
    try:
        P = json.load(open(fetch(run, "params.json", rd, mtype)))
        md = json.load(open(fetch(run, "multisubject_metadata.json", rd, mtype)))
        _cfg = json.load(open(fetch(run, cfg_fn, rd, mtype)))
        arch = _cfg.get("architecture", _cfg)  # GRU nests under 'architecture'; disRNN is flat
        ckey = json.dumps(meta["data_cfg"], sort_keys=True)
        if ckey not in gt_cache:
            gt_cache[ckey] = regen_gt(meta["data_cfg"])
        gt = gt_cache[ckey]
        s2i = md["subject_id_to_index"]
        # model-agnostic subject-embedding extraction (GRU or disRNN)
        subj_emb = extract_subject_embeddings_from_params(P)  # (N, D)
        mix_cols = [c for c in gt.columns if c.startswith("mixweight_")]
        fams = [c.replace("mixweight_", "") for c in mix_cols]

        # ---------- (1) SUBJECT embedding -> mixture weights ----------
        # one row per subject
        subj_gt = gt.groupby("subject_id")[mix_cols].first().reset_index()
        Xs = np.array([subj_emb[s2i[sid]] for sid in subj_gt["subject_id"]])
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        mix_r2 = {}
        for c in mix_cols:
            yhat = cross_val_predict(LinearRegression(), Xs, subj_gt[c].values, cv=kf)
            mix_r2[c] = float(r2_score(subj_gt[c].values, yhat))
        mix_r2_mean = float(np.mean(list(mix_r2.values())))

        # dominant-family classification from subject embedding
        dom = gt.groupby("subject_id")["preset_name"].agg(lambda s: s.value_counts().index[0])
        dom = dom.reindex(subj_gt["subject_id"]).values
        try:
            yhat_dom = cross_val_predict(LogisticRegression(max_iter=2000), Xs, dom, cv=5)
            dom_acc = float(accuracy_score(dom, yhat_dom))
        except Exception:
            dom_acc = None

        out = {"run": run, "enc": enc, "emb": emb, "N": len(subj_gt),
               "lik_rel": meta.get("lik_rel"),
               "mix_r2_per_family": mix_r2, "mix_r2_mean": mix_r2_mean,
               "dominant_family_acc": dom_acc, "families": fams}

        # ---------- (2) SESSION embedding -> per-session family (scalar only) ----------
        if enc == "scalar":
            ctx = compute_session_conditioned_context_dataframe(
                P, session_context=md["session_context"],
                session_encoding_type=arch["session_encoding_type"],
                session_integration_type=arch.get("session_integration_type", "direct"),
                session_fourier_k=int(arch.get("session_fourier_k", 4)),
                session_delta_n_layers=int(arch["session_delta_n_layers"]),
                session_delta_hidden_size=int(arch["session_delta_hidden_size"]),
                session_curriculum_lambda=1.0,
                session_max_index_by_subject_index=md["session_max_index_by_subject_index"],
                train_session_ids=md.get("train_session_ids"),
                eval_session_ids=md.get("eval_session_ids"),
                selected_subject_indices=None)
            ecols = [c for c in ctx.columns if c.startswith("embedding_")]
            m = ctx.merge(gt[["subject_id", "session_id", "preset_name"]],
                          on=["subject_id", "session_id"], how="inner")
            Xsess = m[ecols].values
            Xsubj = np.array([subj_emb[s2i[sid]] for sid in m["subject_id"]])  # broadcast
            groups = m["subject_id"].values
            y = m["preset_name"].values
            gkf = GroupKFold(n_splits=5)
            yhat_sess = cross_val_predict(LogisticRegression(max_iter=2000), Xsess, y, cv=gkf, groups=groups)
            yhat_subj = cross_val_predict(LogisticRegression(max_iter=2000), Xsubj, y, cv=gkf, groups=groups)
            out["persession_family_acc_sessioncond"] = float(accuracy_score(y, yhat_sess))
            out["persession_family_acc_subjectonly"] = float(accuracy_score(y, yhat_subj))
            labs = sorted(set(y))
            out["persession_confusion_sessioncond"] = confusion_matrix(y, yhat_sess, labels=labs).tolist()
            out["persession_labels"] = labs
            out["n_session_rows"] = int(len(m))
            m.to_csv(os.path.join(outdir, f"ctx_{run}.csv"), index=False)

        # save per-subject embedding + mixweights for figures
        emb_df = subj_gt.copy()
        for d in range(Xs.shape[1]):
            emb_df[f"emb_{d}"] = Xs[:, d]
        emb_df["dominant_family"] = dom
        emb_df.to_csv(os.path.join(outdir, f"emb_{run}.csv"), index=False)

        results[run] = out
        print(f"{run} {enc}/D{emb}: mix_R2_mean={mix_r2_mean:.3f} dom_acc={dom_acc} "
              f"sess_fam={out.get('persession_family_acc_sessioncond')} "
              f"subj_fam={out.get('persession_family_acc_subjectonly')}", flush=True)
    except Exception as ex:
        import traceback; print(f"{run} FAILED: {type(ex).__name__}: {ex}", flush=True)
        traceback.print_exc()

json.dump(results, open(os.path.join(outdir, "stage4b_recovery.json"), "w"), indent=1)
print("WROTE", len(results), "runs", flush=True)
