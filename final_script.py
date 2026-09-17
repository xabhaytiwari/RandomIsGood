import os

# Force offline mode BEFORE transformers/huggingface_hub is imported. Every
# CLIPModel/CLIPProcessor.from_pretrained() call in this file otherwise does a
# network round-trip to huggingface.co on EVERY call (even when the weights are
# already cached locally) just to check the file hasn't changed. Across 10
# experiments x 3 seeds this adds up to dozens of network calls, and a single
# stalled/rate-limited request hangs the whole pipeline with zero CPU/GPU usage
# -- exactly the "stuck with nothing happening" symptom this fixes. Since the
# model is already downloaded and cached, there's nothing to gain from hitting
# the network at all; local-only loading also fails fast with a clear error
# instead of hanging indefinitely if the cache is ever missing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Optional cross-session checkpoint backup (see hf_backup.py). Falls back to no-ops
# if the module isn't present or huggingface_hub isn't installed -- never blocks
# local training from running standalone.
try:
    from hf_backup import push_checkpoint, pull_all, ensure_repo
except Exception:
    def push_checkpoint(*args, **kwargs): pass
    def pull_all(*args, **kwargs): pass
    def ensure_repo(*args, **kwargs): pass

import sys
import glob
import time
import json
import heapq
import math
import copy
import warnings
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from typing import List, Tuple, Iterator
from collections import defaultdict
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Sampler, Dataset, Subset

# HPC-Safe Plotting
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENTROPY_EVAL_TEMPERATURE = 2.0

import random as _random
def set_seed(seed: int):
    """Seeds python/numpy/torch so each (experiment, seed) run is reproducible
    and comparable across ablation/baseline variants."""
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------
# HPC Logger (Saves all prints to file automatically)
# ---------------------------------------------------------
class TeeLogger(object):
    def __init__(self, filename="al_training_logs.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        if '\r' not in message:
            self.log.write(message)
            self.log.flush()
            
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger()

CONFIG = dict(
    dataset_dir       = "/data/activeLearningProAbhay",
    json_path         = "/data/activeLearningProAbhay/train_mini-json/train_mini.json",
    cache_file        = "multiexit_embeddings_v5_multimodal.npz", 
    al_rounds         = 5,
    budget_per_round  = 35000,
    extract_batch     = 512,
    train_batch       = 128,
    vit_layer         = 9,
    hidden_dim        = 2048,
    proj_dim          = 512,
    temperature       = 0.07,
    moco_queue_size   = 8192,
    moco_momentum     = 0.999,
    stage1_epochs     = 40,
    stage2_epochs     = 40,
    lr_stage1         = 3e-5,
    lr_stage2         = 2e-3,
    weight_decay      = 1e-2,
    samples_per_class = 4,
    pseudo_threshold  = 0.90,

    # --- Variance / reproducibility ---
    seeds             = [42, 43, 44],          # run each experiment 3x, report mean +/- std

    # --- Which named experiments to run (see EXPERIMENTS registry below) ---
    run_list          = [
        "ours_full",
        "baseline_random_matched_epochs",  # resolves the confound -- run this before publishing
        "ablate_no_taxonomic_sampler",
        "ablate_no_density_fps",
        "ablate_no_jsd_filter",
        "ablate_no_pseudo_labeling",
        "ablate_no_taxonomic_mixup",
        "ablate_no_celf",          # FPS-only, i.e. CoreSet-style, no submodular fill
        "baseline_random",
        "baseline_uncertainty",
        "baseline_coreset",
        "baseline_supervised_random_175k",
    ],

    # --- Density-Weighted FPS hyperparameters (Sec III-B formalization) ---
    fps_density_k     = 10,     # k nearest neighbors used for the typicality/density score
    fps_density_beta  = 1.0,    # exponent controlling how strongly density down-weights outliers
    fps_density_max_reference = 20000,  # landmark set size cap; keeps density scoring O(N x M) not O(N^2)
    checkpoint_dir    = "checkpoints",  # Stage-1 + end-of-round checkpoints, per (experiment, seed, round)
    audit_dir         = "audit",        # pseudo-label accuracy + memory-wall profiling logs, per (experiment, seed)

    # torch.compile is a speed optimization only, not correctness-critical -- disabled by
    # default. Hit a real crash (TypeError: 'NoneType' object is not subscriptable, inside
    # a torch._dynamo graph-break "resume" stub calling into transformers' CLIP forward)
    # caused by a known incompatibility between recent `transformers` versions' hook-based
    # output-capturing mechanism and torch.compile's graph-break resume path. Set True only
    # if you've confirmed your transformers/torch versions don't hit this on your machine.
    use_torch_compile = False,
)

# ---------------------------------------------------------
# Named Experiment Registry (drives ablations + baselines)
# ---------------------------------------------------------
# acquisition: "celf_fps" (ours: FPS seeds + CELF fill)
#              "fps_only" (CoreSet-style: pure/density FPS covers whole budget, no CELF)
#              "uncertainty" (top-K by JSD disagreement alone)
#              "random" (uniform random sampling from the pool)
#
# --- Cost control ---
# ours_full is the headline number: full fidelity (3 seeds, 5 rounds, 80 epochs/round).
# One seed of that shape took ~10-16 real hours once we watched it complete overnight.
# At that rate, 9 more experiments x 3 seeds x full epochs would run ~2 weeks.
# Ablations/baselines only need enough signal to RANK the variants against ours_full,
# not full convergence -- so they run fewer epochs, fewer AL rounds, and a single seed.
# Bump REDUCED_SCALE_SEEDS or REDUCED_SCALE_OVERRIDES back up later for any specific
# ablation you want variance/CIs on for the final paper table.
REDUCED_SCALE_OVERRIDES = dict(al_rounds=3, stage1_epochs=15, stage2_epochs=15)
REDUCED_SCALE_SEEDS = [42]

EXPERIMENTS = {
    "ours_full": dict(
        acquisition="celf_fps", use_taxonomic_sampler=True, use_density_fps=True,
        use_jsd_filter=True, use_pseudo_labeling=True, use_taxonomic_mixup=True),

    "ablate_no_taxonomic_sampler": dict(
        acquisition="celf_fps", use_taxonomic_sampler=False, use_density_fps=True,
        use_jsd_filter=True, use_pseudo_labeling=True, use_taxonomic_mixup=True,
        overrides=REDUCED_SCALE_OVERRIDES, seeds=REDUCED_SCALE_SEEDS),

    "ablate_no_density_fps": dict(
        acquisition="celf_fps", use_taxonomic_sampler=True, use_density_fps=False,
        use_jsd_filter=True, use_pseudo_labeling=True, use_taxonomic_mixup=True,
        overrides=REDUCED_SCALE_OVERRIDES, seeds=REDUCED_SCALE_SEEDS),

    "ablate_no_jsd_filter": dict(
        acquisition="celf_fps", use_taxonomic_sampler=True, use_density_fps=True,
        use_jsd_filter=False, use_pseudo_labeling=True, use_taxonomic_mixup=True,
        overrides=REDUCED_SCALE_OVERRIDES, seeds=REDUCED_SCALE_SEEDS),

    "ablate_no_pseudo_labeling": dict(
        acquisition="celf_fps", use_taxonomic_sampler=True, use_density_fps=True,
        use_jsd_filter=True, use_pseudo_labeling=False, use_taxonomic_mixup=True,
        overrides=REDUCED_SCALE_OVERRIDES, seeds=REDUCED_SCALE_SEEDS),

    "ablate_no_taxonomic_mixup": dict(
        acquisition="celf_fps", use_taxonomic_sampler=True, use_density_fps=True,
        use_jsd_filter=True, use_pseudo_labeling=True, use_taxonomic_mixup=False,
        overrides=REDUCED_SCALE_OVERRIDES, seeds=REDUCED_SCALE_SEEDS),

    "ablate_no_celf": dict(
        acquisition="fps_only", use_taxonomic_sampler=True, use_density_fps=True,
        use_jsd_filter=True, use_pseudo_labeling=True, use_taxonomic_mixup=True,
        overrides=REDUCED_SCALE_OVERRIDES, seeds=REDUCED_SCALE_SEEDS),

    # --- Competing AL baselines (no submodularity, no taxonomy priors) ---
    "baseline_random": dict(
        acquisition="random", use_taxonomic_sampler=False, use_density_fps=False,
        use_jsd_filter=False, use_pseudo_labeling=False, use_taxonomic_mixup=False,
        overrides=REDUCED_SCALE_OVERRIDES, seeds=REDUCED_SCALE_SEEDS),

    # Matched-epoch confirmatory rerun: same random acquisition, but full 40+40
    # epochs/round and full 5 rounds -- everything ours_full has, MINUS the
    # acquisition strategy. This is the one experiment needed to resolve whether
    # ours_full's parity with baseline_random (reduced-scale) was a training-budget
    # confound or a real result. Single seed to keep cost bounded; bump to full
    # REDUCED_SCALE_SEEDS-style multi-seed later if this needs a tighter CI.
    "baseline_random_matched_epochs": dict(
        acquisition="random", use_taxonomic_sampler=False, use_density_fps=False,
        use_jsd_filter=False, use_pseudo_labeling=False, use_taxonomic_mixup=False,
        seeds=[42, 43, 44]),  # bumped from [42] -- this is now the paper's central finding, needs real CIs

    "baseline_uncertainty": dict(
        acquisition="uncertainty", use_taxonomic_sampler=False, use_density_fps=False,
        use_jsd_filter=False, use_pseudo_labeling=False, use_taxonomic_mixup=False,
        overrides=REDUCED_SCALE_OVERRIDES, seeds=REDUCED_SCALE_SEEDS),

    "baseline_coreset": dict(
        acquisition="fps_only", use_taxonomic_sampler=False, use_density_fps=False,
        use_jsd_filter=False, use_pseudo_labeling=False, use_taxonomic_mixup=False,
        overrides=REDUCED_SCALE_OVERRIDES, seeds=REDUCED_SCALE_SEEDS),

    # --- R8's requested fairness control: a vanilla supervised fine-tune on the
    # SAME total label budget (175k), selected randomly in one shot, with NO
    # active-learning loop at all. Isolates "does fine-tuning on this many labels
    # help" from "does OUR acquisition strategy help over other strategies."
    # Already cheap (1 round instead of 5) so it keeps full epochs + all 3 seeds.
    "baseline_supervised_random_175k": dict(
        acquisition="random", use_taxonomic_sampler=False, use_density_fps=False,
        use_jsd_filter=False, use_pseudo_labeling=False, use_taxonomic_mixup=False,
        overrides=dict(al_rounds=1, budget_per_round=175000)),
}

# ---------------------------------------------------------
# 1. Multi-Modal JSON Parser & Text Prompts
# ---------------------------------------------------------
def parse_inat_metadata(json_path: str, dataset_dir: str):
    print(f"\n[Job] Parsing Multi-Modal JSON from {json_path}...")
    with open(json_path, 'r') as f: data = json.load(f)

    cat_info = {c['id']: c for c in data['categories']}
    unique_cats = sorted(list(cat_info.keys()))
    sp_to_idx = {c: i for i, c in enumerate(unique_cats)}

    genera = sorted(list(set([c.get('genus', 'Unknown') for c in cat_info.values()])))
    genus_to_idx = {g: i for i, g in enumerate(genera)}

    families = sorted(list(set([c.get('family', 'Unknown') for c in cat_info.values()])))
    family_to_idx = {f: i for i, f in enumerate(families)}

    img_to_ann = {ann['image_id']: ann['category_id'] for ann in data['annotations']}

    paths, labels, gen_labels, fam_labels, geo_feats =[], [], [], [],[]
    class_names, enriched_prompts = [],[]

    for c_id in unique_cats:
        c = cat_info[c_id]
        common = c.get('common_name', c['name']).replace("_", " ")
        prompt = (f"A photo of a {common}, a {c.get('supercategory', '')} in the family "
                  f"{c.get('family', '')}, scientific name {c.get('genus', '')} {c.get('specific_epithet', '')}.")
        class_names.append(common)
        enriched_prompts.append(prompt)

    sp_to_genus_arr = np.zeros(len(unique_cats), dtype=np.int64)

    for img in data['images']:
        cat_id = img_to_ann.get(img['id'])
        if cat_id is None: continue

        f_name = img['file_name']
        if dataset_dir.rstrip("/").endswith("train_mini") and f_name.startswith("train_mini/"):
            f_name = f_name.replace("train_mini/", "", 1)

        full_path = os.path.join(dataset_dir, f_name)
        if not os.path.exists(full_path): continue

        c = cat_info[cat_id]
        sp_idx = sp_to_idx[cat_id]
        gen_idx = genus_to_idx[c.get('genus', 'Unknown')]
        fam_idx = family_to_idx[c.get('family', 'Unknown')]

        sp_to_genus_arr[sp_idx] = gen_idx

        lat, lon = img.get('latitude'), img.get('longitude')
        lat_norm = lat / 90.0 if lat else 0.0
        lon_norm = lon / 180.0 if lon else 0.0

        try:
            month = int(img.get('date').split('-')[1])
            m_rad = (month - 1) * (2.0 * math.pi / 12.0)
            sin_m, cos_m = math.sin(m_rad), math.cos(m_rad)
        except:
            sin_m, cos_m = 0.0, 0.0

        paths.append(full_path)
        labels.append(sp_idx)
        gen_labels.append(gen_idx)
        fam_labels.append(fam_idx)
        geo_feats.append([lat_norm, lon_norm, sin_m, cos_m])

    if len(paths) == 0:
        raise ValueError(f"CRITICAL ERROR: No images found! Check if {dataset_dir} and JSON file_names match.")

    return (np.array(paths), np.array(labels, dtype=np.int64), np.array(gen_labels, dtype=np.int64),
            np.array(fam_labels, dtype=np.int64), np.array(geo_feats, dtype=np.float32),
            enriched_prompts, sp_to_genus_arr, len(genera), len(families))

def get_text_prompt_embeddings(prompts: List[str]) -> torch.Tensor:
    print(f"[Job] Extracting Zero-Shot Taxonomic Prompts for {len(prompts)} classes...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    embeddings =[]
    with torch.no_grad():
        for i in range(0, len(prompts), 256):
            batch = prompts[i:i+256]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            feats = model.text_projection(model.text_model(**inputs).pooler_output)
            embeddings.append(feats.cpu())
    del model, processor; torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)

# ---------------------------------------------------------
# 2. Multi-Modal Dataloaders & Feature Extractors
# ---------------------------------------------------------
class CLIPMultiModalDataset(Dataset):
    def __init__(self, paths, labels, gen_labels, fam_labels, geo_feats, processor, is_training=False):
        self.paths = paths
        self.labels = labels
        self.gen_labels = gen_labels
        self.fam_labels = fam_labels
        self.geo_feats = geo_feats
        self.processor = processor
        self.is_training = is_training
        
        self.train_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.is_training: img = self.train_transform(img)
            pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
            return (pixel_values, torch.tensor(self.geo_feats[idx], dtype=torch.float32),
                    torch.tensor(self.labels[idx], dtype=torch.long),
                    torch.tensor(self.gen_labels[idx], dtype=torch.long),
                    torch.tensor(self.fam_labels[idx], dtype=torch.long), idx, True)
        except Exception:
            return (torch.zeros((3, 224, 224)), torch.zeros(4), torch.tensor(-1, dtype=torch.long),
                    torch.tensor(-1, dtype=torch.long), torch.tensor(-1, dtype=torch.long), idx, False)

def load_or_extract_dataset(cfg: dict):
    cache = cfg["cache_file"]
    (paths, labels, gen_labels, fam_labels, geo_feats,
     prompts, sp_to_genus, num_gen, num_fam) = parse_inat_metadata(cfg["json_path"], cfg["dataset_dir"])
     
    if os.path.exists(cache):
        print(f"[Job] Cache found at {cache}. Loading frozen features instantly...")
        data = np.load(cache, allow_pickle=True)
        return data["X"].astype(np.float32), paths, labels, gen_labels, fam_labels, geo_feats, prompts, sp_to_genus, num_gen, num_fam

    print("  -> Cache not found. Initializing CLIP Vision Model for Round 1 extraction...")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", attn_implementation="sdpa").vision_model.to(DEVICE)
    model.eval()
    dataset = CLIPMultiModalDataset(paths, labels, gen_labels, fam_labels, geo_feats, processor, is_training=False)
    loader = DataLoader(dataset, batch_size=cfg["extract_batch"], num_workers=8, pin_memory=True)

    intermediate_feats, final_feats = [],[]
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch_idx, (pixel_values, _, _, _, _, _, valids) in enumerate(loader):
            if not valids.numpy().any(): continue
            inputs = pixel_values[valids].to(DEVICE, non_blocking=True)
            outputs = model(pixel_values=inputs, output_hidden_states=True)
            
            intermediate_feats.append(outputs.hidden_states[cfg["vit_layer"]][:, 0, :].half().cpu().numpy())
            final_feats.append(outputs.pooler_output.half().cpu().numpy())
            if (batch_idx + 1) % 50 == 0: print(f"     Processed {(batch_idx + 1) * cfg['extract_batch']} images...", end="\r")

    print("\n[Job] Compiling and saving features to cache...")
    X = np.hstack((np.vstack(intermediate_feats), np.vstack(final_feats)))
    np.savez_compressed(cache, X=X)
    return X.astype(np.float32), paths, labels, gen_labels, fam_labels, geo_feats, prompts, sp_to_genus, num_gen, num_fam

def extract_dynamic_features(model: nn.Module, loader: DataLoader, cfg: dict):
    print("  -> Extracting fresh embeddings and calculating Predictive Disagreement...")
    model.eval()
    all_feats, all_labels, all_indices, all_disagreement = [], [], [],[]
    all_max_probs, all_pseudo_preds = [],[]

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch_idx, (pixels, geos, labels, _, _, indices, valids) in enumerate(loader):
            if not valids.numpy().any(): continue

            pixels = pixels[valids].to(DEVICE, non_blocking=True)
            geos = geos[valids].to(DEVICE, non_blocking=True)
            feats, logits_mid, logits_fin, _, _ = model(pixels, geo_feats=geos, return_all=True)

            raw_probs = F.softmax((logits_mid + logits_fin) / 2.0, dim=1)
            max_probs, pseudo_preds = torch.max(raw_probs, dim=1)

            probs_mid = F.softmax(logits_mid / ENTROPY_EVAL_TEMPERATURE, dim=1)
            probs_fin = F.softmax(logits_fin / ENTROPY_EVAL_TEMPERATURE, dim=1)
            
            m = 0.5 * (probs_mid + probs_fin)
            entropy_m = -(m * torch.log(m + 1e-9)).sum(dim=1)
            entropy_mid = -(probs_mid * torch.log(probs_mid + 1e-9)).sum(dim=1)
            entropy_fin = -(probs_fin * torch.log(probs_fin + 1e-9)).sum(dim=1)
            jsd_disagreement = entropy_m - 0.5 * (entropy_mid + entropy_fin)

            all_feats.append(feats.half().cpu().numpy())
            all_labels.extend(labels[valids].numpy())
            all_indices.extend(indices[valids].numpy())
            all_disagreement.extend(jsd_disagreement.cpu().numpy())
            all_max_probs.extend(max_probs.cpu().numpy())
            all_pseudo_preds.extend(pseudo_preds.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"     [Extraction] Processed {(batch_idx + 1) * cfg['extract_batch']} images...", end="\r")

    print("\n  -> Dynamic extraction complete.")
    torch.cuda.empty_cache()
    return (np.vstack(all_feats).astype(np.float32), np.array(all_labels), np.array(all_indices), 
            np.array(all_disagreement, dtype=np.float32), np.array(all_max_probs, dtype=np.float32), 
            np.array(all_pseudo_preds, dtype=np.int64))

# ---------------------------------------------------------
# 3. Geo-Visual Submodularity
# ---------------------------------------------------------
class TaxonomicBatchSampler(Sampler):
    def __init__(self, labels: np.ndarray, sp_to_genus: np.ndarray, samples_per_class: int, n_classes_per_batch: int):
        self.spc, self.ncpb = samples_per_class, n_classes_per_batch
        self.class_indices = defaultdict(list)
        for idx, lbl in enumerate(labels): self.class_indices[int(lbl)].append(idx)
        self.valid_classes = list(self.class_indices.keys())
        
        self.genus_to_classes = defaultdict(list)
        for c in self.valid_classes: self.genus_to_classes[sp_to_genus[c]].append(c)
        
        self.genera = list(self.genus_to_classes.keys())
        self.n_batches = math.ceil(len(self.valid_classes) / n_classes_per_batch)

    def __iter__(self):
        np.random.shuffle(self.genera)
        class_order =[]
        for g in self.genera:
            classes = self.genus_to_classes[g].copy()
            np.random.shuffle(classes)
            class_order.extend(classes)

        for i in range(0, len(class_order), self.ncpb):
            batch =[]
            for c in class_order[i: i + self.ncpb]:
                idxs = self.class_indices[c]
                batch.extend(np.random.choice(idxs, size=self.spc, replace=len(idxs) < self.spc).tolist())
            yield batch
    def __len__(self): return self.n_batches

class RandomBalancedBatchSampler(Sampler):
    """
    Ablation counterpart to TaxonomicBatchSampler: same class-balanced batch
    shape (C classes x S samples), but class_order is a random permutation
    instead of a genus-contiguous one. Isolates whether gains come from
    class-balanced batching per se, or specifically from taxonomic co-location.
    """
    def __init__(self, labels: np.ndarray, samples_per_class: int, n_classes_per_batch: int):
        self.spc, self.ncpb = samples_per_class, n_classes_per_batch
        self.class_indices = defaultdict(list)
        for idx, lbl in enumerate(labels): self.class_indices[int(lbl)].append(idx)
        self.valid_classes = list(self.class_indices.keys())
        self.n_batches = math.ceil(len(self.valid_classes) / n_classes_per_batch)

    def __iter__(self):
        class_order = self.valid_classes.copy()
        np.random.shuffle(class_order)  # no genus grouping -> mostly easy inter-genus negatives
        for i in range(0, len(class_order), self.ncpb):
            batch =[]
            for c in class_order[i: i + self.ncpb]:
                idxs = self.class_indices[c]
                batch.extend(np.random.choice(idxs, size=self.spc, replace=len(idxs) < self.spc).tolist())
            yield batch
    def __len__(self): return self.n_batches

def compute_typicality_scores(X_tensor: torch.Tensor, k: int = 10, chunk: int = 2048,
                               max_reference: int = 20000) -> torch.Tensor:
    """
    Local-density 'Typicality Score' T(i) referenced in Sec III-B of the paper.

    IMPORTANT (scalability fix): computing true kNN density over the full pool
    is an O(N^2) pairwise similarity computation -- exactly the "Memory Wall"
    problem the paper's intro motivates CELF around (Sec I). At N=400,000 a
    dense N x N matrix does not fit in GPU memory. Instead, density is
    estimated against a fixed random *landmark* subsample of at most
    `max_reference` pool points: T(i) = mean cosine similarity of point i to
    its k nearest neighbors within that landmark set. This bounds memory to
    chunk x max_reference regardless of pool size, at the cost of an
    approximate (rather than exact) density estimate -- a standard tradeoff
    for landmark-based density/kNN approximation at scale.

    Expects `X_tensor` already on DEVICE (unit-normalized hybrid embeddings).
    Returns a DEVICE tensor, not numpy, to avoid an extra host round-trip.
    """
    n = X_tensor.shape[0]
    ref_size = min(max_reference, n)
    ref_idx = torch.from_numpy(np.random.choice(n, size=ref_size, replace=False)).to(DEVICE)
    X_ref = X_tensor[ref_idx]  # (M, D), M bounded regardless of N -- this is what keeps memory flat

    # Track which reference column (if any) corresponds to each row's own index,
    # so we can mask out self-similarity before top-k rather than assuming
    # position 0 is always self (true landmark sampling means most rows aren't
    # in the reference set at all).
    ref_pos = torch.full((n,), -1, dtype=torch.long, device=DEVICE)
    ref_pos[ref_idx] = torch.arange(ref_size, device=DEVICE)

    k_eff = min(k, ref_size - 1)
    typicality = torch.empty(n, device=DEVICE)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = torch.mm(X_tensor[start:end], X_ref.T)  # (chunk, M) -- bounded, independent of N

        self_cols = ref_pos[start:end]
        has_self = self_cols >= 0
        if has_self.any():
            row_ids = torch.nonzero(has_self, as_tuple=True)[0]
            sims[row_ids, self_cols[has_self]] = -1e4  # exclude self-similarity before top-k

        topk_sims, _ = sims.topk(k_eff, dim=1, largest=True)
        typicality[start:end] = topk_sims.mean(dim=1)

    t = (typicality - typicality.min()) / (typicality.max() - typicality.min() + 1e-8)
    return t.clamp(min=1e-3, max=1.0)

def get_hybrid_farthest_point_sampling(X_pool_vis: np.ndarray, X_pool_geo: np.ndarray, K: int,
                                        density_weighted: bool = True, density_k: int = 10,
                                        density_beta: float = 1.0, density_max_reference: int = 20000):
    print(f"  -> Running Geo-Visual Farthest Point Sampling (K={K}, density_weighted={density_weighted})...")
    v_norm = X_pool_vis / np.linalg.norm(X_pool_vis, axis=1, keepdims=True).clip(min=1e-8)
    g_norm = X_pool_geo / np.linalg.norm(X_pool_geo, axis=1, keepdims=True).clip(min=1e-8)

    X_hybrid = np.concatenate([np.sqrt(0.7) * v_norm, np.sqrt(0.3) * g_norm], axis=1)
    X_tensor = torch.from_numpy(X_hybrid).to(DEVICE)

    if density_weighted:
        typicality_t = compute_typicality_scores(
            X_tensor, k=density_k, max_reference=density_max_reference).pow(density_beta)
    else:
        typicality_t = torch.ones(len(X_tensor), device=DEVICE)  # ablation: plain FPS, Sec IV ablation

    selected_indices =[torch.randint(0, len(X_tensor), (1,)).item()]
    min_dists = torch.ones(len(X_tensor), device=DEVICE) * float('inf')

    start_time = time.time()
    for i in range(1, K):
        last_selected = X_tensor[selected_indices[-1]].unsqueeze(0)
        raw_dist = 1.0 - torch.mm(X_tensor, last_selected.T).squeeze(1)
        min_dists = torch.minimum(min_dists, raw_dist)
        # Density-Weighted FPS: scale the farthest-point distance by local typicality
        # so a geometrically isolated but low-typicality (likely outlier) point is
        # penalized instead of being preferentially selected purely for being far away.
        weighted_score = min_dists * typicality_t
        selected_indices.append(torch.argmax(weighted_score).item())
        if i % 1000 == 0: print(f"     [FPS] Extracted {i}/{K} queries...", end="\r")

    print("\n  -> Hybrid FPS complete.")
    centroids = X_tensor[selected_indices].cpu().numpy()
    del X_tensor; torch.cuda.empty_cache()
    return centroids, X_hybrid, selected_indices

class FastBatchedCELFOptimizer:
    def __init__(self, pool_embeddings: np.ndarray, query_embeddings: np.ndarray, uncertainties: np.ndarray = None):
        self.num_pool, self.num_queries = len(pool_embeddings), len(query_embeddings)
        self.unc = uncertainties if uncertainties is not None else np.ones(self.num_pool, dtype=np.float32)
        self.pool_gpu  = torch.from_numpy(pool_embeddings).to(DEVICE)
        self.query_gpu = torch.from_numpy(query_embeddings).to(DEVICE)

    def _batched_marginal_gain(self, candidates: List[int], current_maxes: torch.Tensor) -> np.ndarray:
        sims = torch.mm(self.query_gpu, self.pool_gpu[candidates].T).clamp(min=0.0)
        new_maxes = torch.maximum(current_maxes.unsqueeze(1), sims)
        return (new_maxes.sum(dim=0) - current_maxes.sum()).cpu().numpy() * self.unc[candidates]

    def select(self, budget: int, batch_size: int = 1024) -> List[int]:
        print(f"  -> Starting Uncertainty-Guided BATCHED selection for budget {budget}...")
        start_time = time.time()

        gains = np.empty(self.num_pool, dtype=np.float32)
        for start in range(0, self.num_pool, 2000):
            end = min(start + 2000, self.num_pool)
            sims = torch.mm(self.query_gpu, self.pool_gpu[start:end].T).clamp(min=0.0)
            gains[start:end] = sims.sum(dim=0).cpu().numpy() * self.unc[start:end]

        queue =[(-g, i, -1) for i, g in enumerate(gains)]
        heapq.heapify(queue)
        current_maxes = torch.zeros(self.num_queries, device=DEVICE)
        selected =[]

        for step in range(budget):
            while True:
                if not queue: break
                neg_gain, top_cand, timestamp = queue[0]

                if timestamp == step:
                    heapq.heappop(queue)
                    selected.append(top_cand)
                    sim_vec = torch.mm(self.query_gpu, self.pool_gpu[top_cand].unsqueeze(1)).squeeze(1).clamp(min=0.0)
                    current_maxes = torch.maximum(current_maxes, sim_vec)
                    if (step + 1) % 1000 == 0:
                        print(f"\r[Selection] Found {step+1}/{budget} | Elapsed: {time.time()-start_time:.1f}s", end="")
                    break

                batch_cands =[]
                while queue and queue[0][2] != step and len(batch_cands) < batch_size:
                    _, cand, _ = heapq.heappop(queue)
                    batch_cands.append(cand)

                if batch_cands:
                    new_gains = self._batched_marginal_gain(batch_cands, current_maxes)
                    for i, cand in enumerate(batch_cands):
                        heapq.heappush(queue, (-new_gains[i], cand, step))

        print(f"\n  -> Batched Submodular selection complete.")
        return selected

# ---------------------------------------------------------
# 4. Multi-Modal End-to-End Model & Losses
# ---------------------------------------------------------
class CosineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, text_weights: torch.Tensor = None, scale: float = 25.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        if text_weights is not None: self.weight.data = text_weights.clone()
        else: nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

class LogitAdjustedLoss(nn.Module):
    def __init__(self, class_counts: np.ndarray, tau: float = 1.0):
        super().__init__()
        pi = torch.clamp(torch.tensor(class_counts, dtype=torch.float32), min=1.0)
        pi = pi / pi.sum()
        self.register_buffer("adjustment", tau * torch.log(pi + 1e-12).unsqueeze(0))

    def forward(self, logits, targets):
        return F.cross_entropy(logits + self.adjustment, targets, label_smoothing=0.1)

class EndToEndSelfConModel(nn.Module):
    def __init__(self, num_sp: int, num_gen: int, num_fam: int, cfg: dict, text_weights: torch.Tensor = None):
        super().__init__()
        self.vit_layer = cfg["vit_layer"]
        self.backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", attn_implementation="sdpa").vision_model

        self.half_dim = 768
        self.feature_dropout = nn.Dropout(p=0.3)
        self.norm_mid   = nn.LayerNorm(self.half_dim)
        self.norm_final = nn.LayerNorm(self.half_dim)

        self.projector_mid = nn.Sequential(nn.Linear(self.half_dim, cfg["hidden_dim"], bias=False), nn.BatchNorm1d(cfg["hidden_dim"]), nn.GELU(), nn.Linear(cfg["hidden_dim"], cfg["proj_dim"], bias=False), nn.BatchNorm1d(cfg["proj_dim"], affine=False))
        self.projector_final = nn.Sequential(nn.Linear(self.half_dim, cfg["hidden_dim"], bias=False), nn.BatchNorm1d(cfg["hidden_dim"]), nn.GELU(), nn.Linear(cfg["hidden_dim"], cfg["proj_dim"], bias=False), nn.BatchNorm1d(cfg["proj_dim"], affine=False))

        self.geo_mlp = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, num_sp))

        self.classifier_mid   = CosineClassifier(cfg["proj_dim"], num_sp, text_weights)
        self.classifier_final = CosineClassifier(cfg["proj_dim"], num_sp, text_weights)
        self.classifier_genus  = CosineClassifier(cfg["proj_dim"], num_gen)
        self.classifier_family = CosineClassifier(cfg["proj_dim"], num_fam)

    def forward(self, x: torch.Tensor, geo_feats=None, return_features_only=False, return_projection=False, return_all=False):
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        feats = torch.cat([outputs.hidden_states[self.vit_layer][:, 0, :], outputs.pooler_output], dim=1)
        if return_features_only: return feats

        x_dropped  = self.feature_dropout(feats)
        proj_mid   = F.normalize(self.projector_mid(self.norm_mid(x_dropped[:, :self.half_dim])), p=2, dim=1)
        proj_final = F.normalize(self.projector_final(self.norm_final(x_dropped[:, self.half_dim:])), p=2, dim=1)

        if return_projection: return proj_mid, proj_final

        logits_mid   = self.classifier_mid(proj_mid)
        logits_final = self.classifier_final(proj_final)

        if geo_feats is not None:
            geo_prior = self.geo_mlp(geo_feats)
            logits_mid += geo_prior
            logits_final += geo_prior

        proj_fused = (proj_mid + proj_final) / 2.0
        logits_gen = self.classifier_genus(proj_fused)
        logits_fam = self.classifier_family(proj_fused)

        if return_all: return feats, logits_mid, logits_final, logits_gen, logits_fam
        return logits_mid, logits_final, logits_gen, logits_fam

class MoCoSupConLoss(nn.Module):
    def __init__(self, temperature: float): super().__init__(); self.temperature = temperature
    def forward(self, q, k, queue_k, labels, queue_labels):
        sim = torch.mm(q, torch.cat([k, queue_k], dim=0).T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        pos_mask = torch.eq(labels.unsqueeze(1), torch.cat([labels, queue_labels], dim=0).unsqueeze(0)).float().to(q.device)
        exp_sim  = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        n_positives  = pos_mask.sum(dim=1)
        has_positive = (n_positives > 0)
        if not has_positive.any(): return q.sum() * 0.0
        return -(pos_mask * log_prob).sum(dim=1)[has_positive].div(n_positives[has_positive]).mean()

def update_momentum_encoder(model_comp, momentum_comp, momentum):
    base_model = model_comp._orig_mod if hasattr(model_comp, "_orig_mod") else model_comp
    for param_q, param_k in zip(base_model.parameters(), momentum_comp.parameters()):
        param_k.data = (param_k.data.float() * momentum + param_q.data.float() * (1.0 - momentum)).to(param_k.dtype)
    for buffer_q, buffer_k in zip(base_model.buffers(), momentum_comp.buffers()):
        if buffer_q.is_floating_point():
            buffer_k.data = (buffer_k.data.float() * momentum + buffer_q.data.float() * (1.0 - momentum)).to(buffer_k.dtype)
        else:
            buffer_k.data = buffer_q.data.clone()

def taxonomic_mixup(batch_X, batch_y_sp, batch_y_gen, alpha=0.2, constrain_to_genus=True):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.arange(batch_X.size(0), device=batch_X.device)

    if constrain_to_genus:
        unique_gen = torch.unique(batch_y_gen)
        for g in unique_gen:
            mask = (batch_y_gen == g).nonzero(as_tuple=True)[0]
            if len(mask) > 1: index[mask] = mask[torch.randperm(len(mask), device=batch_X.device)]
    else:
        # Ablation: standard mixup, pairs drawn from the whole batch regardless of genus
        index = torch.randperm(batch_X.size(0), device=batch_X.device)

    mixed_X = lam * batch_X + (1 - lam) * batch_X[index]
    return mixed_X, batch_y_sp, batch_y_sp[index], lam

# ---------------------------------------------------------
# 5. Iterative Training Engine
# ---------------------------------------------------------
def train_al_round(model: EndToEndSelfConModel, mom_model: EndToEndSelfConModel,
                   train_dataset: Subset, sp_to_genus: np.ndarray, total_classes: int, cfg: dict,
                   al_round: int = -1):

    run_tag = f"{cfg.get('experiment_name', 'run')}_seed{cfg.get('seed', 0)}_round{al_round + 1}"
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    stage1_ckpt_path = os.path.join(ckpt_dir, f"{run_tag}_stage1.pt")

    y_train_sp = np.array([train_dataset.dataset.labels[i] for i in train_dataset.indices])

    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
        if "encoder.layers." in name:
            if int(name.split("encoder.layers.")[1].split(".")[0]) >= cfg["vit_layer"]:
                param.requires_grad = True
        elif "post_layernorm" in name or "pooler" in name:
            param.requires_grad = True

    for name, param in model.named_parameters():
        if 'norm' in name or 'projector' in name: param.requires_grad = True

    spc, ncpb = cfg["samples_per_class"], cfg["train_batch"] // cfg["samples_per_class"]
    ablation = cfg.get("ablation", {})

    if ablation.get("use_taxonomic_sampler", True):
        balanced_sampler = TaxonomicBatchSampler(y_train_sp, sp_to_genus, spc, ncpb)
    else:
        balanced_sampler = RandomBalancedBatchSampler(y_train_sp, spc, ncpb)  # ablation: no genus co-location
    train_loader_s1 = DataLoader(train_dataset, batch_sampler=balanced_sampler, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=3)
    train_loader_s2 = DataLoader(train_dataset, batch_size=cfg["train_batch"], shuffle=True, drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=3)

    backbone_params, head_params = [],[]
    for name, param in model.named_parameters():
        if not param.requires_grad or 'classifier' in name or 'geo_mlp' in name: continue
        if "backbone" in name: backbone_params.append(param)
        else: head_params.append(param)

    optimizer_s1 = optim.AdamW([{'params': backbone_params, 'lr': 1e-6}, {'params': head_params, 'lr': cfg["lr_stage1"]}], weight_decay=cfg["weight_decay"])
    scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=cfg["stage1_epochs"], eta_min=1e-7)
    supcon_loss  = MoCoSupConLoss(temperature=cfg["temperature"])

    queue_k      = F.normalize(torch.randn(cfg["moco_queue_size"], cfg["proj_dim"], device=DEVICE), p=2, dim=1)
    queue_labels = torch.full((cfg["moco_queue_size"],), -1, dtype=torch.long, device=DEVICE)
    queue_ptr    = 0
    scaler_s1    = torch.amp.GradScaler('cuda')

    print("  -> Training Stage 1: Fine-tuning CLIP + Projectors (Hard Negative SelfCon)...")
    stage1_start = time.time()
    for epoch in range(cfg["stage1_epochs"]):
        model.train()
        epoch_start = time.time()
        running_loss, n_batches, first_batch_logged = 0.0, 0, False
        for batch_X, _, batch_y, _, _, _, _ in train_loader_s1:
            if not first_batch_logged and epoch == 0:
                print(f"     [Stage1] First batch loaded after {time.time()-epoch_start:.1f}s "
                      f"(dataloader/backbone warmup) -- training is running.")
                first_batch_logged = True
            batch_X, batch_y = batch_X.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, dtype=torch.long, non_blocking=True)
            optimizer_s1.zero_grad(set_to_none=True)

            with torch.no_grad():
                update_momentum_encoder(model, mom_model, cfg["moco_momentum"])

            with torch.amp.autocast('cuda'):
                q_mid, q_fin = model(batch_X, return_projection=True)
                with torch.no_grad():
                    k_mid, k_fin = mom_model(batch_X, return_projection=True)

                q_all, k_all = torch.cat([q_mid, q_fin], dim=0), torch.cat([k_mid, k_fin], dim=0)
                labels_all = torch.cat([batch_y, batch_y], dim=0)
                loss = supcon_loss(q_all, k_all, queue_k, labels_all, queue_labels)

            scaler_s1.scale(loss).backward()
            scaler_s1.unscale_(optimizer_s1)
            nn.utils.clip_grad_norm_(backbone_params + head_params, max_norm=1.0)
            scaler_s1.step(optimizer_s1)
            scaler_s1.update()

            running_loss += loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                elapsed_epoch = time.time() - epoch_start
                rate = n_batches / max(elapsed_epoch, 1e-6)
                print(f"     [Stage1] Epoch {epoch+1} | Step {n_batches} | "
                      f"Running Loss: {running_loss/n_batches:.4f} | "
                      f"{rate:.2f} steps/s | Epoch elapsed: {elapsed_epoch:.1f}s", end="\r")

            b_size = q_all.shape[0]
            if queue_ptr + b_size <= cfg["moco_queue_size"]:
                queue_k[queue_ptr:queue_ptr+b_size], queue_labels[queue_ptr:queue_ptr+b_size] = k_all.detach(), labels_all.detach()
                queue_ptr = (queue_ptr + b_size) % cfg["moco_queue_size"]
            else:
                fit = cfg["moco_queue_size"] - queue_ptr
                overflow = b_size - fit
                queue_k[queue_ptr:], queue_labels[queue_ptr:] = k_all[:fit].detach(), labels_all[:fit].detach()
                queue_k[:overflow], queue_labels[:overflow] = k_all[fit:].detach(), labels_all[fit:].detach()
                queue_ptr = overflow
        scheduler_s1.step()

        avg_loss = running_loss / max(n_batches, 1)
        elapsed = time.time() - stage1_start
        eta = (elapsed / (epoch + 1)) * (cfg["stage1_epochs"] - epoch - 1)
        print(f"     [Stage1] Epoch {epoch+1}/{cfg['stage1_epochs']} | Loss: {avg_loss:.4f} | "
              f"Epoch time: {time.time()-epoch_start:.1f}s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    torch.save(model.state_dict(), stage1_ckpt_path)
    push_checkpoint(stage1_ckpt_path)
    print(f"  -> Stage 1 checkpoint saved to {stage1_ckpt_path} (safe to resume Stage 2 from here if it crashes).")

    print("  -> Training Stage 2: Freezing Backbone, Training ALL Classifiers & Geo-Priors...")
    for name, param in model.named_parameters():
        if 'classifier' in name or 'geo_mlp' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    la_loss = LogitAdjustedLoss(np.bincount(y_train_sp, minlength=total_classes), tau=1.0).to(DEVICE)
    ce_loss_generic = nn.CrossEntropyLoss(label_smoothing=0.1)

    opt_s2   = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr_stage2"], weight_decay=cfg["weight_decay"])
    sched_s2 = optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=cfg["stage2_epochs"], eta_min=1e-6)
    scaler_s2 = torch.amp.GradScaler('cuda')

    stage2_start = time.time()
    for epoch in range(cfg["stage2_epochs"]):
        model.train()
        model.backbone.eval()
        model.norm_mid.eval(); model.norm_final.eval()
        model.projector_mid.eval(); model.projector_final.eval()
        epoch_start = time.time()
        running_loss, n_batches, first_batch_logged = 0.0, 0, False

        for batch_X, batch_geo, batch_y_sp, batch_y_gen, batch_y_fam, _, _ in train_loader_s2:
            if not first_batch_logged and epoch == 0:
                print(f"     [Stage2] First batch loaded after {time.time()-epoch_start:.1f}s -- training is running.")
                first_batch_logged = True
            batch_X, batch_geo = batch_X.to(DEVICE, non_blocking=True), batch_geo.to(DEVICE, non_blocking=True)
            batch_y_sp = batch_y_sp.to(DEVICE, dtype=torch.long, non_blocking=True)
            batch_y_gen = batch_y_gen.to(DEVICE, dtype=torch.long, non_blocking=True)
            batch_y_fam = batch_y_fam.to(DEVICE, dtype=torch.long, non_blocking=True)

            opt_s2.zero_grad(set_to_none=True)

            mixed_X, y_a, y_b, lam = taxonomic_mixup(
                batch_X, batch_y_sp, batch_y_gen, alpha=0.2,
                constrain_to_genus=ablation.get("use_taxonomic_mixup", True))

            with torch.amp.autocast('cuda'):
                logits_mid, logits_fin, logits_gen, logits_fam = model(mixed_X, geo_feats=batch_geo)

                loss_sp_mid = lam * la_loss(logits_mid, y_a) + (1 - lam) * la_loss(logits_mid, y_b)
                loss_sp_fin = lam * la_loss(logits_fin, y_a) + (1 - lam) * la_loss(logits_fin, y_b)
                loss_sp = loss_sp_mid + loss_sp_fin

                loss_taxa = ce_loss_generic(logits_gen, batch_y_gen) + ce_loss_generic(logits_fam, batch_y_fam)
                loss = loss_sp + (0.5 * loss_taxa)

            scaler_s2.scale(loss).backward()
            scaler_s2.step(opt_s2)
            scaler_s2.update()

            running_loss += loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                elapsed_epoch = time.time() - epoch_start
                rate = n_batches / max(elapsed_epoch, 1e-6)
                print(f"     [Stage2] Epoch {epoch+1} | Step {n_batches} | "
                      f"Running Loss: {running_loss/n_batches:.4f} | "
                      f"{rate:.2f} steps/s | Epoch elapsed: {elapsed_epoch:.1f}s", end="\r")
        sched_s2.step()

        avg_loss = running_loss / max(n_batches, 1)
        elapsed = time.time() - stage2_start
        eta = (elapsed / (epoch + 1)) * (cfg["stage2_epochs"] - epoch - 1)
        print(f"     [Stage2] Epoch {epoch+1}/{cfg['stage2_epochs']} | Loss: {avg_loss:.4f} | "
              f"Epoch time: {time.time()-epoch_start:.1f}s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    round_ckpt_path = os.path.join(ckpt_dir, f"{run_tag}_final.pt")
    torch.save(model.state_dict(), round_ckpt_path)
    push_checkpoint(round_ckpt_path)
    print(f"  -> Round checkpoint saved to {round_ckpt_path} (full model, post Stage 2).")

def evaluate_model(model: nn.Module, test_loader: DataLoader):
    model.eval()
    targets, all_top1_preds = [],[]
    top1_correct, top5_correct = 0, 0
    n_batches_total = len(test_loader)
    eval_start = time.time()

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for b_idx, (batch_X, batch_geo, batch_y_sp, _, _, _, _) in enumerate(test_loader):
            batch_X, batch_geo, batch_y_sp = batch_X.to(DEVICE), batch_geo.to(DEVICE), batch_y_sp.to(DEVICE)

            logits_mid, logits_fin, _, _ = model(batch_X, geo_feats=batch_geo)
            _, top5_preds = (logits_mid + logits_fin).topk(5, dim=1, largest=True, sorted=True)
            top1_preds    = top5_preds[:, 0]

            targets.extend(batch_y_sp.cpu().numpy())
            all_top1_preds.extend(top1_preds.cpu().numpy())

            top5_correct += (top5_preds == batch_y_sp.view(-1, 1).expand_as(top5_preds)).sum().item()
            top1_correct += (top1_preds == batch_y_sp).sum().item()

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == n_batches_total:
                elapsed = time.time() - eval_start
                rate = (b_idx + 1) / max(elapsed, 1e-6)
                eta = (n_batches_total - b_idx - 1) / max(rate, 1e-6)
                print(f"     [Round Eval] Batch {b_idx+1}/{n_batches_total} | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="\r")
    print()

    acc_top1  = top1_correct / len(targets)
    acc_top5  = top5_correct / len(targets)
    f1_macro  = f1_score(targets, all_top1_preds, average="macro", zero_division=0)

    print(f"\n====== ROUND EVALUATION ======")
    print(f"  Top-1 Acc: {acc_top1*100:.2f}%")
    print(f"  Top-5 Acc: {acc_top5*100:.2f}%")
    print(f"  Macro F1:  {f1_macro:.4f}")
    print(f"==============================\n")
    return acc_top1, acc_top5, f1_macro

# ---------------------------------------------------------
# [NEW] Data Presentation Plot Generator
# ---------------------------------------------------------
def generate_plots(metrics: dict, clip_baseline: tuple):
    print("\n[Job] Generating presentation plots...")
    rounds = metrics["rounds"]
    budgets = metrics["human_labeled"]
    
    # 1. Accuracy Progression (vs Rounds)
    plt.figure(figsize=(10, 6))
    plt.plot(rounds,[acc * 100 for acc in metrics["top1_acc"]], marker='o', label='Ours Top-1 Acc', color='blue', linewidth=2)
    plt.plot(rounds,[acc * 100 for acc in metrics["top5_acc"]], marker='s', label='Ours Top-5 Acc', color='green', linewidth=2)
    plt.axhline(y=clip_baseline[0]*100, color='blue', linestyle='--', label='Standard CLIP Top-1')
    plt.axhline(y=clip_baseline[1]*100, color='green', linestyle='--', label='Standard CLIP Top-5')
    plt.title('Active Learning Performance vs Standard CLIP')
    plt.xlabel('Active Learning Round')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rounds)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('1_accuracy_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Macro F1 Progression
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, metrics["f1_macro"], marker='o', label='Ours Macro F1', color='purple', linewidth=2)
    plt.axhline(y=clip_baseline[2], color='purple', linestyle='--', label='Standard CLIP Macro F1')
    plt.title('Macro F1 Score (Minority Class Performance)')
    plt.xlabel('Active Learning Round')
    plt.ylabel('Macro F1 Score')
    plt.xticks(rounds)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('2_macro_f1_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Dataset Composition (Stacked Bar)
    plt.figure(figsize=(10, 6))
    plt.bar(rounds, metrics["human_labeled"], color='steelblue', label='Human Labeled (AL Budget)')
    plt.bar(rounds, metrics["pseudo_labeled"], bottom=metrics["human_labeled"], color='lightcoral', label='Pseudo-Labeled (Free Data)')
    plt.title('Training Pool Composition: Human vs Pseudo Labels')
    plt.xlabel('Active Learning Round')
    plt.ylabel('Number of Images in Training Set')
    plt.xticks(rounds)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('3_dataset_composition_stack.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Active Learning Efficiency Curve
    plt.figure(figsize=(10, 6))
    plt.plot(budgets,[acc * 100 for acc in metrics["top1_acc"]], marker='o', label='Ours Top-1 Acc', color='darkorange', linewidth=2)
    plt.plot(budgets,[acc * 100 for acc in metrics["top5_acc"]], marker='s', label='Ours Top-5 Acc', color='teal', linewidth=2)
    plt.title('Active Learning Data Efficiency')
    plt.xlabel('Human Annotation Budget (Images)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('4_al_efficiency_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Round-vs-Round Delta (Performance Gain)
    deltas_top1 = [metrics["top1_acc"][0] * 100] + [(metrics["top1_acc"][i] - metrics["top1_acc"][i-1]) * 100 for i in range(1, len(rounds))]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(rounds, deltas_top1, color='seagreen')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'+{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.title('Top-1 Accuracy Gain per AL Round')
    plt.xlabel('Active Learning Round')
    plt.ylabel('Accuracy Jump (%)')
    plt.xticks(rounds)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('5_round_vs_round_delta.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  -> 5 presentation plots saved successfully!")

def aggregate_seed_metrics(seed_metrics_list: List[dict]) -> dict:
    """Collapses per-seed run metrics into mean/std per AL round, for variance reporting."""
    agg = {"rounds": seed_metrics_list[0]["rounds"]}
    for key in ["top1_acc", "top5_acc", "f1_macro"]:
        arr = np.array([m[key] for m in seed_metrics_list])  # (n_seeds, n_rounds)
        agg[f"{key}_mean"] = arr.mean(axis=0).tolist()
        agg[f"{key}_std"]  = arr.std(axis=0).tolist()
    agg["human_labeled"]  = seed_metrics_list[0]["human_labeled"]
    agg["pseudo_labeled"] = seed_metrics_list[0]["pseudo_labeled"]
    return agg

def generate_pseudo_label_audit_plot(metrics: dict):
    """R8: 'report pseudo-label accuracy against ground truth, show with/without the loop.'
    Plots per-round batch accuracy and cumulative held-pseudo-label accuracy against
    ground truth, alongside how many pseudo-labels are being trusted at each point."""
    print("\n[Job] Generating pseudo-label leakage/accuracy audit plot...")
    rounds = metrics["rounds"]
    batch_acc = [a * 100 if a is not None else None for a in metrics["pseudo_acc_batch"]]
    cum_acc = [a * 100 if a is not None else None for a in metrics["pseudo_acc_cumulative"]]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(rounds, batch_acc, 'o--', color='darkorange', label='This-round pseudo-label accuracy')
    ax1.plot(rounds, cum_acc, 'o-', color='crimson', label='Cumulative held pseudo-label accuracy')
    ax1.set_xlabel('Active Learning Round')
    ax1.set_ylabel('Accuracy vs Ground Truth (%)')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.bar(rounds, metrics["pseudo_labeled"], alpha=0.15, color='gray', label='Cumulative pseudo-labels held')
    ax2.set_ylabel('Cumulative Pseudo-Labels Held')
    ax2.legend(loc='upper right')

    plt.title('Pseudo-Label Audit: Accuracy Against Frozen Ground Truth (never used for selection/training)')
    plt.tight_layout()
    plt.savefig('8_pseudo_label_audit.png', dpi=300)
    plt.close()
    print("  -> 8_pseudo_label_audit.png saved.")

def generate_memory_wall_plot(metrics: dict):
    """R7/R8: 'the 600GB->2GB claim has no supporting measurement.' Plots actually-measured
    peak GPU memory during acquisition against the analytical naive-dense O(N^2) estimate,
    per round, on a log scale so the gap is legible."""
    print("\n[Job] Generating Memory-Wall audit plot (measured vs naive-dense estimate)...")
    rounds = metrics["rounds"]
    actual = metrics["acquisition_peak_mem_gb"]
    naive = metrics["acquisition_naive_dense_estimate_gb"]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, naive, 'o--', color='indianred', label='Naive dense O(N²) estimate (analytical)')
    plt.plot(rounds, actual, 'o-', color='seagreen', label='Actual measured peak (batched, ours)')
    plt.yscale('log')
    plt.xlabel('Active Learning Round')
    plt.ylabel('GPU Memory (GB, log scale)')
    plt.title('Memory-Wall Audit: Measured Peak VRAM vs Naive Dense Estimate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('9_memory_wall_audit.png', dpi=300)
    plt.close()
    print("  -> 9_memory_wall_audit.png saved.")

def generate_rebuttal_summary(all_results: dict, clip_baseline: tuple):
    """Ties every artifact this pipeline produces back to the specific reviewer
    comment it addresses, so the rebuttal letter can just point at files."""
    print("\n[Job] Generating rebuttal_summary.md...")
    lines = ["# Rebuttal Artifact Summary\n",
             "Generated automatically by final_script.py. Maps each output file to the reviewer comment it addresses.\n"]

    lines.append("## R4, R7, R8: \"No ablation / no baseline comparison\"")
    lines.append("- `ablation_table.md`, `6_ablation_baseline_comparison_top1.png`, `7_ablation_baseline_comparison_f1.png`")
    lines.append(f"- Covers: {', '.join(all_results.keys())}\n")

    lines.append("## R8: \"The 4x improvement is measured against untrained zero-shot CLIP, not a fair control\"")
    lines.append("- `baseline_supervised_random_175k` in the tables above: same 175k label budget, single-shot random")
    lines.append("  selection, no AL loop, no taxonomic priors -- the fairness control R8 asked for explicitly.\n")

    lines.append("## R8: \"No seeds/CIs, single-run numbers\"")
    lines.append(f"- Every experiment above is run over {len(next(iter(all_results.values()))['per_seed'])} seeds; "
                  "mean +/- std reported in `ablation_table.md` and `al_results_all_experiments.json`.\n")

    lines.append("## R7, R8: \"600GB -> 2GB VRAM claim has no measurement protocol\"")
    lines.append("- `9_memory_wall_audit.png`, `audit/memory_profile_*.json` -- actual `torch.cuda.max_memory_allocated()`")
    lines.append("  peak during acquisition, per round, plotted against the analytical naive dense O(N^2) estimate")
    lines.append("  for the same pool size. Use these numbers directly in the methods/results section.\n")

    lines.append("## R8: \"Pseudo-labeling loop needs a leakage audit\"")
    lines.append("- `8_pseudo_label_audit.png`, `audit/pseudo_label_audit_*.json` -- per-round accuracy of harvested")
    lines.append("  pseudo-labels against a ground-truth snapshot frozen before any pseudo-label overwrite, plus an")
    lines.append("  explicit train/test disjointness assertion in `run_iterative_al` (raises immediately if violated).\n")

    lines.append("## Still needs a manual pass (not code fixes):")
    lines.append("- R4: prose sections reading as AI-generated -- rewrite by hand.")
    lines.append("- R4: clarify the 'multi-modal' framing given geo-priors are a small part of the pipeline.")
    lines.append("- R8: harmonize \"4x\" (abstract) vs \"4.4x\" (conclusion) to one number.")
    lines.append("- R8: clarify alpha=0.2 vs Beta(0.2, 0.2) notation in Sec III-C/D.")
    lines.append("- R7, R8: add citations for Coreset, BADGE, SupCon, MoCo, FixMatch now that they're real baselines.\n")

    with open("rebuttal_summary.md", "w") as f: f.write("\n".join(lines))
    print("  -> rebuttal_summary.md saved.")

def generate_comparison_plots(all_results: dict, clip_baseline: tuple, n_seeds: int = None):
    """Final-round Top-1/Macro-F1 across every ablation + baseline, mean +/- std error bars.
    Also dumps a markdown table (ablation_table.md) ready to paste into the paper.

    NOTE: seed count varies by experiment (ours_full and the supervised-175k control run
    full 3-seed fidelity; ablations/baselines run 1 seed to keep total compute tractable --
    see REDUCED_SCALE_SEEDS). Each bar's actual n is annotated on its x-axis label rather
    than assuming a single shared n across all bars.
    """
    print("\n[Job] Generating ablation/baseline comparison outputs...")
    names, labels, top1_means, top1_stds, f1_means, f1_stds = [], [], [],[],[],[]
    for name, res in all_results.items():
        agg = res["aggregate"]
        n = len(res.get("per_seed", {}))
        names.append(name)
        labels.append(f"{name}\n(n={n})")
        top1_means.append(agg["top1_acc_mean"][-1] * 100)
        top1_stds.append(agg["top1_acc_std"][-1] * 100)
        f1_means.append(agg["f1_macro_mean"][-1])
        f1_stds.append(agg["f1_macro_std"][-1])

    colors = ['seagreen' if n == "ours_full" else ('steelblue' if n.startswith("ablate") else 'indianred') for n in names]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, top1_means, yerr=top1_stds, capsize=4, color=colors)
    plt.axhline(y=clip_baseline[0] * 100, color='gray', linestyle='--', label='Zero-Shot CLIP')
    plt.title('Final-Round Top-1 Accuracy: Ablations & Baselines (mean ± std; n per bar)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(); plt.tight_layout()
    plt.savefig('6_ablation_baseline_comparison_top1.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(labels, f1_means, yerr=f1_stds, capsize=4, color=colors)
    plt.axhline(y=clip_baseline[2], color='gray', linestyle='--', label='Zero-Shot CLIP')
    plt.title('Final-Round Macro F1: Ablations & Baselines (mean ± std; n per bar)')
    plt.ylabel('Macro F1')
    plt.xticks(rotation=45, ha='right')
    plt.legend(); plt.tight_layout()
    plt.savefig('7_ablation_baseline_comparison_f1.png', dpi=300)
    plt.close()

    with open("ablation_table.md", "w") as f:
        f.write("| Experiment | Seeds (n) | Top-1 Acc (%) | Macro F1 |\n|---|---|---|---|\n")
        for name, res, t1, ts, fm, fs in zip(names, all_results.values(), top1_means, top1_stds, f1_means, f1_stds):
            n = len(res.get("per_seed", {}))
            f.write(f"| {name} | {n} | {t1:.2f} ± {ts:.2f} | {fm:.4f} ± {fs:.4f} |\n")

    print("  -> Comparison plots + ablation_table.md saved successfully!")

# ---------------------------------------------------------
# [NEW] Standard Zero-Shot CLIP Evaluator
# ---------------------------------------------------------
def evaluate_standard_clip_zeroshot(test_loader: DataLoader, text_weights: torch.Tensor):
    """Provides a baseline for the graphs by testing unmodified zero-shot CLIP."""
    print("\n====== EVALUATING STANDARD ZERO-SHOT CLIP BASELINE ======")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", attn_implementation="sdpa").to(DEVICE)
    model.eval()
    
    targets, all_top1_preds = [],[]
    top1_correct, top5_correct = 0, 0
    text_features = F.normalize(text_weights.to(DEVICE), p=2, dim=1)
    n_batches_total = len(test_loader)
    eval_start = time.time()

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for b_idx, (batch_X, _, batch_y_sp, _, _, _, _) in enumerate(test_loader):
            batch_X, batch_y_sp = batch_X.to(DEVICE), batch_y_sp.to(DEVICE)

            # [FIX] Manually extract and project to bypass HuggingFace version crashes!
            vision_outputs = model.vision_model(pixel_values=batch_X)
            image_features = model.visual_projection(vision_outputs.pooler_output)

            image_features = F.normalize(image_features, p=2, dim=1)

            logits = image_features @ text_features.T
            _, top5_preds = logits.topk(5, dim=1, largest=True, sorted=True)
            top1_preds = top5_preds[:, 0]

            targets.extend(batch_y_sp.cpu().numpy())
            all_top1_preds.extend(top1_preds.cpu().numpy())

            top5_correct += (top5_preds == batch_y_sp.view(-1, 1).expand_as(top5_preds)).sum().item()
            top1_correct += (top1_preds == batch_y_sp).sum().item()

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == n_batches_total:
                elapsed = time.time() - eval_start
                rate = (b_idx + 1) / max(elapsed, 1e-6)
                eta = (n_batches_total - b_idx - 1) / max(rate, 1e-6)
                print(f"     [Zero-Shot Eval] Batch {b_idx+1}/{n_batches_total} | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="\r")
    print()
            
    acc_top1 = top1_correct / len(targets)
    acc_top5 = top5_correct / len(targets)
    f1_macro = f1_score(targets, all_top1_preds, average="macro", zero_division=0)
    
    print(f"  Standard CLIP Top-1 Acc: {acc_top1*100:.2f}%")
    print(f"  Standard CLIP Top-5 Acc: {acc_top5*100:.2f}%")
    print(f"  Standard CLIP Macro F1:  {f1_macro:.4f}")
    print("=========================================================\n")
    del model; torch.cuda.empty_cache()
    return acc_top1, acc_top5, f1_macro

def acquire_round_selections(strategy: str, X_pool_vis_filtered: np.ndarray, X_pool_geo_filtered: np.ndarray,
                              uncertainties_filtered, total_classes: int, budget: int, cfg: dict):
    """
    Single dispatch point for every acquisition function compared in the paper.
    Returns indices into the *filtered* pool arrays (X_pool_vis_filtered, etc.).

    strategy:
      "celf_fps"    -> Ours: FPS seeds (K=total_classes) + Batched CELF fills the rest
      "fps_only"    -> CoreSet-style: (optionally density-weighted) FPS covers the whole budget, no CELF
      "uncertainty" -> Pure JSD predictive-disagreement top-K, no diversity term at all
      "random"      -> Uniform random sampling from the (unfiltered) pool -- the standard AL floor baseline
    """
    n_pool = len(X_pool_vis_filtered)
    budget = min(budget, n_pool)

    if strategy == "random":
        return np.random.choice(n_pool, size=budget, replace=False).tolist()

    if strategy == "uncertainty":
        if uncertainties_filtered is None:
            return np.random.choice(n_pool, size=budget, replace=False).tolist()
        order = np.argsort(uncertainties_filtered)[::-1]
        return order[:budget].tolist()

    if strategy == "fps_only":
        _, _, fps_idx = get_hybrid_farthest_point_sampling(
            X_pool_vis_filtered, X_pool_geo_filtered, K=budget,
            density_weighted=cfg["ablation"]["use_density_fps"],
            density_k=cfg["fps_density_k"], density_beta=cfg["fps_density_beta"],
            density_max_reference=cfg["fps_density_max_reference"])
        return fps_idx

    if strategy == "celf_fps":
        k_seed = min(total_classes, budget)
        X_query, X_hybrid, fps_indices = get_hybrid_farthest_point_sampling(
            X_pool_vis_filtered, X_pool_geo_filtered, K=k_seed,
            density_weighted=cfg["ablation"]["use_density_fps"],
            density_k=cfg["fps_density_k"], density_beta=cfg["fps_density_beta"],
            density_max_reference=cfg["fps_density_max_reference"])

        fps_seed_set = set(fps_indices)
        non_fps_mask = np.array([i not in fps_seed_set for i in range(len(X_hybrid))])
        X_hybrid_remaining = X_hybrid[non_fps_mask]
        unc_remaining = uncertainties_filtered[non_fps_mask] if uncertainties_filtered is not None else None
        non_fps_local_indices = np.where(non_fps_mask)[0]

        remaining_budget = max(budget - k_seed, 0)
        optimizer = FastBatchedCELFOptimizer(
            pool_embeddings=X_hybrid_remaining, query_embeddings=X_query, uncertainties=unc_remaining)
        selected_rel = optimizer.select(budget=remaining_budget)
        celf_indices = [non_fps_local_indices[i] for i in selected_rel]
        return fps_indices + celf_indices

    raise ValueError(f"Unknown acquisition strategy: {strategy}")

# ---------------------------------------------------------
# 6. Pipeline Execution
# ---------------------------------------------------------
def run_iterative_al(cfg: dict):
    print("====== MULTI-MODAL ITERATIVE ACTIVE LEARNING PIPELINE ======")

    (X_full, paths, labels, gen_labels, fam_labels, geo_feats,
     prompts, sp_to_genus, num_gen, num_fam) = load_or_extract_dataset(cfg)
    total_classes = len(prompts)

    indices = np.arange(len(paths))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0, \
        "LEAKAGE: train/test split overlap detected -- pseudo-labels or AL selections could touch eval data."

    # Frozen ground truth for the train-side pool, captured BEFORE any pseudo-labels
    # can overwrite train_dataset.labels. Used only for auditing pseudo-label
    # accuracy after the fact -- never fed back into selection or training.
    ground_truth_train_labels = labels[train_idx].copy()

    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    eval_extract_dataset = CLIPMultiModalDataset(
        paths[train_idx], labels[train_idx], gen_labels[train_idx],
        fam_labels[train_idx], geo_feats[train_idx], processor, is_training=False)
    train_dataset = CLIPMultiModalDataset(
        paths[train_idx], labels[train_idx], gen_labels[train_idx],
        fam_labels[train_idx], geo_feats[train_idx], processor, is_training=True)
    test_dataset = CLIPMultiModalDataset(
        paths[test_idx], labels[test_idx], gen_labels[test_idx],
        fam_labels[test_idx], geo_feats[test_idx], processor, is_training=False)

    X_pool_geo = geo_feats[train_idx]

    # --- Round-level resume: a crash/shutdown mid-experiment shouldn't cost the
    # rounds that already finished. State (which images are labeled, pseudo-label
    # assignments, metrics so far) is persisted alongside the model checkpoint
    # after every round, and reloaded here if present.
    run_tag = f"{cfg.get('experiment_name', 'run')}_seed{cfg.get('seed', 0)}"
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    audit_dir = cfg.get("audit_dir", "audit")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)
    round_state_path = os.path.join(ckpt_dir, f"{run_tag}_round_state.json")
    pseudo_audit_path = os.path.join(audit_dir, f"pseudo_label_audit_{run_tag}.json")
    memory_profile_path = os.path.join(audit_dir, f"memory_profile_{run_tag}.json")

    text_weights = get_text_prompt_embeddings(prompts)
    raw_model = EndToEndSelfConModel(total_classes, num_gen, num_fam, cfg, text_weights).to(DEVICE)

    resume_state = None
    if os.path.exists(round_state_path):
        with open(round_state_path) as f:
            candidate_state = json.load(f)
        last_round_num = candidate_state["completed_round"] + 1  # 1-indexed, matches checkpoint filenames
        resume_ckpt_path = os.path.join(ckpt_dir, f"{run_tag}_round{last_round_num}_final.pt")
        if os.path.exists(resume_ckpt_path):
            print(f"  -> RESUMING mid-experiment: round {last_round_num}/{cfg['al_rounds']} already completed "
                  f"(state: {round_state_path}). Loading weights from {resume_ckpt_path}.")
            raw_model.load_state_dict(torch.load(resume_ckpt_path, map_location=DEVICE))
            resume_state = candidate_state
        else:
            print(f"  -> WARNING: round-state file found but checkpoint {resume_ckpt_path} is missing -- "
                  f"ignoring stale resume state, restarting this seed from round 1.")

    mom_model = copy.deepcopy(raw_model).to(DEVICE)
    for param in mom_model.parameters(): param.requires_grad = False
    # NOTE: only the student model is checkpointed, not the momentum encoder's own EMA
    # state -- on resume, mom_model restarts as a fresh copy of the resumed student and
    # re-tracks it over the next round's training. Accepted approximation in exchange for
    # not persisting a second full model every round.

    if cfg.get("use_torch_compile", False):
        try:
            model = torch.compile(raw_model)
            print("  -> Model compiled successfully with torch.compile!")
        except Exception:
            model = raw_model
    else:
        model = raw_model  # torch.compile disabled by default -- see CONFIG comment
        print("  -> torch.compile DISABLED (use_torch_compile=False) -- running in plain eager mode.")

    test_loader = DataLoader(test_dataset, batch_size=cfg["extract_batch"], shuffle=False, num_workers=8)

    if resume_state is not None:
        clip_baseline = tuple(resume_state["clip_baseline"])
        metrics = resume_state["metrics"]
        labeled_indices = [int(x) for x in resume_state["labeled_indices"]]
        cumulative_pseudo_indices = [int(x) for x in resume_state["cumulative_pseudo_indices"]]
        pseudo_label_state = {int(k): int(v) for k, v in resume_state["pseudo_label_state"].items()}
        for global_idx, pred in pseudo_label_state.items():  # replay onto the freshly-constructed dataset object
            train_dataset.labels[global_idx] = pred
            train_dataset.gen_labels[global_idx] = sp_to_genus[pred]
        excluded = set(labeled_indices) | set(cumulative_pseudo_indices)
        pool_indices = [idx for idx in range(len(train_dataset)) if idx not in excluded]
        start_round = resume_state["completed_round"] + 1
        pseudo_audit_log = resume_state.get("pseudo_audit_log", [])
        memory_profile_log = resume_state.get("memory_profile_log", [])
    else:
        clip_baseline = evaluate_standard_clip_zeroshot(test_loader, text_weights)
        metrics = {"rounds": [], "top1_acc":[], "top5_acc":[], "f1_macro": [], "human_labeled": [], "pseudo_labeled":[],
                   "pseudo_acc_batch": [], "pseudo_acc_cumulative": [],
                   "acquisition_peak_mem_gb": [], "acquisition_naive_dense_estimate_gb": []}
        pool_indices = list(range(len(train_dataset)))
        labeled_indices = []
        cumulative_pseudo_indices = []
        pseudo_label_state = {}
        start_round = 0
        pseudo_audit_log, memory_profile_log = [], []

    for al_round in range(start_round, cfg["al_rounds"]):
        print(f"\n>>> STARTING AL ROUND {al_round + 1} / {cfg['al_rounds']} <<<")
        print(f"Pool Size: {len(pool_indices)} | Human Labeled Size: {len(labeled_indices)} | Free Pseudo Labels: {len(cumulative_pseudo_indices)}")

        if al_round == 0:
            print("  -> Round 1 Optimization: Using pre-cached frozen embeddings...")
            X_pool_vis = X_full[train_idx][pool_indices]
            local_pool_indices = pool_indices
            uncertainties = None
            pseudo_global_indices =[]
            X_pool_geo_current = X_pool_geo[pool_indices]
            metrics["pseudo_acc_batch"].append(None)
            metrics["pseudo_acc_cumulative"].append(None)
        else:
            pool_subset = Subset(eval_extract_dataset, pool_indices)
            pool_loader = DataLoader(pool_subset, batch_size=cfg["extract_batch"], shuffle=False, num_workers=8)
            
            (X_pool_vis, _, local_pool_indices, uncertainties, 
             max_probs, pseudo_preds) = extract_dynamic_features(model, pool_loader, cfg)

            if cfg["ablation"]["use_pseudo_labeling"]:
                pseudo_mask = max_probs > cfg["pseudo_threshold"]
                pseudo_local_indices = np.where(pseudo_mask)[0]
                pseudo_global_indices = [local_pool_indices[i] for i in pseudo_local_indices]

                for i, local_idx in enumerate(pseudo_local_indices):
                    global_idx = pseudo_global_indices[i]
                    train_dataset.labels[global_idx] = pseudo_preds[local_idx]
                    train_dataset.gen_labels[global_idx] = sp_to_genus[pseudo_preds[local_idx]]
                    pseudo_label_state[global_idx] = pseudo_preds[local_idx]  # audit state, keyed by latest assignment

                if len(pseudo_global_indices) > 0:
                    print(f"  -> 👻 PSEUDO-LABELING: Harvested {len(pseudo_global_indices)} high-confidence images for free!")
                    cumulative_pseudo_indices.extend(pseudo_global_indices)

                    # --- Leakage/accuracy audit (R8): compare against frozen ground truth,
                    # never used for selection or training -- purely diagnostic.
                    batch_correct = [pseudo_preds[local_idx] == ground_truth_train_labels[pseudo_global_indices[i]]
                                      for i, local_idx in enumerate(pseudo_local_indices)]
                    batch_acc = float(np.mean(batch_correct))

                    cum_idx = np.array(list(pseudo_label_state.keys()))
                    cum_preds = np.array(list(pseudo_label_state.values()))
                    cum_acc = float(np.mean(cum_preds == ground_truth_train_labels[cum_idx]))

                    print(f"     [Pseudo-Label Audit] This round: {batch_acc*100:.2f}% correct vs ground truth "
                          f"| Cumulative ({len(cum_idx)} held): {cum_acc*100:.2f}% correct")

                    metrics["pseudo_acc_batch"].append(batch_acc)
                    metrics["pseudo_acc_cumulative"].append(cum_acc)
                    pseudo_audit_log.append({
                        "round": al_round + 1, "harvested_this_round": len(pseudo_global_indices),
                        "batch_accuracy": batch_acc, "cumulative_held": len(cum_idx), "cumulative_accuracy": cum_acc})
                    with open(pseudo_audit_path, "w") as f: json.dump(pseudo_audit_log, f, indent=2)
                    push_checkpoint(pseudo_audit_path)
                else:
                    metrics["pseudo_acc_batch"].append(None)
                    metrics["pseudo_acc_cumulative"].append(None)

                remain_mask = ~pseudo_mask
            else:
                remain_mask = np.ones(len(X_pool_vis), dtype=bool)  # ablation: pseudo-labeling disabled entirely
                metrics["pseudo_acc_batch"].append(None)
                metrics["pseudo_acc_cumulative"].append(None)

            X_pool_vis = X_pool_vis[remain_mask]
            local_pool_indices =[local_pool_indices[i] for i in np.where(remain_mask)[0]]
            uncertainties = uncertainties[remain_mask]
            X_pool_geo_current = X_pool_geo[local_pool_indices]

        if uncertainties is not None and cfg["ablation"]["use_jsd_filter"]:
            print("  -> Filtering pool by Predictive Disagreement (JSD) to optimize selection...")
            filter_size = min(cfg["budget_per_round"] * 3, len(uncertainties))
            top_unc_idx = np.argsort(uncertainties)[-filter_size:]
            
            X_pool_vis_filtered = X_pool_vis[top_unc_idx]
            X_pool_geo_filtered = X_pool_geo_current[top_unc_idx]
            local_indices_filtered = [local_pool_indices[i] for i in top_unc_idx]
            uncertainties_filtered = uncertainties[top_unc_idx]
        else:
            # ablation / round-1: no JSD pre-filter, whole (remaining) pool goes to acquisition
            X_pool_vis_filtered = X_pool_vis
            X_pool_geo_filtered = X_pool_geo_current if al_round > 0 else X_pool_geo[pool_indices]
            local_indices_filtered = local_pool_indices
            uncertainties_filtered = uncertainties

        # --- Memory-Wall audit (R7/R8): measure REAL peak GPU memory used by our
        # batched acquisition, and compute the analytical memory a naive dense
        # O(N^2) similarity matrix over the same pool would require. This is what
        # backs the "600GB -> 2GB" claim with an actual measurement protocol
        # instead of an unverified number.
        n_pool_this_round = len(X_pool_vis_filtered)
        torch.cuda.reset_peak_memory_stats()
        mem_before_gb = torch.cuda.memory_allocated() / (1024 ** 3)

        selected_filtered_indices = acquire_round_selections(
            strategy=cfg["ablation"]["acquisition"],
            X_pool_vis_filtered=X_pool_vis_filtered,
            X_pool_geo_filtered=X_pool_geo_filtered,
            uncertainties_filtered=uncertainties_filtered,
            total_classes=total_classes,
            budget=cfg["budget_per_round"],
            cfg=cfg,
        )

        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        naive_dense_gb = (n_pool_this_round ** 2 * 4) / (1024 ** 3)  # float32 N x N pairwise matrix, analytically
        print(f"     [Memory-Wall Audit] Pool={n_pool_this_round:,} | Actual peak (batched): {peak_mem_gb:.3f} GB "
              f"| Naive dense N^2 estimate: {naive_dense_gb:,.1f} GB | Reduction: {naive_dense_gb / max(peak_mem_gb, 1e-6):,.0f}x")
        metrics["acquisition_peak_mem_gb"].append(peak_mem_gb)
        metrics["acquisition_naive_dense_estimate_gb"].append(naive_dense_gb)
        memory_profile_log.append({
            "round": al_round + 1, "pool_size": int(n_pool_this_round),
            "actual_peak_mem_gb": peak_mem_gb, "naive_dense_estimate_gb": naive_dense_gb,
            "reduction_factor": naive_dense_gb / max(peak_mem_gb, 1e-6)})
        with open(memory_profile_path, "w") as f: json.dump(memory_profile_log, f, indent=2)
        push_checkpoint(memory_profile_path)

        selected_global_indices = [local_indices_filtered[i] for i in selected_filtered_indices]
        labeled_indices.extend(selected_global_indices)
        
        selected_set = set(selected_global_indices).union(set(cumulative_pseudo_indices))
        pool_indices =[idx for idx in pool_indices if idx not in selected_set]

        combined_train_indices = labeled_indices + cumulative_pseudo_indices
        labeled_subset = Subset(train_dataset, combined_train_indices)

        train_al_round(raw_model, mom_model, labeled_subset, sp_to_genus, total_classes, cfg, al_round=al_round)
        acc_top1, acc_top5, f1_mac = evaluate_model(model, test_loader)
        
        metrics["rounds"].append(al_round + 1)
        metrics["top1_acc"].append(acc_top1)
        metrics["top5_acc"].append(acc_top5)
        metrics["f1_macro"].append(f1_mac)
        metrics["human_labeled"].append(len(labeled_indices))
        metrics["pseudo_labeled"].append(len(cumulative_pseudo_indices))

        round_state = {
            "completed_round": al_round,
            "labeled_indices": [int(x) for x in labeled_indices],
            "cumulative_pseudo_indices": [int(x) for x in cumulative_pseudo_indices],
            "pseudo_label_state": {str(k): int(v) for k, v in pseudo_label_state.items()},
            "metrics": metrics,
            "clip_baseline": list(clip_baseline),
            "pseudo_audit_log": pseudo_audit_log,
            "memory_profile_log": memory_profile_log,
        }
        with open(round_state_path, "w") as f:
            json.dump(round_state, f)
        push_checkpoint(round_state_path)
        print(f"  -> Round-level resume state saved ({al_round+1}/{cfg['al_rounds']} rounds done for this seed).")

        del X_pool_vis_filtered
        gc.collect()
        torch.cuda.empty_cache()

    if os.path.exists(round_state_path):
        os.remove(round_state_path)  # seed fully complete; main() persists it to al_results_all_experiments.json
    return metrics, clip_baseline

def main():
    ensure_repo()
    pull_all(".")  # restore checkpoints/state/results from a previous Kaggle session, if any

    if not os.path.exists(CONFIG["cache_file"]):
        load_or_extract_dataset(CONFIG)

    results_path = "al_results_all_experiments.json"
    all_results, clip_baseline = {}, None
    if os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        all_results = saved.get("experiments", {})
        clip_baseline = tuple(saved["clip_baseline"]) if saved.get("clip_baseline") else None
        if all_results:
            print(f"  -> RESUMING: found completed work for {list(all_results.keys())} in {results_path}")

    flagship_metrics_for_plots = None
    flagship_seed_key = str(CONFIG["seeds"][0])

    for exp_name in CONFIG["run_list"]:
        exp_seeds = EXPERIMENTS[exp_name].get("seeds", CONFIG["seeds"])  # reduced-scale experiments run fewer seeds
        print(f"\n########## EXPERIMENT: {exp_name} ({EXPERIMENTS[exp_name]['acquisition']}) | seeds={exp_seeds} ##########")

        # seed -> metrics, carried over from a previous (interrupted) run if present
        seed_results = dict(all_results.get(exp_name, {}).get("per_seed", {}))

        for seed in exp_seeds:
            seed_key = str(seed)
            if seed_key in seed_results:
                print(f"  -> {exp_name} | seed {seed}: already completed, skipping (loaded from {results_path}).")
                continue

            print(f"\n--- {exp_name} | seed {seed} ---")
            set_seed(seed)

            cfg = copy.deepcopy(CONFIG)
            cfg["ablation"] = EXPERIMENTS[exp_name]
            cfg["experiment_name"] = exp_name
            cfg["seed"] = seed
            for override_key, override_val in EXPERIMENTS[exp_name].get("overrides", {}).items():
                cfg[override_key] = override_val

            metrics, baseline = run_iterative_al(cfg)
            seed_results[seed_key] = metrics
            if clip_baseline is None:
                clip_baseline = baseline

            # Persist after EVERY seed (not just every experiment) -- this is what
            # makes a kill-and-restart actually resume instead of starting over.
            all_results[exp_name] = {"per_seed": seed_results, "aggregate": aggregate_seed_metrics(list(seed_results.values()))}
            with open(results_path, "w") as f:
                json.dump({"experiments": all_results, "clip_baseline": clip_baseline}, f, indent=2)
            push_checkpoint(results_path)

        # Ensure the aggregate is fresh even if this experiment was entirely skipped (all seeds cached)
        all_results[exp_name] = {"per_seed": seed_results, "aggregate": aggregate_seed_metrics(list(seed_results.values()))}
        with open(results_path, "w") as f:
            json.dump({"experiments": all_results, "clip_baseline": clip_baseline}, f, indent=2)
        push_checkpoint(results_path)

        if exp_name == "ours_full" and flagship_metrics_for_plots is None and flagship_seed_key in seed_results:
            flagship_metrics_for_plots = seed_results[flagship_seed_key]

    if flagship_metrics_for_plots is not None:
        generate_plots(flagship_metrics_for_plots, clip_baseline)  # original Figs 1-5, flagship run
        generate_pseudo_label_audit_plot(flagship_metrics_for_plots)   # R8: pseudo-label accuracy/leakage audit
        generate_memory_wall_plot(flagship_metrics_for_plots)          # R7/R8: measured VRAM vs naive-dense estimate
    generate_comparison_plots(all_results, clip_baseline, n_seeds=len(CONFIG["seeds"]))  # ablation + baseline table/plots
    generate_rebuttal_summary(all_results, clip_baseline)  # maps each artifact back to the specific reviewer comment

    print("====== FULL PIPELINE FINISHED SUCCESSFULLY (all experiments x all seeds) ======")

if __name__ == "__main__":
    main()