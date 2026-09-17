import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300})

with open("al_results_all_experiments.json") as f:
    data = json.load(f)
exps = data["experiments"]
clip_top1, clip_top5, clip_f1 = data["clip_baseline"]

ours = exps["ours_full"]["aggregate"]
rounds = ours["rounds"]
top1_mean = np.array(ours["top1_acc_mean"]) * 100
top1_std = np.array(ours["top1_acc_std"]) * 100
top5_mean = np.array(ours["top5_acc_mean"]) * 100
top5_std = np.array(ours["top5_acc_std"]) * 100
f1_mean = np.array(ours["f1_macro_mean"])
f1_std = np.array(ours["f1_macro_std"])

# Fig 1
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.errorbar(rounds, top1_mean, yerr=top1_std, marker='o', capsize=3, color='#1f77b4', label='Ours Top-1 Acc')
ax.errorbar(rounds, top5_mean, yerr=top5_std, marker='o', capsize=3, color='#2ca02c', label='Ours Top-5 Acc')
ax.axhline(clip_top1*100, color='#1f77b4', linestyle='--', alpha=0.6, label='Zero-Shot CLIP Top-1')
ax.axhline(clip_top5*100, color='#2ca02c', linestyle='--', alpha=0.6, label='Zero-Shot CLIP Top-5')
ax.set_xlabel('Active Learning Round'); ax.set_ylabel('Accuracy (%)')
ax.set_title('Active Learning Performance vs. Zero-Shot CLIP Baseline\n(mean ± std over 3 seeds)')
ax.set_xticks(rounds); ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig('figs/fig1_accuracy_progression.pdf'); plt.close()

# Fig 2
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.errorbar(rounds, f1_mean, yerr=f1_std, marker='o', capsize=3, color='#9467bd', label='Ours Macro F1')
ax.axhline(clip_f1, color='gray', linestyle='--', label='Zero-Shot CLIP Macro F1')
ax.set_xlabel('Active Learning Round'); ax.set_ylabel('Macro F1')
ax.set_title('Macro F1 Progression — Minority Class Recovery\n(mean ± std over 3 seeds)')
ax.set_xticks(rounds); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig('figs/fig2_macro_f1.pdf'); plt.close()

# Fig 3: now fully-powered 3-vs-3 seed comparison
rand_matched = exps["baseline_random_matched_epochs"]["aggregate"]
n_seeds_rand_matched = len(exps["baseline_random_matched_epochs"]["per_seed"])
rm_top1 = np.array(rand_matched["top1_acc_mean"]) * 100
rm_top1_std = np.array(rand_matched["top1_acc_std"]) * 100

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.errorbar(rounds, top1_mean, yerr=top1_std, marker='o', capsize=3, linewidth=2, color='#1f77b4', label='Ours (Full), n=3')
ax.errorbar(rounds, rm_top1, yerr=rm_top1_std, marker='s', capsize=3, linewidth=2, color='#d62728',
            label=f'Random Sampling (matched epochs), n={n_seeds_rand_matched}')
ax.axhline(clip_top1*100, color='gray', linestyle=':', alpha=0.7, label='Zero-Shot CLIP')
ax.set_xlabel('Active Learning Round'); ax.set_ylabel('Top-1 Accuracy (%)')
ax.set_title('The Controlled Comparison: Same Epochs, Same Rounds,\nDifferent Acquisition Strategy (both arms, 3 seeds)')
ax.set_xticks(rounds); ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig('figs/fig3_matched_epoch_comparison.pdf'); plt.close()

# Fig 4: reduced-scale comparison (unchanged data)
names, top1_at_r3 = [], []
DISPLAY = {"ours_full":"Ours (Full)","ablate_no_taxonomic_sampler":"w/o Taxonomic Batch Sampler",
    "ablate_no_density_fps":"w/o Density-Weighted FPS","ablate_no_jsd_filter":"w/o JSD Filter",
    "ablate_no_pseudo_labeling":"w/o Pseudo-Labeling","ablate_no_taxonomic_mixup":"w/o Taxonomic Mixup",
    "ablate_no_celf":"w/o CELF (FPS-only)","baseline_random":"Random (reduced scale)",
    "baseline_uncertainty":"Uncertainty-only (JSD top-k)","baseline_coreset":"CoreSet-style (FPS-only)"}
for key in ["ours_full","ablate_no_taxonomic_sampler","ablate_no_density_fps","ablate_no_jsd_filter",
            "ablate_no_pseudo_labeling","ablate_no_taxonomic_mixup","ablate_no_celf","baseline_random",
            "baseline_uncertainty","baseline_coreset"]:
    agg = exps[key]["aggregate"]; idx = agg["rounds"].index(3)
    names.append(DISPLAY[key]); top1_at_r3.append(agg["top1_acc_mean"][idx]*100)
order = np.argsort(top1_at_r3)[::-1]
names_sorted = [names[i] for i in order]; top1_sorted = [top1_at_r3[i] for i in order]
def color_for(n):
    if n=='Ours (Full)': return '#2ca02c'
    if 'w/o' in n: return '#1f77b4'
    return '#d62728'
colors = [color_for(n) for n in names_sorted]
fig, ax = plt.subplots(figsize=(8.5,5.5))
bars = ax.barh(names_sorted, top1_sorted, color=colors)
ax.axvline(clip_top1*100, color='gray', linestyle='--', label='Zero-Shot CLIP')
ax.set_xlabel('Top-1 Accuracy (%) at Matched Budget (105k labels, Round 3), 15+15 epochs, 1 seed')
ax.set_title('Reduced-Scale Comparison: Ours vs. Component Ablations\nand Other Engineered Acquisition Strategies')
for bar,val in zip(bars,top1_sorted): ax.text(val+0.3, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=8)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig('figs/fig4_engineered_baselines_comparison.pdf'); plt.close()

# Fig 5: memory wall (unchanged)
mem_all = [exps["ours_full"]["per_seed"][s] for s in exps["ours_full"]["per_seed"]]
naive_gb = np.mean([m["acquisition_naive_dense_estimate_gb"] for m in mem_all], axis=0)
actual_gb = np.mean([m["acquisition_peak_mem_gb"] for m in mem_all], axis=0)
fig, ax = plt.subplots(figsize=(6,4.2))
ax.plot(rounds, naive_gb, 'o--', color='#d62728', label='Naive Dense O(N²) Estimate (analytical)')
ax.plot(rounds, actual_gb, 'o-', color='#2ca02c', label='Actual Measured Peak (Batched CELF, ours)')
ax.set_yscale('log'); ax.set_xlabel('Active Learning Round'); ax.set_ylabel('GPU Memory (GB, log scale)')
ax.set_title('Memory-Wall Audit: Measured vs. Naive-Dense VRAM\n(mean over 3 seeds)')
ax.set_xticks(rounds); ax.legend(fontsize=8)
for r,n,a in zip(rounds, naive_gb, actual_gb): ax.annotate(f'{n/a:.0f}×', xy=(r, np.sqrt(n*a)), fontsize=8, ha='center', color='dimgray')
plt.tight_layout(); plt.savefig('figs/fig5_memory_wall.pdf'); plt.close()

# Fig 6: pseudo-label audit (unchanged)
pacc_batch = [np.mean([m["pseudo_acc_batch"][i] for m in mem_all if m["pseudo_acc_batch"][i] is not None]) if any(m["pseudo_acc_batch"][i] is not None for m in mem_all) else None for i in range(len(rounds))]
pacc_cum = [np.mean([m["pseudo_acc_cumulative"][i] for m in mem_all if m["pseudo_acc_cumulative"][i] is not None]) if any(m["pseudo_acc_cumulative"][i] is not None for m in mem_all) else None for i in range(len(rounds))]
pseudo_count = ours["pseudo_labeled"]
fig, ax1 = plt.subplots(figsize=(6.5,4.2))
r_valid = [r for r,v in zip(rounds, pacc_batch) if v is not None]
batch_valid = [v*100 for v in pacc_batch if v is not None]; cum_valid = [v*100 for v in pacc_cum if v is not None]
ax1.plot(r_valid, batch_valid, 'o--', color='darkorange', label='This-round pseudo-label accuracy')
ax1.plot(r_valid, cum_valid, 'o-', color='crimson', label='Cumulative held pseudo-label accuracy')
ax1.set_xlabel('Active Learning Round'); ax1.set_ylabel('Accuracy vs. Ground Truth (%)')
ax1.set_ylim(80,100); ax1.set_xticks(rounds); ax1.legend(loc='lower right', fontsize=8)
ax2 = ax1.twinx(); ax2.bar(rounds, pseudo_count, alpha=0.15, color='gray'); ax2.set_ylabel('Cumulative Pseudo-Labels Held')
plt.title('Pseudo-Label Audit: Accuracy Against Frozen Ground Truth')
plt.tight_layout(); plt.savefig('figs/fig6_pseudo_label_audit.pdf'); plt.close()

print("Done. n_seeds_rand_matched =", n_seeds_rand_matched)
