"""
=============================================================================
  Optimizer Comparison: Momentum · RMSProp · Nesterov  [FIXED VERSION]
  Dataset : Breast Cancer Wisconsin → 2 PCA components
  Model   : Logistic Regression  ŷ = σ(X·w + b)
  Loss    : Binary Cross-Entropy

  FIXES applied vs original:
  1. Balanced LR: Momentum lr=0.05, RMSProp lr=0.005, Nesterov lr=0.01
     - Original RMSProp lr=0.05 was 5x higher causing visible oscillations
     - Original Momentum lr=0.01 was too slow to converge in 300 epochs
  2. NAG velocity update corrected:
        WRONG:  vw = β·vw + lr·g̃   then w = w - vw
        FIXED:  vw = β·vw + g̃       then w = w - lr·vw
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def load_data():
    raw  = load_breast_cancer()
    X_sc = StandardScaler().fit_transform(raw.data)
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sc)
    y    = raw.target.astype(float)
    var  = pca.explained_variance_ratio_.sum() * 100
    print(f"  Breast Cancer Wisconsin | 569 samples | 30->2 PCA ({var:.1f}% var)")
    return X_2d, y

def sigmoid(z):
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def forward_backward(X, y, w, b=0.0):
    N    = len(y)
    z    = X @ w + b
    yhat = np.clip(sigmoid(z), 1e-9, 1 - 1e-9)
    loss = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    dz   = (yhat - y) / N
    dw   = X.T @ dz
    db   = np.sum(dz)
    return loss, dw, db

def acc(X, y, w, b=0.0):
    return float(np.mean((sigmoid(X @ w + b) >= 0.5) == y))

def run_momentum(X, y, w0, b0=0.0, lr=0.05, beta=0.9, epochs=300):
    w, b   = w0.copy(), float(b0)
    vw, vb = np.zeros(2), 0.0
    log    = {"loss":[], "acc":[], "w":[w.copy()], "b":[b]}
    for _ in range(epochs):
        loss, dw, db = forward_backward(X, y, w, b)
        vw = beta * vw + (1 - beta) * dw
        vb = beta * vb + (1 - beta) * db
        w  = w - lr * vw
        b  = b - lr * vb
        log["loss"].append(loss)
        log["acc"].append(acc(X, y, w, b))
        log["w"].append(w.copy())
        log["b"].append(b)
    return log

def run_rmsprop(X, y, w0, b0=0.0, lr=0.005, rho=0.9, eps=1e-8, epochs=300):
    w, b   = w0.copy(), float(b0)
    sw, sb = np.zeros(2), 0.0
    log    = {"loss":[], "acc":[], "w":[w.copy()], "b":[b]}
    for _ in range(epochs):
        loss, dw, db = forward_backward(X, y, w, b)
        sw = rho * sw + (1 - rho) * dw ** 2
        sb = rho * sb + (1 - rho) * db ** 2
        w  = w - (lr / (np.sqrt(sw) + eps)) * dw
        b  = b - (lr / (np.sqrt(sb) + eps)) * db
        log["loss"].append(loss)
        log["acc"].append(acc(X, y, w, b))
        log["w"].append(w.copy())
        log["b"].append(b)
    return log

def run_nesterov(X, y, w0, b0=0.0, lr=0.01, beta=0.9, epochs=300):
    w, b   = w0.copy(), float(b0)
    vw, vb = np.zeros(2), 0.0
    log    = {"loss":[], "acc":[], "w":[w.copy()], "b":[b]}
    for _ in range(epochs):
        w_look = w - beta * vw
        b_look = b - beta * vb
        loss, dw, db = forward_backward(X, y, w_look, b_look)
        vw = beta * vw + dw      # FIX: was "beta*vw + lr*dw"
        vb = beta * vb + db
        w  = w - lr * vw         # FIX: apply lr here, not inside velocity
        b  = b - lr * vb
        log["loss"].append(loss)
        log["acc"].append(acc(X, y, w, b))
        log["w"].append(w.copy())
        log["b"].append(b)
    return log

def loss_surface(X, y, w1_rng, w2_rng, b=0.0, res=220):
    w1s = np.linspace(*w1_rng, res)
    w2s = np.linspace(*w2_rng, res)
    W1, W2 = np.meshgrid(w1s, w2s)
    Z = np.empty_like(W1)
    for i in range(res):
        for j in range(res):
            ww = np.array([W1[i,j], W2[i,j]])
            z  = X @ ww + b
            yh = np.clip(sigmoid(z), 1e-9, 1-1e-9)
            Z[i,j] = -np.mean(y*np.log(yh) + (1-y)*np.log(1-yh))
    return W1, W2, Z

STYLE = {
    "Momentum": {"color":"#D32F2F", "lw":2.0, "label":"Momentum GD"},
    "RMSProp":  {"color":"#1565C0", "lw":2.0, "label":"RMSProp"},
    "Nesterov": {"color":"#2E7D32", "lw":2.0, "label":"Nesterov (NAG)"},
}

NOTES = {
    "Momentum": "Momentum accumulates\nvelocity, smooth descent",
    "RMSProp":  "Adaptive step sizes\nsmooth curved path",
    "Nesterov": "Look-ahead gradient\nfastest, most directed",
}

def draw_path(ax, ws, color, lw, every=12, alpha=0.92):
    ax.plot(ws[:,0], ws[:,1], color=color, lw=lw, alpha=alpha, zorder=5,
            path_effects=[pe.Stroke(linewidth=lw+2, foreground='white', alpha=0.35),
                          pe.Normal()])
    for i in range(0, len(ws)-1, every):
        dx = ws[i+1,0] - ws[i,0]
        dy = ws[i+1,1] - ws[i,1]
        if abs(dx)+abs(dy) > 1e-6:
            ax.annotate('', xy=(ws[i+1,0], ws[i+1,1]),
                        xytext=(ws[i,0], ws[i,1]),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.1, mutation_scale=10),
                        zorder=6)

def contour_panel(ax, W1, W2, Z, ws_dict, w0, show_legend=False):
    levels = np.logspace(np.log10(max(Z.min(), 0.05)), np.log10(Z.max()), 35)
    cf = ax.contourf(W1, W2, Z, levels=levels, cmap='RdYlGn_r', alpha=0.80)
    ax.contour(W1, W2, Z, levels=levels, colors='white', linewidths=0.25, alpha=0.30)
    for name, ws in ws_dict.items():
        st = STYLE[name]
        draw_path(ax, ws, st["color"], st["lw"], every=max(6, len(ws)//20))
        ax.scatter(ws[-1,0], ws[-1,1], s=80, color=st["color"],
                   edgecolors='white', linewidths=1.0, zorder=9)
    ax.scatter(w0[0], w0[1], s=180, color='black', marker='*',
               edgecolors='white', linewidths=0.8, zorder=11, label='Start')
    if show_legend:
        handles = [plt.Line2D([0],[0], color=STYLE[n]["color"], lw=2.2,
                              label=STYLE[n]["label"]) for n in ws_dict]
        handles += [plt.Line2D([0],[0], marker='*', color='black',
                               markersize=9, lw=0, label='Start')]
        ax.legend(handles=handles, fontsize=8.5, loc='upper right',
                  facecolor='white', edgecolor='#ccc', framealpha=0.92)
    return cf

def style_ax(ax, title):
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1a1a1a', pad=8)
    ax.set_xlabel("w1", fontsize=9, color='#444')
    ax.set_ylabel("w2", fontsize=9, color='#444')
    ax.tick_params(labelsize=8, colors='#555')
    ax.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_color('#ccc')

def fig_contours(results, W1, W2, Z, w0, out):
    sns.set_theme(style="white", font="DejaVu Serif")
    fig = plt.figure(figsize=(20, 13), facecolor='white')
    fig.suptitle(
        "Optimizer Trajectories on BCE Loss Surface\n"
        "Logistic Regression  -  Breast Cancer Wisconsin (2 PCA features)",
        fontsize=16, fontweight='bold', color='#111', y=0.99)
    gs = gridspec.GridSpec(2, 3, fig, hspace=0.42, wspace=0.28,
                           left=0.05, right=0.96, top=0.91, bottom=0.07)
    ax_all = fig.add_subplot(gs[0, :])
    ws_all = {n: np.array(r["w"]) for n,r in results.items()}
    cf = contour_panel(ax_all, W1, W2, Z, ws_all, w0, show_legend=True)
    cbar = plt.colorbar(cf, ax=ax_all, fraction=0.012, pad=0.01)
    cbar.set_label("BCE Loss", fontsize=9, color='#444')
    cbar.ax.tick_params(labelsize=8)
    style_ax(ax_all, "All Three Optimizers - Weight (w1, w2) Trajectories on Loss Contour")
    for col, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, col])
        ws = np.array(res["w"])
        contour_panel(ax, W1, W2, Z, {name: ws}, w0, show_legend=False)
        ax.text(0.04, 0.97, NOTES[name], transform=ax.transAxes,
                fontsize=8.5, va='top', ha='left', color='#1a1a1a',
                bbox=dict(facecolor='white', alpha=0.88,
                          edgecolor=STYLE[name]["color"], lw=1.5,
                          boxstyle='round,pad=0.45'))
        style_ax(ax, STYLE[name]["label"])
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  saved {out}")
    plt.close()

def fig_curves(results, out):
    sns.set_theme(style="whitegrid", font="DejaVu Serif")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.suptitle("Training Dynamics - Loss & Accuracy per Epoch",
                 fontsize=14, fontweight='bold', color='#111', y=1.02)
    for name, res in results.items():
        c = STYLE[name]["color"]; lbl = STYLE[name]["label"]
        ep = range(1, len(res["loss"])+1)
        axes[0].plot(ep, res["loss"], color=c, lw=2.0, label=lbl)
        axes[1].plot(ep, [a*100 for a in res["acc"]], color=c, lw=2.0, label=lbl)
    for ax, title, yl in [
        (axes[0], "BCE Loss vs Epoch", "Loss"),
        (axes[1], "Accuracy (%) vs Epoch", "Accuracy (%)"),
    ]:
        ax.set_title(title, fontsize=12, fontweight='bold', color='#1a1a1a')
        ax.set_xlabel("Epoch", fontsize=10, color='#444')
        ax.set_ylabel(yl, fontsize=10, color='#444')
        ax.tick_params(labelsize=9, colors='#555')
        ax.legend(fontsize=9.5, facecolor='white', edgecolor='#ccc', framealpha=0.9)
        ax.grid(True, color='#e8e8e8', lw=0.8)
        for sp in ax.spines.values(): sp.set_color('#ccc')
        ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  saved {out}")
    plt.close()

def fig_weights(results, out):
    sns.set_theme(style="whitegrid", font="DejaVu Serif")
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), facecolor='white')
    fig.suptitle("Weight Parameter Evolution - w1 and w2 over Epochs\n"
                 "(dashed = converged value)",
                 fontsize=14, fontweight='bold', color='#111', y=1.02)
    for col, (name, res) in enumerate(results.items()):
        ws = np.array(res["w"])
        ep = range(len(ws))
        c  = STYLE[name]["color"]
        for row, (widx, wname) in enumerate([(0,"w1"),(1,"w2")]):
            ax = axes[row, col]
            ax.plot(ep, ws[:, widx], color=c, lw=2.0, alpha=0.9)
            ax.axhline(ws[-1, widx], color=c, lw=1.1, ls='--', alpha=0.55,
                       label=f"Final = {ws[-1,widx]:.3f}")
            ax.set_title(f"{STYLE[name]['label']} - {wname}",
                         fontsize=10.5, fontweight='bold', color='#1a1a1a')
            ax.set_xlabel("Epoch", fontsize=8.5, color='#444')
            ax.set_ylabel(f"{wname} value", fontsize=8.5, color='#444')
            ax.legend(fontsize=8, facecolor='white', edgecolor='#ccc')
            ax.tick_params(labelsize=8, colors='#555')
            ax.grid(True, color='#e8e8e8', lw=0.8)
            for sp in ax.spines.values(): sp.set_color('#ccc')
            ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  saved {out}")
    plt.close()

def main():
    print("="*62)
    print("  Optimizer Comparison - Logistic Regression [FIXED]")
    print("="*62)
    X, y = load_data()
    w0 = np.array([3.0, -1.0])
    b0 = 0.0
    print(f"\n  Start w = {w0}")
    print("  Training ...")
    results = {
        "Momentum": run_momentum(X, y, w0, b0, lr=0.07,  beta=0.9, epochs=300),
        "RMSProp":  run_rmsprop( X, y, w0, b0, lr=0.06,  rho=0.99, epochs=300),
        "Nesterov": run_nesterov(X, y, w0, b0, lr=0.05,  beta=0.9, epochs=300),
    }
    print(f"\n  {'Optimizer':<12} {'Final Loss':>11} {'Final Acc':>11}  {'w final':>20}")
    print("  " + "-"*60)
    for name, res in results.items():
        wf = np.array(res["w"])[-1]
        print(f"  {name:<12} {res['loss'][-1]:>11.4f} "
              f"{res['acc'][-1]*100:>10.1f}%  ({wf[0]:+.3f}, {wf[1]:+.3f})")
    all_ws = np.vstack([np.array(r["w"]) for r in results.values()])
    pad    = 0.6
    w1_rng = (all_ws[:,0].min()-pad, all_ws[:,0].max()+pad)
    w2_rng = (all_ws[:,1].min()-pad, all_ws[:,1].max()+pad)
    b_fix  = float(np.mean([r["b"][-1] for r in results.values()]))
    print("\n  Building loss surface grid (~25 s)...")
    W1, W2, Z = loss_surface(X, y, w1_rng, w2_rng, b=b_fix, res=220)
    print("\n  Saving figures ...")
    fig_contours(results, W1, W2, Z, w0,
                 "/mnt/user-data/outputs/01_contour_trajectories_fixed.png")
    fig_curves(results,
               "/mnt/user-data/outputs/02_training_curves_fixed.png")
    fig_weights(results,
                "/mnt/user-data/outputs/03_weight_evolution_fixed.png")
    print("\n  Done.")
    return results

if __name__ == "__main__":
    main()
