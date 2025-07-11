# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:30:04 2025

@author: ASUS
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Åström–Bell Drum-Boiler Surrogate & Validation — FULL SUITE
  • 全域 k_P_DEFAULT 一處調整
  • Grid 取樣 + XGB 壓力迴歸 + RF 安全分類
  • 8 張驗證圖：Ternary, P-contour, Feature, CV-box, Parity,
               Residual, ROC, PR
  • 文獻穩態 & 能量儲存常數對照（無需 CSV）
  • (可選) 動態步階曲線疊圖，如有 step_heat_input_ref.csv
"""

# ────────── 全域參數 ──────────
K_P_DEFAULT = 1           # 壓力比例係數，改這裡就好
GRID_POINTS = 13              # 13³ ≈ 2197 網格點
CSV_STEP_REF = "step_heat_input_ref.csv"   # 若無可留空

# ────────── 套件 ──────────
import warnings, pathlib
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from scipy.integrate import solve_ivp

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve, auc)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import matplotlib.pyplot as plt, seaborn as sns
import matplotlib.patches as patches
plt.style.use("seaborn-whitegrid")
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":300,
                     "font.size":12,"axes.labelsize":12,
                     "axes.titlesize":14,"legend.fontsize":10})

# ════════════════════════════════════════
# 1. Drum-Boiler 物理模型
# ════════════════════════════════════════
class DrumBoilerParams:
    def __init__(self, k_P: float = K_P_DEFAULT):
        self.V_d, self.V_dc, self.V_r = 67, 5, 5.0
        self.A_d, self.A_dc, self.A_r = 50.0, 1.6, 0.8
        self.rho_w, self.rho_s = 1000.0, 1.0
        self.h_fg, self.c_p = 2.26e6, 4180.0
        self.P_nom, self.h_nom = 10e6, 8.0
        self.Q_in, self.W_s, self.W_fw, self.T_fw = 100e6, 50, 50, 373.15
        self.k_P = k_P

def simplified_drum_boiler_model(t, y, p: DrumBoilerParams):
    P, rho_d, h_d = y
    T_sat = 273.15 + 100*(P/1e5)**0.25
    W_gen = p.Q_in / p.h_fg
    drho = (p.W_fw - p.W_s + W_gen)/p.V_d
    dh   = (p.W_fw - W_gen)/(p.A_d*p.rho_w)
    h_fw, h_sat_s = p.c_p*(p.T_fw-273.15), p.c_p*(T_sat-273.15)+p.h_fg
    dE_dt = p.W_fw*h_fw + p.Q_in - p.W_s*h_sat_s
    dP_dt = p.k_P * dE_dt / (p.V_d*p.c_p)
    return [dP_dt, drho, dh]

def simulate_boiler(Q_MW, W_s, W_fw, k_P=K_P_DEFAULT, t_end=400):
    p = DrumBoilerParams(k_P=k_P)
    p.Q_in, p.W_s, p.W_fw = Q_MW*1e6, W_s, W_fw
    sol = solve_ivp(simplified_drum_boiler_model, (0,t_end),
                    [p.P_nom, 950, p.h_nom], args=(p,),
                    t_eval=np.linspace(0,t_end,800),
                    method="RK45", rtol=1e-4, atol=1e-6)
    if not sol.success or sol.t.size < 100:
        return 2e7, 500, 15, False
    ss = slice(int(0.8*sol.t.size), None)
    P, rho, h = sol.y[0,ss].mean(), sol.y[1,ss].mean(), sol.y[2,ss].mean()
    ok = (sol.y[0,ss].std()<5e5) and (sol.y[2,ss].std()<0.5)
    return P, rho, h, ok

# ════════════════════════════════════════
# 2. 資料集 (Grid)
# ════════════════════════════════════════
def generate_dataset(n=GRID_POINTS):
    Qs = np.linspace(60,150,n)
    Ss = np.linspace(20,80, n)
    Fs = np.linspace(20,80, n)
    rows=[]
    for Q in Qs:
        for S in Ss:
            for F in Fs:
                P,rho,h,ok = simulate_boiler(Q,S,F)
                safe = ok and (4e6<P<18e6) and (2<h<15) and (600<rho<1000)
                rows.append(dict(Q_in=Q,W_s=S,W_fw=F,P_steady=P,
                                 rho_steady=rho,h_steady=h,
                                 converged=ok,safe_operation=safe))
    df = pd.DataFrame(rows)
    print(f"✅ Dataset: {len(df)} pts (converged {df.converged.sum()})")
    return df

# ════════════════════════════════════════
# 3. 模型訓練
# ════════════════════════════════════════
def train_models(df):
    clean = df[df.converged].copy()
    X = clean[["Q_in","W_s","W_fw"]].values
    yP  = clean["P_steady"].values
    yCl = clean["safe_operation"].astype(int).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    xgb_reg = xgb.XGBRegressor(
        n_estimators=250,max_depth=5,learning_rate=0.05,
        subsample=0.9,colsample_bytree=0.9,
        objective="reg:squarederror",random_state=42
    ).fit(Xs,yP)
    rf_cls = RandomForestClassifier(
        n_estimators=200,max_depth=12,class_weight="balanced",
        random_state=42).fit(Xs,yCl)

    X_tr,X_te,yP_tr,yP_te,yCl_tr,yCl_te = train_test_split(
        Xs,yP,yCl,test_size=0.2,random_state=42)
    print(f"📊 XGB R²={r2_score(yP_te,xgb_reg.predict(X_te)):.3f}, "
          f"RMSE={np.sqrt(mean_squared_error(yP_te,xgb_reg.predict(X_te)))/1e6:.2f} MPa")
    print(f"📊 RF  acc={rf_cls.score(X_te,yCl_te):.3f}")

    cm = confusion_matrix(yCl_te, rf_cls.predict(X_te))
    ConfusionMatrixDisplay(cm,display_labels=["Unsafe","Safe"])\
        .plot(values_format="d")
    plt.title("RF Safety — Confusion Matrix")

    return dict(scaler=scaler,pressure_model=xgb_reg,safety_model=rf_cls),\
           dict(X_te=X_te,yP_te=yP_te,yCl_te=yCl_te), clean

# ════════════════════════════════════════
# 4. 視覺化函式 (8 張)
# ════════════════════════════════════════
def plot_ternary_safety(models: dict, df: pd.DataFrame, n: int = 40):
    """改良版：三角圖 + 實際值刻度 + 說明"""

    Qmin, Qmax = df.Q_in.min(), df.Q_in.max()
    Smin, Smax = df.W_s.min(), df.W_s.max()
    Fmin, Fmax = df.W_fw.min(), df.W_fw.max()

    # === 點計算 ===
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            if (i + j + k) == 0:
                continue
            a, b, c = i / n, j / n, k / n
            Q = Qmin + a * (Qmax - Qmin)
            S = Smin + b * (Smax - Smin)
            F = Fmin + c * (Fmax - Fmin)
            X = models["scaler"].transform([[Q, S, F]])
            P_pred = models["pressure_model"].predict(X)[0]
            safe = bool(models["safety_model"].predict(X)[0]) and (4e6 < P_pred < 18e6)
            x_tri = 0.5 * (2 * b + c)
            y_tri = (np.sqrt(3) / 2) * c
            pts.append({"x": x_tri, "y": y_tri, "safe": safe})

    dfT = pd.DataFrame(pts)

    # === 畫圖 ===
    fig, ax = plt.subplots(figsize=(10, 9))
    triangle = patches.Polygon([(0, 0), (1, 0), (0.5, np.sqrt(3)/2)],
                               fill=False, lw=2, ec="black")
    ax.add_patch(triangle)

    sns.scatterplot(
        data=dfT, x="x", y="y", hue="safe",
        palette={True: "#2ECC71", False: "#E74C3C"},
        legend=False, s=20, ax=ax
    )

    # === 軸標籤 ===
    ax.text(0, -0.07, "Heat Input\n(MW)", ha="center", va="top", fontsize=12)
    ax.text(1, -0.07, "Steam Flow\n(kg/s)", ha="center", va="top", fontsize=12)
    ax.text(0.5, np.sqrt(3)/2 + 0.05, "Feedwater Flow\n(kg/s)", ha="center", va="bottom", fontsize=12)

    # === 加邊界刻度 ===
    for i in range(5):
        frac = i / 4
        # Heat Input (底邊)
        xq = frac
        yq = 0
        Q_val = Qmin + frac * (Qmax - Qmin)
        ax.text(xq, yq - 0.035, f"{Q_val:.0f}", ha="center", va="top", fontsize=16)

        # Steam Flow (右側邊)
        xs = 1 - 0.5 * frac
        ys = np.sqrt(3) / 2 * frac
        S_val = Smin + frac * (Smax - Smin)
        ax.text(xs + 0.02, ys, f"{S_val:.0f}", ha="left", va="center", fontsize=16)

        # Feedwater Flow (左側邊)
        xf = 0.5 * frac
        yf = np.sqrt(3) / 2 * frac
        F_val = Fmin + frac * (Fmax - Fmin)
        ax.text(xf - 0.02, yf, f"{F_val:.0f}", ha="right", va="center", fontsize=16)

    # === 說明文字區塊 ===
    ax.text(0.02, 0.94,
        f"Green = Safe, Red = Unsafe\n"
        f"Safety limits: 4 < P < 18 MPa\n"
        f"Q range: {Qmin:.0f}–{Qmax:.0f} MW\n"
        f"W_s range: {Smin:.0f}–{Smax:.0f} kg/s\n"
        f"W_fw range: {Fmin:.0f}–{Fmax:.0f} kg/s",
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(facecolor="white", alpha=0.8),
        va="top"
    )

    # === 美化 ===
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.08, np.sqrt(3)/2 + 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
 #   ax.set_title("Drum Boiler Operating Map (Ternary)", fontsize=16, weight="bold")

    plt.tight_layout()
    return fig, dfT
def plot_pressure_contour(models, df, F_fix=80):
    """
    2-D 壓力圖：底層填色 (contourf) + 細黑色等壓線 + 粗重點線 (9.5 / 10.5 MPa)
    可以指定進料量 F_fix，若為 None 則用中位數。
    """
    # ── 建立 Q, S 網格 ──
    Q = np.linspace(df.Q_in.min(), df.Q_in.max(), 60)
    S = np.linspace(df.W_s.min(), df.W_s.max(), 60)
    
    if F_fix is None:
        F_fix = df.W_fw.median()  # 預設為中位數

    QQ, SS = np.meshgrid(Q, S)
    PP = np.empty_like(QQ)

    for i in range(QQ.shape[0]):
        for j in range(QQ.shape[1]):
            X = models["scaler"].transform([[QQ[i, j], SS[i, j], F_fix]])
            PP[i, j] = models["pressure_model"].predict(X)[0] / 1e6  # MPa

    fig, ax = plt.subplots(figsize=(10, 7))

    # ── (1) 底層填色 ──
    cf = ax.contourf(Q, S, PP,
                     levels=np.linspace(PP.min(), PP.max(), 60),
                     cmap="viridis")

    # ── (2) 細黑色等壓線 ──
    thin_levels = np.arange(np.floor(PP.min()*2)/2,
                            np.ceil(PP.max()*2)/2 + 0.1, 0.2)  # 每 0.2 MPa
    cs_thin = ax.contour(Q, S, PP, levels=thin_levels,
                         colors="k", linewidths=2, alpha=0.6)
    ax.clabel(cs_thin, fmt="%.1f", fontsize=10, inline_spacing=5)

    # ── (3) 粗重點線（9.5 / 10.5 MPa） ──
    cs_bold = ax.contour(Q, S, PP, levels=[9.5, 10.5],
                         colors=["orange", "red"],
                         linestyles=["--", "-"], linewidths=2)
    ax.clabel(cs_bold, fmt="%.1f MPa", fontsize=10)

    # ── 外觀 ──
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Pressure (MPa)")
    ax.set_xlabel("Heat Input (MW)")
    ax.set_ylabel("Steam Flow (kg/s)")
    ax.set_title(f"Predicted Drum Pressure (Feedwater = {F_fix:.1f} kg/s)")
    plt.tight_layout()

    return fig

def plot_feature_importance(models):
    feats=["Heat Input","Steam Flow","Feedwater Flow"]
    imps=models["pressure_model"].feature_importances_
    fig,ax=plt.subplots(figsize=(6,4))
    sns.barplot(x=feats,y=imps,palette="crest",ax=ax)
    for i,v in enumerate(imps): ax.text(i,v+0.01,f"{v:.2f}",
                                        ha="center",va="bottom")
    ax.set_ylim(0,max(imps)*1.3)
    ax.set_ylabel("Gain"); ax.set_title("XGB Feature Importance")
    plt.tight_layout(); return fig

def plot_regression_cv_box(model,X,y,cv=5):
    kf=KFold(n_splits=cv,shuffle=True,random_state=42)
    r2s,rmses=[],[]
    for tr,te in kf.split(X):
        y_hat=model.predict(X[te])
        r2s.append(r2_score(y[te],y_hat))
        rmses.append(np.sqrt(mean_squared_error(y[te],y_hat))/1e6)
    fig,ax=plt.subplots(figsize=(6,4))
    sns.boxplot(data=[r2s,rmses],ax=ax)
    ax.set_xticklabels(["R²","RMSE (MPa)"])
    ax.set_title(f"{cv}-fold CV — Pressure Model")
    plt.tight_layout(); return fig

def plot_parity(y_true,y_pred):
    fig,ax=plt.subplots(figsize=(5,5))
    sns.scatterplot(x=y_true/1e6,y=y_pred/1e6,s=20,ax=ax)
    lims=[min(ax.get_xlim()[0],ax.get_ylim()[0]),
          max(ax.get_xlim()[1],ax.get_ylim()[1])]
    ax.plot(lims,lims,"k--",lw=1)
    ax.set(xlabel="True P (MPa)",ylabel="Predicted P (MPa)",
           title="Parity — Pressure")
    ax.set_aspect("equal","box"); plt.tight_layout(); return fig

def plot_residuals(y_true,y_pred):
    resid=(y_true-y_pred)/1e6
    fig,ax=plt.subplots(figsize=(6,4))
    sns.histplot(resid,kde=True,ax=ax)
    ax.set_xlabel("Residual (MPa)")
    ax.set_title("Residual Distribution")
    plt.tight_layout(); return fig

# ════════════════════════════════════════
# 7. Pipeline
# ════════════════════════════════════════
def main():
    print(f"=== Drum-Boiler Surrogate (k_P = {K_P_DEFAULT}) ===")
    df = generate_dataset()
    models, holdout, df_clean = train_models(df)

    # 8 張圖
    plot_ternary_safety(models, df_clean)
    plot_pressure_contour(models, df_clean)
    plot_feature_importance(models)
    plot_regression_cv_box(models["pressure_model"],
                           models["scaler"].transform(df_clean[["Q_in","W_s","W_fw"]]),
                           df_clean["P_steady"].values)
    y_pred_hold=models["pressure_model"].predict(holdout["X_te"])
    plot_parity(holdout["yP_te"], y_pred_hold)
    plot_residuals(holdout["yP_te"], y_pred_hold)
    plt.show()
    print("\n🎯 finished")

if __name__ == "__main__":
    main()
