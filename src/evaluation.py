# -*- coding: utf-8 -*-
"""
评估与可视化模块
- 指标计算：MAE、RMSE、R²
- 图表：实际vs预测时序、模型对比柱状图、残差图、特征重要性图、气温总览图
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 中文字体（Windows）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def compute_metrics(y_true, y_pred) -> dict:
    """计算 MAE / RMSE / R²"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def plot_actual_vs_predicted(dates, y_true, y_pred, save_path,
                              title="昆明日均气温：实际值 vs 预测值（测试集）"):
    """实际值与预测值时序对比图"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_true, label="实际值", color="#2E86AB", linewidth=1.2, alpha=0.85)
    ax.plot(dates, y_pred, label="预测值", color="#E63946", linewidth=1.0, alpha=0.8, linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("日均气温 (℃)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results_df, save_path):
    """四模型 MAE / RMSE / R² 对比柱状图（双子图）"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#A8DADC", "#457B9D", "#1D3557", "#E63946"]
    models = results_df["model"].tolist()

    # 左：MAE / RMSE
    x = np.arange(len(models))
    w = 0.35
    axes[0].bar(x - w / 2, results_df["MAE"], w, label="MAE", color="#457B9D")
    axes[0].bar(x + w / 2, results_df["RMSE"], w, label="RMSE", color="#E63946")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("误差 (℃)")
    axes[0].set_title("MAE / RMSE 对比（越低越好）", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(results_df["MAE"]):
        axes[0].text(i - w / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(results_df["RMSE"]):
        axes[0].text(i + w / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)

    # 右：R²
    bars = axes[1].bar(models, results_df["R2"], color=colors)
    axes[1].set_ylabel("R²")
    axes[1].set_title("R² 决定系数对比（越高越好）", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, results_df["R2"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                      f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("四种回归模型预测性能对比", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true, y_pred, save_path):
    """残差分布图（直方图 + 残差时序）"""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(residuals, bins=40, color="#457B9D", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="#E63946", linestyle="--", linewidth=1.5)
    axes[0].set_title(f"残差分布（均值={residuals.mean():.3f}℃，标准差={residuals.std():.3f}℃）",
                       fontsize=12, fontweight="bold")
    axes[0].set_xlabel("残差 (℃)")
    axes[0].set_ylabel("频数")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.4, s=12, color="#1D3557")
    axes[1].axhline(0, color="#E63946", linestyle="--", linewidth=1.5)
    axes[1].set_title("残差 vs 预测值（检查异方差）", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("预测值 (℃)")
    axes[1].set_ylabel("残差 (℃)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, feature_names, save_path, top_n: int = 15):
    """特征重要性水平柱状图（取 Top N）"""
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    bars = ax.barh(fi["feature"], fi["importance"], color="#2A9D8F")
    ax.set_title(f"随机森林特征重要性 Top {top_n}", fontsize=14, fontweight="bold")
    ax.set_xlabel("重要性（归一化）")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, v in zip(bars, fi["importance"]):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fi.sort_values("importance", ascending=False)


def plot_temperature_overview(df, save_path):
    """全量日均气温时序总览图"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["temp_mean"], color="#2A9D8F", linewidth=0.8, alpha=0.8)
    ax.set_title("昆明日均气温时序总览（2020-01 ~ 2025-09）", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("日均气温 (℃)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
