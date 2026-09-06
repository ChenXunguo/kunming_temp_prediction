# -*- coding: utf-8 -*-
"""
项目2主入口：昆明日均气温预测模型构建与优化 全流程
流程：数据加载 → 特征工程 → 时序划分 → 四模型对比 → 网格搜索调优 →
      评估可视化 → 特征重要性 → 模型导出 → 预测接口验证
"""
import sys
import logging
from pathlib import Path

# 确保项目根目录在 sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, BEST_MODEL_PATH,
    FIGURE_DIR, REPORT_DIR, LAG_ORDER, TRAIN_RATIO,
    RANDOM_STATE, RF_PARAM_GRID, CV_SPLITS,
)
from src.data_loader import load_raw_data, basic_eda
from src.feature_engineering import build_features, split_train_test
from src.model_training import train_and_evaluate
from src.model_tuning import tune_random_forest, get_feature_importance
from src.evaluation import (
    compute_metrics, plot_actual_vs_predicted, plot_model_comparison,
    plot_residuals, plot_feature_importance, plot_temperature_overview,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("昆明日均气温预测模型构建与优化 — 全流程启动")
    logger.info("=" * 60)

    # ========== 1. 数据加载与 EDA ==========
    logger.info("\n【1/7】数据加载与基础探查")
    df = load_raw_data(RAW_DATA_PATH)
    eda = basic_eda(df)
    plot_temperature_overview(df, FIGURE_DIR / "01_temperature_overview.png")

    # ========== 2. 特征工程 ==========
    logger.info("\n【2/7】特征工程（时间特征 + 1~7天滞后特征）")
    feat = build_features(df, lag_order=LAG_ORDER)
    feat.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")
    logger.info(f"特征矩阵已保存：{PROCESSED_DATA_PATH}，形状={feat.shape}")
    logger.info(f"特征列（{len([c for c in feat.columns if c not in ('date','temp_mean','temp_mean_next')])}个）："
                f"{[c for c in feat.columns if c not in ('date','temp_mean','temp_mean_next')]}")

    # ========== 3. 时序划分 ==========
    logger.info("\n【3/7】按时间顺序 8:2 划分训练/测试集（严格规避数据泄露）")
    (X_train, X_test, y_train, y_test,
     feature_names, scaler, train_dates, test_dates) = split_train_test(
        feat, train_ratio=TRAIN_RATIO, scale=True
    )

    # ========== 4. 四模型对比 ==========
    logger.info("\n【4/7】四模型对比训练（线性回归 / 决策树 / 随机森林 / GBDT）")
    results_df, predictions, models = train_and_evaluate(
        X_train, X_test, y_train, y_test, random_state=RANDOM_STATE
    )
    plot_model_comparison(results_df, FIGURE_DIR / "02_model_comparison.png")
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # ========== 5. 网格搜索调优（随机森林） ==========
    logger.info("\n【5/7】随机森林网格搜索调优（GridSearchCV + 时序5折交叉验证）")
    tune_result = tune_random_forest(
        X_train, y_train, RF_PARAM_GRID,
        cv_splits=CV_SPLITS, random_state=RANDOM_STATE
    )
    best_model = tune_result["best_model"]
    best_params = tune_result["best_params"]
    best_cv_mae = tune_result["best_cv_mae"]
    tune_result["cv_results_df"].to_csv(
        REPORT_DIR / "gridsearch_cv_results.csv", index=False, encoding="utf-8-sig"
    )
    logger.info(f"最优参数：{best_params}")
    logger.info(f"交叉验证 MAE：{best_cv_mae:.4f}℃")

    # ========== 6. 最优模型测试集评估 + 可视化 ==========
    logger.info("\n【6/7】最优模型测试集评估与可视化")
    y_pred_best = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred_best)
    logger.info(f"测试集评估：MAE={test_metrics['MAE']:.4f}℃, "
                f"RMSE={test_metrics['RMSE']:.4f}℃, R²={test_metrics['R2']:.4f}")

    plot_actual_vs_predicted(
        test_dates, y_test, y_pred_best,
        FIGURE_DIR / "03_actual_vs_predicted_rf.png"
    )
    plot_residuals(y_test, y_pred_best, FIGURE_DIR / "04_residuals_rf.png")

    # 特征重要性
    fi = get_feature_importance(best_model, feature_names)
    fi.to_csv(REPORT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_feature_importance(best_model, feature_names,
                            FIGURE_DIR / "05_feature_importance.png", top_n=15)
    logger.info(f"特征重要性 Top 5：\n{fi.head(5).to_string(index=False)}")

    # ========== 7. 模型导出 ==========
    logger.info("\n【7/7】导出最优模型（pkl）")
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "metadata": {
            "model_type": "RandomForestRegressor(tuned)",
            "best_params": best_params,
            "cv_mae": best_cv_mae,
            "test_mae": test_metrics["MAE"],
            "test_rmse": test_metrics["RMSE"],
            "test_r2": test_metrics["R2"],
            "train_period": f"{pd.Timestamp(train_dates[0]).date()} ~ {pd.Timestamp(train_dates[-1]).date()}",
            "test_period": f"{pd.Timestamp(test_dates[0]).date()} ~ {pd.Timestamp(test_dates[-1]).date()}",
            "lag_order": LAG_ORDER,
            "random_state": RANDOM_STATE,
        },
    }
    joblib.dump(bundle, BEST_MODEL_PATH)
    logger.info(f"模型已导出：{BEST_MODEL_PATH}")

    # ========== 预测接口验证 ==========
    logger.info("\n【验证】加载导出模型并执行一次预测")
    from src.predictor import load_predictor
    predictor = load_predictor(str(BEST_MODEL_PATH))
    # 用测试集最后8天真实气温（含目标当日）做预测验证
    recent_8 = list(y_test[-8:])  # 测试集倒数第8~倒数第1天（8个值，最后一个为目标当日）
    demo_result = predictor.predict(pd.Timestamp(test_dates[-1]), recent_8)
    logger.info(f"预测示例：{demo_result}")
    logger.info(f"实际次日气温（测试集最后一天的次日，即测试集外第一天）：需实际数据验证")

    # ========== 最终总结 ==========
    logger.info("\n" + "=" * 60)
    logger.info("全流程完成！最终结果汇总：")
    logger.info("=" * 60)
    logger.info(f"数据：{eda['total_days']} 天（{eda['date_range']}）")
    logger.info(f"特征：{len(feature_names)} 个（时间特征 + 1~7天滞后）")
    logger.info(f"训练/测试：{len(X_train)}/{len(X_test)} 行（时序8:2）")
    logger.info(f"\n四模型对比（测试集）：")
    for _, row in results_df.iterrows():
        logger.info(f"  {row['model']:<8s} MAE={row['MAE']:.3f}  RMSE={row['RMSE']:.3f}  R²={row['R2']:.3f}")
    logger.info(f"\n最优模型（调优后随机森林）：")
    logger.info(f"  最优参数：{best_params}")
    logger.info(f"  交叉验证 MAE：{best_cv_mae:.3f}℃")
    logger.info(f"  测试集 MAE：{test_metrics['MAE']:.3f}℃")
    logger.info(f"  测试集 RMSE：{test_metrics['RMSE']:.3f}℃")
    logger.info(f"  测试集 R²：{test_metrics['R2']:.3f}")
    logger.info(f"\n产物：")
    logger.info(f"  模型：{BEST_MODEL_PATH}")
    logger.info(f"  图表：{FIGURE_DIR}/")
    logger.info(f"  报告：{REPORT_DIR}/")
    logger.info("=" * 60)

    return {
        "results_df": results_df,
        "best_params": best_params,
        "test_metrics": test_metrics,
        "feature_importance": fi,
    }


if __name__ == "__main__":
    main()
