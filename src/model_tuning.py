# -*- coding: utf-8 -*-
"""
模型调优与可解释性模块
- 使用 GridSearchCV + 时序 5 折交叉验证（TimeSeriesSplit）优化随机森林超参数
- 严格时序切分，杜绝数据泄露
- 输出最优模型、最优参数、交叉验证结果、特征重要性
"""
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)


def tune_random_forest(X_train, y_train, param_grid: dict,
                       cv_splits: int = 5, random_state: int = 42,
                       n_jobs: int = -1) -> dict:
    """
    对随机森林执行网格搜索 + 时序交叉验证。

    关键：使用 TimeSeriesSplit 而非普通 KFold——
    普通 KFold 会随机切分，训练折可能包含未来数据，造成数据泄露；
    TimeSeriesSplit 保证每一折的训练集都在验证集之前。

    返回字典：best_model, best_params, best_score(MAE), cv_results_df
    """
    base_model = RandomForestRegressor(random_state=random_state)
    # 时序交叉验证切分器
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",  # 以 MAE 为优化目标
        cv=tscv,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,  # 用最优参数在全部训练集上重新拟合
    )

    n_combos = int(np.prod([len(v) for v in param_grid.values()]))
    logger.info(f"开始网格搜索：参数组合数={n_combos}，"
                f"时序 {cv_splits} 折交叉验证")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    # neg_mean_absolute_error → 取负得 MAE
    best_cv_mae = -grid.best_score_

    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results = cv_results.sort_values("rank_test_score").reset_index(drop=True)
    # 转为正的 MAE 便于阅读
    cv_results["mean_test_MAE"] = -cv_results["mean_test_score"]

    logger.info(f"网格搜索完成：最优参数={best_params}，"
                f"交叉验证 MAE={best_cv_mae:.4f}℃")

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_cv_mae": round(best_cv_mae, 4),
        "cv_results_df": cv_results,
    }


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    """从最优随机森林提取特征重要性，按重要性降序"""
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    fi["cumulative_importance"] = fi["importance"].cumsum()
    return fi


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.config import RAW_DATA_PATH, RF_PARAM_GRID, CV_SPLITS
    from src.data_loader import load_raw_data
    from src.feature_engineering import build_features, split_train_test

    df = load_raw_data(RAW_DATA_PATH)
    feat = build_features(df)
    X_train, X_test, y_train, y_test, feature_names, scaler, _, _ = split_train_test(feat)
    result = tune_random_forest(X_train, y_train, RF_PARAM_GRID, CV_SPLITS)
    print("\n最优参数：", result["best_params"])
    print("交叉验证 MAE：", result["best_cv_mae"])
    fi = get_feature_importance(result["best_model"], feature_names)
    print("\n特征重要性 Top 10：")
    print(fi.head(10).to_string(index=False))
