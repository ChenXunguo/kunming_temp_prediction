# -*- coding: utf-8 -*-
"""
模型对比训练模块
对比：线性回归、决策树、随机森林、GBDT
评估指标：MAE、RMSE、R²
"""
import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

logger = logging.getLogger(__name__)


def get_model_dict(random_state: int = 42) -> dict:
    """返回待对比的模型字典（名称→模型实例）"""
    return {
        "线性回归": LinearRegression(),
        "决策树": DecisionTreeRegressor(random_state=random_state, max_depth=10),
        "随机森林": RandomForestRegressor(
            n_estimators=200, max_depth=None,
            min_samples_leaf=2, random_state=random_state, n_jobs=-1
        ),
        "GBDT": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=random_state
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test,
                        random_state: int = 42) -> tuple:
    """
    训练全部模型并在测试集上评估。
    返回：(results_df, predictions_dict, models_dict)
    """
    from src.evaluation import compute_metrics

    model_dict = get_model_dict(random_state)
    results = []
    predictions = {}
    models = {}

    for name, model in model_dict.items():
        logger.info(f"训练模型：{name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        metrics["model"] = name
        results.append(metrics)
        predictions[name] = y_pred
        models[name] = model
        logger.info(f"{name} → MAE={metrics['MAE']:.3f}, "
                    f"RMSE={metrics['RMSE']:.3f}, R²={metrics['R2']:.3f}")

    results_df = pd.DataFrame(results)[["model", "MAE", "RMSE", "R2"]]
    results_df = results_df.sort_values("MAE").reset_index(drop=True)
    logger.info(f"模型对比结果（按MAE升序）：\n{results_df.to_string(index=False)}")
    return results_df, predictions, models


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.config import RAW_DATA_PATH
    from src.data_loader import load_raw_data
    from src.feature_engineering import build_features, split_train_test

    df = load_raw_data(RAW_DATA_PATH)
    feat = build_features(df)
    X_train, X_test, y_train, y_test, *_ = split_train_test(feat)
    results_df, _, _ = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\n最终对比：")
    print(results_df)
