# -*- coding: utf-8 -*-
"""
特征工程模块
- 时间特征：月、日、季节（one-hot）、一年中第几天
- 滞后特征：日均气温 lag_1 ~ lag_7（捕捉时序依赖）
- 目标变量：次日日均气温 temp_mean_next
- 按时间顺序 8:2 划分训练/测试集，严格规避数据泄露
"""
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _season_of_month(month: int) -> str:
    """月份 → 季节（昆明北半球季节划分）"""
    if month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    elif month in (9, 10, 11):
        return "autumn"
    else:
        return "winter"


def build_features(df: pd.DataFrame, temp_col: str = "temp_mean",
                   lag_order: int = 7) -> pd.DataFrame:
    """
    从原始日数据构造特征矩阵。
    注意：滞后特征仅使用历史数据（shift），不存在未来信息泄露。
    """
    feat = df[["date", temp_col]].copy()

    # ---- 时间特征 ----
    feat["month"] = feat["date"].dt.month
    feat["day"] = feat["date"].dt.day
    feat["dayofyear"] = feat["date"].dt.dayofyear
    feat["season"] = feat["month"].apply(_season_of_month)
    # 季节 one-hot 编码（对线性回归友好）
    season_dummies = pd.get_dummies(feat["season"], prefix="season").astype(int)
    feat = pd.concat([feat, season_dummies], axis=1)
    feat = feat.drop(columns=["season"])

    # ---- 目标变量：次日日均气温（先于重命名创建，避免列名找不到） ----
    feat["temp_mean_next"] = feat[temp_col].shift(-1)

    # ---- 多阶温度滞后特征 lag_1 ~ lag_N ----
    # 注意：temp_mean（当日气温）作为 temp_current 保留，是预测次日气温最核心的特征
    feat = feat.rename(columns={temp_col: "temp_current"})
    for i in range(1, lag_order + 1):
        feat[f"lag_{i}"] = feat["temp_current"].shift(i)

    # 丢弃因滞后/目标产生的 NaN 行（首尾各若干行）
    before = len(feat)
    feat = feat.dropna().reset_index(drop=True)
    logger.info(f"特征构造完成：{before} → {len(feat)} 行（丢弃首尾 NaN）")

    return feat


def split_train_test(feat: pd.DataFrame, target_col: str = "temp_mean_next",
                     train_ratio: float = 0.8, scale: bool = True):
    """
    按时间顺序划分训练/测试集（前 80% 训练，后 20% 测试），
    严禁随机打乱（时序数据打乱会造成数据泄露）。

    返回：X_train, X_test, y_train, y_test, feature_names, scaler(或None), train_dates, test_dates
    """
    feature_cols = [c for c in feat.columns
                    if c not in ("date", target_col)]
    X = feat[feature_cols].values
    y = feat[target_col].values
    dates = feat["date"].values

    split_idx = int(len(feat) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    train_dates, test_dates = dates[:split_idx], dates[split_idx:]

    logger.info(f"时序划分：训练 {len(X_train)} 行（{pd.Timestamp(train_dates[0]).date()} ~ "
                f"{pd.Timestamp(train_dates[-1]).date()}），"
                f"测试 {len(X_test)} 行（{pd.Timestamp(test_dates[0]).date()} ~ "
                f"{pd.Timestamp(test_dates[-1]).date()}）")

    scaler = None
    if scale:
        # 关键：Scaler 仅在训练集上 fit，再 transform 测试集，杜绝泄露
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("特征已标准化（StandardScaler 仅 fit 训练集）")

    return (X_train, X_test, y_train, y_test,
            feature_cols, scaler, train_dates, test_dates)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.config import RAW_DATA_PATH
    from src.data_loader import load_raw_data
    df = load_raw_data(RAW_DATA_PATH)
    feat = build_features(df)
    print("特征列：", list(feat.columns))
    print(feat.head(3))
    print(feat.tail(3))
