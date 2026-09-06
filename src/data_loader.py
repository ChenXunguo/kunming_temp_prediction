# -*- coding: utf-8 -*-
"""数据加载与基础探查"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_raw_data(csv_path) -> pd.DataFrame:
    """
    加载昆明历史日均气象数据 CSV。
    列：time, temperature_2m_mean, temperature_2m_max, temperature_2m_min, precipitation_sum
    """
    df = pd.read_csv(csv_path)
    # 时间列转 datetime 并排序
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    # 重命名为更简洁的列名
    df = df.rename(columns={
        "time": "date",
        "temperature_2m_mean": "temp_mean",
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "precipitation",
    })
    logger.info(f"加载数据：{len(df)} 行，时间范围 "
                f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def basic_eda(df: pd.DataFrame) -> dict:
    """基础统计探查，返回关键指标字典"""
    stats = {
        "total_days": len(df),
        "date_range": f"{df['date'].min().date()} ~ {df['date'].max().date()}",
        "temp_mean_avg": round(df["temp_mean"].mean(), 2),
        "temp_mean_min": round(df["temp_mean"].min(), 2),
        "temp_mean_max": round(df["temp_mean"].max(), 2),
        "temp_mean_std": round(df["temp_mean"].std(), 2),
        "missing_temp_mean": int(df["temp_mean"].isna().sum()),
    }
    logger.info(f"EDA: {stats}")
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.config import RAW_DATA_PATH
    df = load_raw_data(RAW_DATA_PATH)
    print(basic_eda(df))
    print(df.head())
