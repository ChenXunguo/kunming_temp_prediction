# -*- coding: utf-8 -*-
"""
气温预测服务模块
- 加载训练好的最优模型（pkl）
- 提供 predict() 接口：输入最近 7 天气温 + 日期，输出次日日均气温预测
- 可作为独立服务，或接入项目1分析系统 / 项目3自研Web框架
"""
import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DailyTempPredictor:
    """昆明次日日均气温预测器"""

    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = bundle["feature_names"]
        self.metadata = bundle.get("metadata", {})
        logger.info(f"模型加载成功：{model_path}，"
                    f"特征数={len(self.feature_names)}，"
                    f"测试集MAE={self.metadata.get('test_mae', 'N/A')}℃")

    @staticmethod
    def _season_of_month(month: int) -> str:
        if month in (3, 4, 5):
            return "spring"
        elif month in (6, 7, 8):
            return "summer"
        elif month in (9, 10, 11):
            return "autumn"
        else:
            return "winter"

    def _build_feature_row(self, target_date, recent_temps: list) -> dict:
        """
        根据目标日期和最近气温构造单行特征。
        recent_temps: [t-7, t-6, ..., t-1, t]（共8个，按时间升序，最后一个为目标当日气温）
        target_date: 要预测"次日"气温的那一天（即 t 日），预测 t+1 日气温
        """
        if len(recent_temps) != 8:
            raise ValueError(f"需要最近8天日均气温（含当日），实际传入 {len(recent_temps)} 个")

        target_date = pd.Timestamp(target_date)
        row = {
            "month": target_date.month,
            "day": target_date.day,
            "dayofyear": target_date.dayofyear,
            "temp_current": recent_temps[-1],  # 当日气温（最核心特征）
        }
        # 季节 one-hot
        season = self._season_of_month(target_date.month)
        for s in ["spring", "summer", "autumn", "winter"]:
            row[f"season_{s}"] = 1 if s == season else 0
        # 滞后特征：lag_1 = 昨天, lag_7 = 7天前
        for i in range(1, 8):
            row[f"lag_{i}"] = recent_temps[7 - i]

        return row

    def predict(self, target_date, recent_temps: list) -> dict:
        """
        预测 target_date 次日的日均气温。
        返回：{"predicted_temp": float, "target_date": str, "next_date": str}
        """
        row = self._build_feature_row(target_date, recent_temps)
        # 按模型训练时的特征顺序排列
        X = np.array([[row[name] for name in self.feature_names]])
        if self.scaler is not None:
            X = self.scaler.transform(X)
        pred = float(self.model.predict(X)[0])

        target_date = pd.Timestamp(target_date)
        next_date = target_date + pd.Timedelta(days=1)
        result = {
            "predicted_temp": round(pred, 2),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "next_date": next_date.strftime("%Y-%m-%d"),
            "unit": "℃",
            "model": self.metadata.get("model_type", "RandomForest"),
        }
        logger.info(f"预测：{next_date.date()} 昆明日均气温 ≈ {pred:.2f}℃")
        return result


def load_predictor(model_path: str = None) -> DailyTempPredictor:
    """便捷加载函数（默认使用项目最优模型路径）"""
    if model_path is None:
        from src.config import BEST_MODEL_PATH
        model_path = str(BEST_MODEL_PATH)
    return DailyTempPredictor(model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    predictor = load_predictor()
    # 示例：最近8天日均气温（含当日）
    demo_temps = [18.5, 19.2, 20.1, 19.8, 20.5, 21.0, 21.2, 21.5]
    result = predictor.predict("2025-09-06", demo_temps)
    print("\n预测结果：", result)
