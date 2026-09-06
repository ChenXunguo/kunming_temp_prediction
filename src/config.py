# -*- coding: utf-8 -*-
"""项目2：昆明日均气温预测模型 — 全局配置"""
from pathlib import Path

# 项目根目录（config.py 在 src/ 下，取上两级）
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据路径
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "kunming_daily_weather_2020_2025.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "features.csv"

# 模型与报告输出
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 最终模型导出文件名
BEST_MODEL_PATH = MODEL_DIR / "kunming_daily_temp_rf_best.pkl"

# ==================== 实验参数 ====================
# 目标列：预测"次日"日均气温
TARGET_COL = "temp_mean_next"
# 原始日均气温列（用于构造滞后特征）
TEMP_COL = "temperature_2m_mean"

# 滞后特征阶数：1~7天
LAG_ORDER = 7
# 训练/测试集按时间顺序 8:2 划分
TRAIN_RATIO = 0.8

# 随机种子（保证可复现）
RANDOM_STATE = 42

# 网格搜索超参数空间（随机森林）
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
# 时序交叉验证折数
CV_SPLITS = 5

# 预测服务配置
PREDICT_HOST = "0.0.0.0"
PREDICT_PORT = 8900
