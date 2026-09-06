# 昆明日均气温预测模型构建与优化（项目 2）

基于昆明 2020-2025 年真实历史气象数据，独立完成日均气温预测模型的构建、对比、调优与部署，作为项目 1 预测模块的深度增强与验证。

## 项目亮点



* **真实数据**：Open-Meteo 历史气象接口，昆明 2020-01-01 \~ 2025-09-05 共 **2075 天**日均气温

* **严格防泄露**：按时间顺序 8:2 划分训练 / 测试集，Scaler 仅 fit 训练集，交叉验证使用 `TimeSeriesSplit`

* **四模型对比**：线性回归 / 决策树 / 随机森林 / GBDT，以 MAE、RMSE、R² 为指标

* **网格搜索调优**：GridSearchCV + 时序 5 折交叉验证，36 组参数 × 5 折 = 180 次拟合

* **可解释性**：随机森林特征重要性分析，量化当日气温、季节因子、滞后特征的贡献

* **落地验证**：最优模型导出 pkl，提供 Flask 轻量预测 API，可独立服务或接入项目 1

## 最终指标（测试集，2024-07-18 \~ 2025-09-04）



| 模型            | MAE (℃)   | RMSE (℃)  | R²        |
| ------------- | --------- | --------- | --------- |
| 线性回归          | 1.040     | 1.533     | 0.879     |
| **随机森林（调优后）** | **1.042** | **1.571** | **0.873** |
| GBDT          | 1.093     | 1.617     | 0.866     |
| 决策树           | 1.361     | 2.153     | 0.762     |

> 优选随机森林：MAE≈1.04℃（稳定控制在 2℃ 以内），R²≈0.87。线性回归在该数据集上略优，但随机森林具备更好的非线性拟合能力和特征可解释性，且两者差距在噪声范围内。

**调优后最优参数**：`n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2`

**特征重要性 Top 5**：



| 排名 | 特征                  | 重要性   | 累计    |
| -- | ------------------- | ----- | ----- |
| 1  | temp\_current（当日气温） | 0.928 | 0.928 |
| 2  | dayofyear（一年中第几天）   | 0.013 | 0.941 |
| 3  | lag\_1（前 1 天气温）     | 0.009 | 0.950 |
| 4  | lag\_4（前 4 天气温）     | 0.009 | 0.959 |
| 5  | lag\_3（前 3 天气温）     | 0.008 | 0.967 |

## 目录结构



```
kunming\_temp\_prediction/

├── data/

│   ├── raw/                          # 原始数据（Open-Meteo CSV）

│   └── processed/                    # 特征矩阵

├── src/

│   ├── config.py                     # 全局配置（路径、超参数）

│   ├── data\_loader.py                # 数据加载与 EDA

│   ├── feature\_engineering.py        # 特征工程（时间特征 + 1\~7天滞后）

│   ├── model\_training.py             # 四模型对比训练

│   ├── model\_tuning.py               # GridSearchCV 调优 + 特征重要性

│   ├── evaluation.py                 # 指标计算与可视化

│   ├── predictor.py                  # 预测器（加载 pkl，单样本/批量预测）

│   └── predict\_api.py                # Flask 轻量预测 API

├── models/

│   └── kunming\_daily\_temp\_rf\_best.pkl  # 最优模型（含 Scaler、特征名、元信息）

├── reports/

│   ├── figures/                      # 5 张评估图表

│   ├── model\_comparison.csv          # 四模型对比结果

│   ├── gridsearch\_cv\_results.csv     # 网格搜索交叉验证明细

│   └── feature\_importance.csv        # 特征重要性排序

├── main.py                           # 全流程入口

├── requirements.txt

└── README.md
```

## 快速开始

### 1. 安装依赖



```
pip install -r requirements.txt
```

### 2. 运行全流程



```
python main.py
```

流程：数据加载 → 特征工程 → 时序划分 → 四模型对比 → 网格搜索调优 → 评估可视化 → 模型导出 → 预测验证。

### 3. 启动预测 API



```
python -m src.predict_api
```

服务默认监听 `0.0.0.0:8900`，提供三个接口：



| 接口            | 方法   | 说明                |
| ------------- | ---- | ----------------- |
| `/health`     | GET  | 健康检查              |
| `/model_info` | GET  | 模型元信息（参数、指标、特征列表） |
| `/predict`    | POST | 气温预测              |

**预测请求示例**：



```
curl -X POST http://127.0.0.1:8900/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{

&#x20;   "target\_date": "2025-09-04",

&#x20;   "recent\_temps": \[17.5, 17.9, 18.3, 18.5, 19.2, 20.1, 20.9, 21.5]

&#x20; }'
```

`recent_temps` 为最近 8 天日均气温（含目标当日，按时间升序），返回次日日均气温预测。

**响应示例**：



```
{

&#x20; "ok": true,

&#x20; "result": {

&#x20;   "predicted\_temp": 21.23,

&#x20;   "target\_date": "2025-09-04",

&#x20;   "next\_date": "2025-09-05",

&#x20;   "unit": "℃",

&#x20;   "model": "RandomForestRegressor(tuned)"

&#x20; }

}
```

## 特征工程说明



| 特征类别 | 特征                                  | 说明                        |
| ---- | ----------------------------------- | ------------------------- |
| 当日气温 | temp\_current                       | 目标当日日均气温（最核心特征，重要性 92.8%） |
| 时间特征 | month, day, dayofyear               | 月、日、一年中第几天                |
| 季节特征 | season\_spring/summer/autumn/winter | 季节 one-hot 编码             |
| 滞后特征 | lag\_1 \~ lag\_7                    | 前 1\~7 天日均气温，捕捉时序依赖       |

**目标变量**：`temp_mean_next`（次日日均气温）

**数据划分**：按时间顺序前 80% 训练（2020-01-08 \~ 2024-07-17，1653 行），后 20% 测试（2024-07-18 \~ 2025-09-04，414 行）。

## 防数据泄露措施



1. **时序划分**：不随机打乱，严格按时间顺序切分

2. **Scaler 隔离**：StandardScaler 仅在训练集上 fit，再 transform 测试集

3. **时序交叉验证**：GridSearchCV 使用 `TimeSeriesSplit(n_splits=5)`，每一折的训练集都在验证集之前，杜绝未来信息泄露

4. **滞后特征仅用历史**：lag 特征通过 `shift()` 构造，不包含未来信息

## 与项目 1 的关系



* 本项目是项目 1 中 `analyzer/prediction_model.py` 的**独立深度增强版**：使用 5 年真实历史数据（而非项目 1 的实时采集小样本），构建更稳健的日均气温预测模型

* 最优模型 pkl 可直接接入项目 1 的分析系统，替换或增强项目 1 的预测模块

* 预测 API 可作为独立微服务，供项目 1 或其他系统调用

## 数据来源

[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) — 免费、无需 API Key，提供昆明（25.04°N, 102.72°E）2020 年以来的逐日气象再分析数据。