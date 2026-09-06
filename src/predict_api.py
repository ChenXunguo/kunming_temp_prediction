# -*- coding: utf-8 -*-
"""
轻量级气温预测 Web API（Flask）
- POST /predict：输入目标日期 + 最近7天日均气温，返回次日日均气温预测
- GET  /health：健康检查
- GET  /model_info：模型元信息

可作为独立服务运行，或接入项目1分析系统 / 项目3自研Web框架。
"""
import logging
from flask import Flask, request, jsonify

from src.predictor import load_predictor
from src.config import PREDICT_HOST, PREDICT_PORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = load_predictor()
    return _predictor

@app.route("/")
def index():
    return """
    <h2>昆明日均气温预测API服务</h2>
    <p>健康检查：<a href="/health">/health</a></p>
    <p>模型信息：<a href="/model_info">/model_info</a></p>
    <p>预测接口：POST /predict (需要post json请求)</p>
    """


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "kunming_daily_temp_predictor"})


@app.route("/model_info", methods=["GET"])
def model_info():
    predictor = get_predictor()
    return jsonify({
        "model_type": predictor.metadata.get("model_type", "RandomForest"),
        "feature_count": len(predictor.feature_names),
        "feature_names": predictor.feature_names,
        "test_mae": predictor.metadata.get("test_mae"),
        "test_rmse": predictor.metadata.get("test_rmse"),
        "test_r2": predictor.metadata.get("test_r2"),
        "train_period": predictor.metadata.get("train_period"),
        "test_period": predictor.metadata.get("test_period"),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    请求体 JSON：
    {
        "target_date": "2025-09-06",   // 目标日期（预测该日的次日气温）
        "recent_temps": [18.5, 19.2, 20.1, 19.8, 20.5, 21.0, 21.2, 21.5]  // 最近8天日均气温（含当日，升序）
    }
    """
    try:
        data = request.get_json(force=True)
        target_date = data.get("target_date")
        recent_temps = data.get("recent_temps")

        if not target_date or not recent_temps:
            return jsonify({"error": "缺少 target_date 或 recent_temps 参数"}), 400
        if len(recent_temps) != 8:
            return jsonify({"error": f"recent_temps 需要8个值（含当日），实际 {len(recent_temps)} 个"}), 400

        predictor = get_predictor()
        result = predictor.predict(target_date, recent_temps)
        return jsonify({"ok": True, "result": result})

    except Exception as e:
        logger.exception("预测接口异常")
        return jsonify({"ok": False, "error": str(e)}), 500


def main():
    logger.info(f"启动气温预测 API 服务：http://{PREDICT_HOST}:{PREDICT_PORT}")
    # 预加载模型
    get_predictor()
    try:
        from waitress import serve
        serve(app, host=PREDICT_HOST, port=PREDICT_PORT)
    except ImportError:
        app.run(host=PREDICT_HOST, port=PREDICT_PORT, debug=False)


if __name__ == "__main__":
    main()
