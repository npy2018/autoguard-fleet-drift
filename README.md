# AutoGuard Fleet Drift

面向OTA灰度车队的**上下文行为指纹、贝叶斯在线变点检测与自适应Conformal告警**。

## 为什么单独做这个仓库

车队级异常不是单车规则问题：新版本样本少、城市和天气分布不同、正常行为会随季节漂移。本仓库专注解决小样本、冷启动与概念漂移。

## 核心方法

1. 按车型、城市、天气、道路建立行为指纹；
2. 使用层级收缩，将小样本上下文拉向成熟全局先验；
3. BOCPD输出在线变点概率；
4. 自适应Conformal p-value提供可解释异常度；
5. 前20个同上下文样本默认Shadow Mode；
6. 可选Chronos-Bolt零样本预测后端，默认实现不需要下载大模型。

## 快速运行

```bash
pip install -e '.[dev]'
python scripts/run_demo.py
uvicorn app:app --reload
```

可选时序基础模型：

```bash
pip install -e '.[foundation]'
```

## 典型输出

- context-aware expected mean/scale；
- z-score；
- conformal p-value；
- changepoint probability；
- drift probability；
- `shadow-mode / monitor / alert` 状态。

## 工程边界

任何固定样本量或统计功效都必须结合真实事件率和分层场景重新计算；本仓库不承诺“200个样本必然足够”。详见 [SOURCES.md](SOURCES.md)。
