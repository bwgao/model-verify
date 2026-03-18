# Model Verify

**[English Documentation](./README.md)**

LLM 模型验证系统 - 检测第三方 API 中转商是否提供真实模型。

## 背景

第三方 LLM API 中转商可能以更便宜的价格声称提供 Claude、GPT-4 等模型，但实际可能：
- 用小模型冒充大模型
- 用量化版本冒充全精度模型
- 完全替换成其他模型

本项目基于多维度验证方法，帮助你检测这些欺骗行为。

## 验证层次

### Layer 1: 身份探测 (Identity Probing)
直接询问模型身份，匹配已知模式。

### Layer 2: 行为指纹 (Behavioral Fingerprinting)
LLMmap 风格的主动指纹识别，通过主题查询分析响应特征。

### Layer 3: 能力基准 (Capability Benchmarking)
测试数学推理、代码生成、知识问答等能力，与已知基线对比。包含40道题目，其中20道高难度题目用于区分模型层级。

### Layer 4: Logprob 分析
对比 token 概率分布（当 API 支持时），计算 KL/JS 散度。

### Layer 5: 延迟指纹 (Latency Fingerprinting)
测量 TTFT、TPS 等指标，与预期性能对比。

### Layer 6: 层级签名 (Tier Signature)
通过行为特征分析区分同一家族不同层级的模型（如 Opus vs Sonnet vs Haiku）。包含4个维度：
- **详细程度**：回答的深度和完整性
- **代码风格**：代码注释、结构、命名习惯
- **推理深度**：逻辑链条的复杂度
- **拒绝模式**：对敏感问题的处理方式

### Layer 7: 模型对比 (Comparison)
A/B 盲测对比，直接比较两个模型的响应差异。适用于有官方模型作为参照的场景。

## 安装

```bash
cd model_verify
source venv/bin/activate
pip install -r requirements.txt
```

## 配置

1. 复制并编辑 provider 配置：
```bash
cp config/providers.yaml config/providers.local.yaml
```

2. 设置 API 密钥环境变量：
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 使用

### 验证模型
```bash
# 完整验证（所有探针）
python main.py verify test_reseller claude-opus-4-6

# 指定探针验证
python main.py verify test_reseller claude-opus-4-6 --probes identity,fingerprint,benchmark

# 层级签名测试（区分 Opus/Sonnet/Haiku）
python main.py verify test_reseller claude-opus-4-6 --probes tier_signature
```

### 模型对比
```bash
# A/B 盲测对比两个模型
python main.py compare test_reseller claude-opus-4-6 official_provider claude-opus-4-6
```

### 收集基线
```bash
python main.py baseline gpt-4o --provider openai_official
```

### 查看报告
```bash
python main.py report example_reseller
python main.py report example_reseller claude-opus-4-20250514 --format json
```

### 持续监控
```bash
python main.py monitor example_reseller gpt-4o --interval 300
```

## 项目结构

```
model_verify/
├── config/              # YAML 配置
│   ├── providers.yaml   # API 提供商配置
│   └── models.yaml      # 模型特征配置
├── baselines/           # 基线数据存储
├── results/             # 验证结果存储
├── probes/              # 验证探针模块
│   ├── identity.py      # 身份探测
│   ├── fingerprint.py   # 行为指纹
│   ├── benchmark.py     # 能力基准（40题含20道高难度）
│   ├── logprob.py       # Logprob 分析
│   ├── latency.py       # 延迟指纹
│   ├── tier_signature.py # 层级签名探测
│   └── comparison.py    # A/B 对比探测
├── analysis/            # 分析模块
│   ├── mmd_test.py      # MMD 统计检验
│   ├── scoring.py       # 评分聚合
│   └── report.py        # 报告生成
├── utils/               # 工具模块
│   ├── types.py         # 类型定义
│   ├── config_loader.py # 配置加载
│   ├── data_store.py    # 数据存储
│   └── api_client.py    # API 客户端
└── main.py              # CLI 入口
```

## 层级区分方法

本系统支持区分同一家族不同层级的模型（如 Claude Opus 4.6 vs Sonnet 4.6 vs Haiku）：

### 1. 硬基准测试
20道高难度题目，覆盖：
- 高等数学（5题）
- 复杂代码（5题）
- 深度推理（5题）
- 科学问题（5题）

### 2. 行为签名
4维度特征分析：
- **详细程度**：Opus 回答更详细，Haiku 更简洁
- **代码风格**：Opus 注释更多，结构更清晰
- **推理深度**：Opus 逻辑链更长，考虑更全面
- **拒绝模式**：不同层级对敏感问题的处理方式不同

### 3. A/B 对比
与官方模型进行盲测对比，评估响应相似度。

## 支持的模型

| 模型家族 | 层级 | 基准分数 |
|---------|------|---------|
| Claude 4.6 | Opus | 硬基准 > 70% |
| Claude 4.6 | Sonnet | 硬基准 40-70% |
| Claude 4.6 | Haiku | 硬基准 < 40% |

## 评分

| 分数 | 判定 | 含义 |
|------|------|------|
| ≥0.8 | PASS | 模型真实可信 |
| 0.5-0.8 | WARN | 存在可疑迹象 |
| <0.5 | FAIL | 模型可能被替换 |

## 参考文献

- *LLMmap: Fingerprinting for Large Language Models* (USENIX Security 2025)
- *Model Equality Testing: Which Model Is This API Serving?* (Stanford 2024)
- *Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs* (UC Berkeley 2025)
- *Real Money, Fake Models: Deceptive Model Claims in Shadow APIs* (CISPA 2026)
