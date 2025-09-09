# 自适应 RAG 系统

> 基于 **LangGraph** 的智能代理规划系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文文档](README.md) | [English Documentation](README_EN.md)

## 项目概述

自适应 RAG 是一个智能代理规划系统，通过 LangGraph 状态管理展示了先进的 AI 编排能力。该项目专注于自适应路由、混合 RAG（检索增强生成），最重要的是，实现了一个新颖的规划系统，该系统将使用从网络搜索中找到的答案的相关评论来进行重新校准，同时还实现了智能查询分类、人机协作规划、多源信息收集和对话管理。

## 项目特色

- **技术栈**: LangGraph + LangChain + OpenAI + ChromaDB
- **自适应查询路由**: 智能分类（直接/普通/规划器）
- **人机协作规划**: 复杂计划的交互式信息收集
- **多源信息收集**: 网络搜索 + 社交平台（Reddit、知乎）
- **高级 RAG**: 混合文档检索（Dense + BM25 + 重排序 + 压缩）
- **对话管理**: 自动摘要 + 持久状态跟踪

## 技术栈

| 组件 | 技术 | 版本 | 用途 |
|-----------|------------|---------|---------|
| **图框架** | LangGraph | 0.6+ | 基于状态的代理编排 |
| **LLM 集成** | LangChain | 0.3+ | 语言模型集成 |
| **语言模型** | GPT-5/GPT-4o/4o-mini | 最新版 | 文本生成和分析 |
| **向量数据库** | ChromaDB | 1.0+ | 文档嵌入存储 |
| **网络搜索** | Tavily API | 最新版 | 实时信息检索 |
| **社交平台** | Reddit + 知乎 APIs | 最新版 | 社区洞察收集 |
| **数据库** | SQLite | 3.0+ | 对话持久化 |

## 核心技术

### 1. 自适应查询路由系统
- **智能分类**: 自动查询分类（直接/普通/规划器）
- **上下文感知路由**: 动态响应策略选择
- **工具选择**: 基于查询复杂度的自适应工具使用

### 2. 人机协作规划
- **交互式信息收集**: 多轮澄清过程
- **智能充分性检测**: 自动评估信息完整性
- **规划生成**: 使用收集的上下文创建综合计划

### 3. 多源信息系统
- **网络搜索集成**: 通过 Tavily API 获取实时信息
- **社交平台洞察**: 来自 Reddit 和知乎的社区观点
- **本地文档检索**: 基于向量的文档搜索能力

### 4. 对话管理系统
- **自动摘要**: 长对话的智能上下文压缩
- **状态持久化**: 基于 SQLite 的对话检查点
- **用户和线程管理**: 有组织的对话跟踪

## 快速开始

### 系统要求

- **Python**: 3.13+
- **内存**: 4GB+（推荐 8GB）
- **API 密钥**: OpenAI、Tavily、Reddit（可选）
- **存储**: 2GB+ 用于依赖项和数据

### 环境设置

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/Adaptive-RAG.git
cd Adaptive-RAG/server

# 2. 安装依赖
pip install -r requirements-lock.txt

# 3. 配置环境
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥

# 4. 运行系统
python main.py
```

设置完成后，系统将引导您完成：
- **用户管理**: 创建或选择用户配置文件
- **线程管理**: 在选定用户下创建或选择线程
- **交互式聊天**: 自适应路由和智能响应

### 测试规划器核心功能
```bash
cd server/tests
python quick_test.py
```

## 项目结构

```
server/
├── main.py                          # 入口点 - 带依赖检查的系统启动器
├── requirements.txt                 
├── requirements-lock.txt            
├── .env.example                     
├── .gitignore                       
├── config/
│   ├── __init__.py
│   └── config.py                    # 配置管理和环境变量
├── core/
│   ├── __init__.py
│   ├── run_time.py                  # 主运行时与图定义
│   ├── nodes.py                     # 图节点：路由器、分析器、规划器、摘要器
│   ├── planner.py                   # 规划系统实现
│   └── db.py                        # 数据库操作和管理
├── data/
├── utils/
│   ├── __init__.py
│   ├── search.py                    # Web_search、advan_web_search 工具调用
│   ├── retrieve.py                  # 文档检索工具
│   ├── reddit_search.py             # Reddit API 集成
│   └── zhihu_search.py              # 知乎平台集成
├── tests/
│   ├── __init__.py
│   └── quick_test.py                # 基础搜索测试
└── deployment/
```

## 核心算法

### 1. 自适应路由策略
```python
# 使用 LLM 进行查询分类
def classify_query(query):
    # direct: 直接查询
    # normal: 需要考虑上下文的查询  
    # planner: 需要详细规划的复杂查询
    return route_classification

# 动态工作流选择
def select_tools(route, context):
    if route == "direct":
        return [llm_node, web_search]
    elif route == "normal":
        return [rewrite_node, llm_node, web_search]
    else:  # planner
        return [rewrite_node, analyzer_node, planner_node, advan_web_search]
```

### 2. 人机协作规划
- **信息缺口分析**: 识别缺失的关键信息
- **交互式澄清**: 多轮问答过程
- **充分性评估**: 确定何时收集到足够信息
- **计划生成**: 使用收集的上下文创建综合计划

### 3. 对话状态管理
- **自动摘要**: 压缩长对话同时保留关键信息
- **上下文保持**: 跨会话维护对话连续性
- **状态持久化**: 基于 SQLite 的对话恢复检查点

## 配置详情

### 主要配置项
在 `server/config/config.py` 中调整系统参数：

```python
# 模型配置
MODEL = "gpt-4o"                            # 主模型
SIDE_MODEL = "gpt-4o-mini"                  # 分类用辅助模型
PLANNER_MODEL = "gpt-5"                     # 规划模型

# 图配置
MAX_ROUNDS = 3                              # 最大 HITL 轮数
RECENT_K = 6                                # 保留的最近消息数
SUMMARIZE_AFTER = 8                         # 触发摘要的阈值

# 检索配置
CHUNK_SIZE = 835                            # 文档块大小
CHUNK_OVERLAP = 120                         # 块重叠
EMB_MODEL = "text-embedding-3-large"        # 嵌入模型
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 重排序模型
```

### 环境变量配置
配置 `.env` 文件：
```bash
# 主要配置
OPENAI_API_KEY=your_openai_api_key
MODEL=gpt-4o
SIDE_MODEL=gpt-4o-mini
PLANNER_MODEL=gpt-5
TAVILY_API_KEY=your_tavily_api_key

# 检索配置
EMB_MODEL=text-embedding-3-large
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Reddit 客户端参数（可选）
CLIENT_ID=your_reddit_client_id
CLIENT_SECRET=your_reddit_client_secret
REDDIT_PASSWORD=your_reddit_password
REDDIT_USERNAME=your_reddit_username
USER_AGENT=python:mybot:v1.0 (by u/your_username)

# LangSmith 设置（可选）
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=Adaptive-rag
```
#### 基于 Reddit + 知乎的网络搜索
![advan_web_search](server/img/advan_web_search.png)
#### 人机协作
![HITL](server/img/HITL.png)

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

### 如果这个项目对您有帮助，请给它一个 Star！

**技术栈**: Python + LangGraph + LangChain + OpenAI + ChromaDB
**邮箱**: jianwend@vt.edu
**项目地址**: https://github.com/Golaugh/Adaptive-rag

[中文文档](README.md) | [English Documentation](README_EN.md)
