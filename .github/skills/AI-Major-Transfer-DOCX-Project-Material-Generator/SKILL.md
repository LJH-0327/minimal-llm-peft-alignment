AI Major Transfer DOCX Project Material Generator Skill
Purpose
该 Skill 用于将人工智能、机器学习、大语言模型相关项目转换为适合：

人工智能专业转专业申请材料
本科科研经历展示
面试项目介绍
Personal Statement 项目支撑材料
的中文 DOCX 文档。

输出目标：

不是论文摘要，不是代码说明书，而是一份体现：

AI问题理解能力
技术路线设计能力
工程实践能力
实验分析能力
学术探索潜力
的项目介绍文档。

Output Format Requirement
最终输出必须为：

Microsoft Word (.docx) 文件

而不是 Markdown。

文档要求：

中文为主体语言
标题层级清晰
适合打印阅读
包含必要的流程图和技术示意图
避免大段代码描述
避免论文式Introduction
推荐结构：

项目名称

项目简介

背景与动机

研究问题

技术路线

核心方法

实验设计

实验结果与分析

个人贡献

AI能力体现

项目总结

未来改进方向
Document Style
Overall Tone
采用：

学术探索
工程实践
本科科研经历
风格。

推荐表达：

“探索……”
“设计……实验流程”
“验证……影响”
“分析……变化”
“理解……机制”
禁止：

“提出了一种新的算法”
“实现领先性能”
“显著超过已有方法”
除非有严格实验数据支持。

DOCX Layout Requirement
Cover Section
第一页包含：

项目名称：

例如：

Minimal LLM PEFT & Alignment Framework

中文副标题：

轻量化大语言模型参数高效微调与偏好对齐框架

包含：

项目类型
技术方向
核心关键词
示例：

项目方向：
人工智能 / 大语言模型 / 参数高效微调

关键词：
LLM
PEFT
LoRA
SFT
Preference Optimization
Model Evaluation
Required Visual Components
DOCX必须至少包含：

1. Overall Technical Pipeline Diagram
展示项目整体流程。

推荐形式：

数据准备
   ↓
模型加载
   ↓
LoRA参数高效微调
   ↓
Supervised Fine-Tuning
   ↓
Preference Optimization
   ↓
模型推理
   ↓
结果分析
说明：

流程图用于帮助非专业招生老师快速理解项目。

2. LLM Adaptation Framework Diagram
展示模型适配逻辑。

推荐：

Base LLM

(Qwen / TinyLlama)

        |
        |
        v

Frozen Parameters
        +
LoRA Adapter

        |
        v

Task Adapted Model
强调：

基础模型保持冻结
仅训练少量参数
降低计算成本
3. Experiment Comparison Diagram
展示实验设计。

推荐：

Base Model

      |
      |
      v

SFT Model

      |
      |
      v

Preference Optimized Model


统一Prompt测试

      |

结果比较
突出：

控制变量实验思想。

Content Generation Rules
Project Overview
必须回答：

项目研究什么？
为什么重要？
使用什么AI方法？
学到了什么？
不要直接写：

构建了一个LLM训练框架。

改为：

本项目围绕低资源条件下的小规模语言模型能力迁移问题，探索参数高效微调与偏好优化方法对于模型任务适应性的影响。

Background Section
采用：

现实问题

↓

技术挑战

↓

研究动机

例如：

大型语言模型具有较强生成能力，
但面对专业领域任务时可能存在知识不足、
回答格式不稳定以及任务适应能力有限的问题。

完整训练模型成本较高，因此探索低成本模型适配方法具有实际意义。
Research Question
必须提出问题，而不是描述工具。

推荐：

在有限计算资源条件下，参数高效微调和偏好优化是否能够提升小规模语言模型对于领域任务和指令任务的适应能力？

避免：

使用LoRA训练模型。

Technical Approach Writing Rules
Base Model
介绍：

模型类型
为什么选择
实验价值
避免：

只列模型名称。

PEFT / LoRA
解释：

重点：

为什么使用LoRA。

包括：

参数冻结
Adapter训练
降低显存需求
支持快速实验
不要堆：

rank
alpha
target_modules

等实现参数。

SFT
描述：

数据格式
指令学习目标
模型行为变化
Preference Optimization
必须说明：

DPO属于：

Preference Optimization 方法

不要写：

完成RLHF训练

除非包含：

Reward Model
PPO
Human Feedback Pipeline
Experiment Section
必须体现实验思维。

包含：

Dataset
例如：

FiQA
PubMedQA
Dolly 15K
UltraFeedback
说明：

数据用途，而不是只列名称。

Experimental Pipeline
使用流程图：

Dataset

 ↓

Training Split

 ↓

LoRA SFT

 ↓

Optional DPO

 ↓

Evaluation Prompt

 ↓

Generation Comparison
Result Analysis Rules
如果没有benchmark：

禁止：

“准确率提高20%”

“性能显著提升”

改为：

实验通过固定Prompt生成结果比较，观察模型在回答结构、任务相关性以及指令遵循方面的变化。

强调：

行为变化
优势
局限
下一步改进
Personal Contribution Section
重点体现申请者能力。

必须包括：

AI Understanding
例如：

理解：

Transformer模型
Fine-tuning
Alignment思想
Engineering Ability
例如：

完成：

数据处理
模型训练流程
checkpoint管理
推理测试
Research Thinking
例如：

能够：

设计实验
分析失败案例
认识评价局限
AI Capability Mapping Table
DOCX中加入表格：

AI能力	项目体现
Machine Learning	数据处理、实验设计、模型比较
Deep Learning	Transformer、训练优化
LLM	PEFT、Instruction Learning、Preference Optimization
Engineering	Pipeline、Checkpoint、Inference
Research Thinking	Error Analysis、Evaluation Design
Resume and Interview Section

DOCX Generation Requirements
生成Word文件时：

必须：

使用标题层级
插入流程图
插入技术路线图
插入实验流程图
使用表格总结关键能力
保持页面阅读友好
推荐页数：

2-3页。

Special Rule For This Project
对于：

Minimal LLM PEFT & Alignment Framework

必须定位为：

大语言模型工程实践与方法探索项目

而不是：

大语言模型算法创新项目

最终核心表达：

学生通过构建完整LLM适配流程，理解现代人工智能系统从数据、模型训练、优化方法到评价分析的完整链路。