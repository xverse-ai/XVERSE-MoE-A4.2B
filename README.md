<div align="center">
<h1>
  XVERSE-MoE-A4.2B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse">🤗 Hugging Face</a>&nbsp｜
        <a href="https://modelscope.cn/organization/xverse" rel="nofollow"><img src="resources/modelscope.png" width="20px" style="max-width: 100%;"> ModelScope</a>&nbsp｜
        <a href="resources/wechat.png">💬 微信社区</a>
</p>

<h4 align="left">
    <p>
        <b>中文</b> |
        <a href="README_EN.md">English</a>
    <p>
</h4>

## 更新信息
- **[2024/04/02]** 发布 MoE 架构的 **XVERSE-MoE-A4.2B** 底座模型，Chat 对齐模型将在后续发布。

## 模型介绍

**XVERSE-MoE-A4.2B** 是由深圳元象科技自主研发的支持多语言的大语言模型（Large Language Model），使用混合专家模型（MoE，Mixture-of-experts）架构，模型的总参数规模为 258 亿，实际激活的参数量为 42 亿，本次开源的模型为底座模型 **XVERSE-MoE-A4.2B**，主要特点如下：

- **模型结构**：XVERSE-MoE-A4.2B 为 Decoder-only 的 Transformer 架构，将密集模型的 FFN 层扩展为专家层，不同于传统 MoE 中每个专家的大小与标准 FFN 相同（如Mixtral 8x7B ），使用了更细粒度的专家，每个专家是标准 FFN 大小的 1/4，并设置了共享专家（Shared Expert）和非共享专家（Non-shared Expert）两类，共享专家在计算时始终被激活，非共享专家通过 Router 选择性激活。
- **训练数据**：构建了 2.7 万亿 token 的高质量、多样化的数据对模型进行充分训练，包含中、英、俄、西等 40 多种语言，通过精细化设置不同类型数据的采样比例，使得中英两种语言表现优异，也能兼顾其他语言效果；模型使用 8K 长度的训练样本进行训练。
- **训练框架**：针对 MoE 模型中独有的专家路由和权重计算逻辑，进行了深入定制优化，开发出一套高效的融合算子，以提升计算效率。同时，为解决 MoE 模型显存占用和通信量大的挑战，设计了计算、通信和 CPU-Offload 的 Overlap 处理方式，从而提高整体吞吐量。

**XVERSE-MoE-A4.2B** 的模型大小、架构和学习率如下：

| total params | activated params | n_layers | d_model | n_heads | d_ff | n_non_shared_experts | n_shared_experts | top_k |   lr   |
| :----------: | :--------------: | :------: | :-----: | :-----: | :--: | :------------------: | :--------------: | :---: | :----: |
|    25.8B     |       4.2B       |    28    |  2560   |   32    | 1728 |          64          |        2         |   6   | 3.5e−4 |

## 评测结果

为了综合评估模型的性能，我们在一系列标准数据集上进行了全面测试，包括C-Eval、CMMLU、Gaokao-Bench、MMLU、AGIEval、RACE-M、CommonSenseQA、PIQA、GSM8K和HumanEval。这些评估覆盖了模型在多个领域的能力，具体包括中文问答、英文问答、语言理解、常识问答、逻辑推理、数学问题解答以及编程能力。评估结果如下：

| 数据集                   | XVERSE-MoE-A4.2B | XVERSE-13B-2 | Baichuan2-13B | Llama2-13B | Llama1-65B | XVERSE-7B | DeepSeek-7B | Mistral-7B | Gemma-7B | DeepSeek-MoE-16B |
| ------------------------ | :--------------: | :----------: | :-----------: | :--------: | :--------: | :-------: | :---------: | :--------: | :------: | :--------------: |
| C-Eval                   |       60.5       |     62.0     |     58.1      |    35.6    |    38.8    |   57.1    |    45.0     |    45.1    |   50.0   |       40.6       |
| CMMLU                    |       64.5       |     65.4     |     62.0      |    38.4    |    40.6    |   61.3    |    47.2     |    44.9    |   50.5   |       42.5       |
| Gaokao-Bench<sup>1</sup> |       60.3       |     65.3     |     54.3      |    35.4    |    38.9    |   61.7    |    35.4     |    40.2    |   42.3   |       29.1       |
| MMLU                     |       60.2       |     60.0     |     59.2      |    54.8    |    63.4    |   56.6    |    48.2     |    62.5    |   64.3   |        45        |
| AGIEval<sup>1</sup>      |       48.0       |     52.4     |     48.2      |    33.4    |    42.4    |   46.9    |    26.4     |    41.2    |   41.7   |       31.7       |
| RACE-M                   |       75.4       |     82.4     |     68.9      |    63.0    |    67.9    |   79.0    |    63.2     |    67.5    |   80.2   |       61.9       |
| CommonSenseQA            |       70.0       |     68.0     |     65.6      |    67.3    |    74.0    |   64.1    |    56.4     |    68.8    |   74.0   |       54.8       |
| PIQA                     |       81.4       |     79.8     |     78.5      |    80.5    |    82.8    |   76.7    |    79.2     |    82.2    |   81.2   |       80.2       |
| GSM8K                    |       51.2       |     52.7     |     52.7      |    28.7    |    50.9    |   19.3    |    17.4     |    35.4    |   46.4   |       18.8       |
| HumanEval                |       29.9       |     32.3     |     17.1      |    18.3    |    23.7    |   10.4    |    26.2     |    26.2    |   32.3   |       26.8       |

> <sup>1：只针对其中的单项选择题进行测试，即排除了填空题、开放性问题和多项选择题</sup>   

对于上述所有比较模型，我们优先汇报其官方公布的结果。在缺少官方结果的情况下，我们采用了 [OpenCompass 榜单](https://opencompass.org.cn/leaderboard-llm)的报告结果。其他结果则来自于我们自行执行的评估流程所获得的数据。   
对于 MMLU ，我们采用作者提供的[评测工具](https://github.com/hendrycks/test)，C-Eval、AGIEval、GAOKAO-Bench 与 MMLU 的评测方式相同，其余评测数据集使用 [OpenCompass 评估框架](https://github.com/open-compass/OpenCompass/)进行评估。

## 使用方法

### 环境安装

1. 下载本仓库：

```shell
git clone https://github.com/xverse-ai/XVERSE-MoE-A4.2B
cd XVERSE-MoE-A4.2B
```

2. 使用 pip 安装依赖：

```shell
pip install -r requirements.txt
```
### Transformers 加载方式

可通过以下代码加载 XVERSE-MoE-A4.2B 模型来进行推理：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("xverse/XVERSE-MoE-A4.2B")
model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-MoE-A4.2B", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()
inputs = tokenizer('北京的景点：故宫、天坛、万里长城等。\n深圳的景点：', return_tensors='pt').input_ids
inputs = inputs.cuda()
generated_ids = model.generate(inputs, max_new_tokens=64, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.1)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
```

### 网页 Demo

可通过以下代码启动一个web server，在浏览器输入访问地址后，可使用 XVERSE-MoE-A4.2B 模型进行推理：

```shell
python text_generation_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

## 局限性与免责申明

XVERSE-MoE-A4.2B 与其他所有 LLM 一样，在某些情况下可能会产生不准确、有偏见或其他令人反感的内容。因此，请谨慎使用模型生成的内容，请勿将生成的有害内容进行传播，在部署任何 XVERSE-MoE-A4.2B 的应用之前，开发人员应根据其具体应用对模型进行安全测试和调优。

我们强烈警告不要将 XVERSE-MoE-A4.2B 模型用于制造或传播有害信息，或进行任何可能损害公众、国家、社会安全或违反法规的活动。如果使用 XVERSE-MoE-A4.2B 模型产生任何问题，无论是数据安全问题、公共舆论风险，还是模型被误解、滥用、传播或不合规使用所引发的任何风险和问题，我们将不承担任何责任。

## 模型开源协议

使用本仓库的源码需要遵循 [Apache-2.0](LICENSE) 开源协议，使用 XVERSE-MoE-A4.2B 的模型权重则需要遵循[模型许可协议](MODEL_LICENSE.pdf)。

XVERSE-MoE-A4.2B 模型权重对学术研究**完全开放**，并且支持**免费商用**。如需申请商业许可证，请填写【[申请表](https://chat.xverse.cn/home/business.html)】，如有其他问题或合作，请联系 <opensource@xverse.cn>。

