<div align="center">
<h1>
  XVERSE-MoE-A4.2B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse">🤗 Hugging Face</a>&nbsp｜
        <a href="https://modelscope.cn/organization/xverse" rel="nofollow"><img src="resources/modelscope.png" width="20px" style="max-width: 100%;"> ModelScope</a>&nbsp｜
        <a href="resources/wechat.png">💬 WeChat</a>
</p>

<h4 align="left">
    <p>
        <a href="README.md">中文</a> |
        <b>English</b>
    <p>
</h4>

## Update Information
- **[2024/04/28]** Released **XVERSE-MoE-A4.2B-Chat** MoE chat model.
- **[2024/04/02]** Released **XVERSE-MoE-A4.2B** MoE base model, the Chat version model will be released later.

## Model Introduction

**XVERSE-MoE-A4.2B-Chat** is the aligned version of model **XVERSE-MoE-A4.2B**.

**XVERSE-MoE-A4.2B** is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology which is using Mixture-of-experts (MoE) architecture. The total parameter scale of the model is 25.8 billion, with an actual number of activated parameters being 4.2 billion. The models released this time is the base model **XVERSE-MoE-A4.2B**. Its key features are as follows:

- **Model Structure**: XVERSE-MoE-A4.2B uses the mainstream Decoder-only Transformer network structure that extends the FFN layer of dense models to expert layers. Unlike traditional MoE model where each expert has the same size as standard FFN (such as Mixtral 8x7B), it uses more fine-grained experts, with each expert being 1/4 the size of a standard FFN. It includes shared experts and non-shared experts, where shared experts are always activated during computation, and non-shared experts are selectively activated through a Router.
- **Training Data**: The model has been thoroughly trained on a diversified and high-quality dataset consisting of 2.7 trillion of tokens, including more than 40 languages such as Chinese, English, Russian, and Spanish. The sampling ratio of different types of data is finely set, which makes the performance of Chinese and English excellent, and also takes into account the effect of other languages; The model is trained using training samples of length 8k.
- **Training Framework**: We conducted in-depth customized optimization for the unique expert routing and weight calculation logic in the MoE model, developed an efficient fusion operator to improve computational efficiency. At the same time, to address the challenges of high memory consumption and communication volume in the MoE model, we designed a processing method for overlapping computation, communication, and CPU-Offload to increase overall throughput.

The models sizes, architectures and learning rate of **XVERSE-MoE-A4.2B** are showed as follows:

| total params | activated params | n_layers | d_model | n_heads | d_ff | n_non_shared_experts | n_shared_experts | top_k |   lr   |
| :----------: | :--------------: | :------: | :-----: | :-----: | :--: | :------------------: | :--------------: | :---: | :----: |
|    25.8B     |       4.2B       |    28    |  2560   |   32    | 1728 |          64          |        2         |   6   | 3.5e−4 |

## Model Evaluation

To comprehensively assess the performance of the model, we conducted extensive testing across a range of standard datasets, including C-Eval, CMMLU, Gaokao-Bench, MMLU, AGIEval, RACE-M, CommonSenseQA, PIQA, GSM8K and HumanEval. These evaluations spanned multiple capabilities of the model, specifically including Chinese question answering, English question answering, language comprehension, common sense questioning, logical reasoning, mathematical problem-solving, and coding ability. The results of the evaluations are as follows:

| Dataset                  | XVERSE-MoE-A4.2B | XVERSE-13B-2 | Baichuan2-13B | Llama2-13B | Llama1-65B | XVERSE-7B | DeepSeek-7B | Mistral-7B | Gemma-7B | DeepSeek-MoE-16B |
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

> <sup>1: Tests are conducted only on single-answer multiple-choice questions, thus excluding fill-in-the-blanks, open-ended questions, and multiple-answer multiple-choice questions.</sup>   

For all the comparison models mentioned above, we prioritize the disclosure of their officially published results. In the absence of official data, we refer to the reported outcomes from [OpenCompass Leaderboard](https://opencompass.org.cn/leaderboard-llm). Results not covered by the aforementioned sources are derived from our own evaluation pipline.   
For MMLU, we adopt the [evaluation tools](https://github.com/hendrycks/test) provided by the authors, C-Eval, AGIEval, GAOKAO-Bench are the same as MMLU. For the remaining evaluation datasets, the [OpenCompass](https://github.com/open-compass/OpenCompass/) is employed for evaluation.

## Usage

### Environment Setup

1. Clone this repository:

```shell
git clone https://github.com/xverse-ai/XVERSE-MoE-A4.2B
cd XVERSE-MoE-A4.2B
```

2. Install the dependencies using pip:

```shell
pip install -r requirements.txt
```

### Loading with Transformers

The XVERSE-MoE-A4.2B model can be loaded for inference using the following code:

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

### Web Demo

The following code can be used to start a web server. By entering the access address in the browser, you can perform inference with the XVERSE-MoE-A4.2B model:

```shell
python chat_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

## Limitations and Disclaimer

Like all other Large Language Models (LLMs), XVERSE-MoE-A4.2B may produce inaccurate, biased, or otherwise offensive content under certain circumstances. Therefore, please use the content generated by the model with caution and refrain from disseminating harmful content. Before deploying any application of XVERSE-MoE-A4.2B, developers should conduct safety tests and optimization of the model according to its specific application.

We strongly warn against the use of the XVERSE-MoE-A4.2B model for producing or spreading harmful information, or conducting any activities that might harm the public, national, or social security, or violate regulations. We assume no responsibility for any problems arising from the use of the XVERSE-MoE-A4.2B model, whether it be data security issues, public opinion risks, or any risks and issues caused by misunderstanding, misuse, dissemination, or non-compliance with the model.

## Open Source License

The use of the source code in this repository must follow the [Apache-2.0](LICENSE) open-source license, while the use of the model weights of XVERSE-MoE-A4.2B needs to adhere to the [Model License Agreement](MODEL_LICENSE.pdf).

The XVERSE-MoE-A4.2B model weights are **fully open** to academic research and support **free commercial use**.  To apply for a commercial license, please fill in the [application form](https://chat.xverse.cn/home/business.html). For other questions or collaborations, please contact <opensource@xverse.cn>.

