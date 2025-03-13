# ktransformers部署Deepseek-r1满血版结合AnythingLLM工具，支持知识库调用
### 基于官方版本v2.1.0修改的Ktransformers，支持流式返回，兼容OpenAI接口格式

- 修改了  `./ktransformers/util/utils.py` 文件，将生成方法新增变量，带有可选的流式返回
- 基于官方的local_chat.py，结合Flask框架，实现了兼容OpenAI的流式接口，位于文件  `./ktransformers/chat_openai.py`

### 部署条件
- 我部署的版本是Deepseek-R1-671b-Q8，该版本相较与满血版的Deepseek性能损失最小，8位精度版本的性能损失基本可以忽略不计（美团给出过证明）
- 需要713G上下的内存（这里我的内存不够，采用了虚拟内存，严重拖慢速度，时间都花在交换区交换数据上了）
- ![](MEM.png)
- 显存仅需14G，没有对显卡有很高的要求
- ![](nvidia-smi.png)
