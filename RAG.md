# 1.LangChain的简介

LangChain由Harrison Chase创建于2022年10月，它是围绕LLMs（大语言模型）建立的一个框架。LangChain自身并不开发LLMs，它的核心理念是为各种LLMs**实现通用接口**，把LLMs相关的组件“链接”在一起，简化LLMs应用的开发难度，方便开发者快速的开发复杂的LLMs应用。

- **LangChain的主要功能**：
  - Prompts提示词工程
  - Models调用各类模型
  - History管理会话历史记录
  - Indexes管理和分析各类文档
  - Chains构建功能的执行链条
  - Agent构建智能体

# 2.LangChain的环境部署

pip install langchain langchain-community langchain-ollama dashscope chromadb

- langchain：核心包
- langchain-community：社区支持包，提供了更多的第三方模型调用（我们用的阿里云千问模型就需要这个包）
- langchain-ollama：Ollama 支持包，支持调用 Ollama 托管部署的本地模型
- dashscope：阿里云通义千问的 Python SDK
- chromadb：轻量向量数据库（后续使用）

# 3.RAG介绍

通用的基础大模型存在一些问题：

- LLM的知识不是实时的，模型训练好后不具备自动更新知识的能力，会导致部分信息滞后
- LLM领域知识是缺乏的，大模型的知识来源于训练数据，这些数据主要来自公开的互联网和开源数据集，无法覆盖特定领域或高度专业化的内部知识
- 幻觉问题，LLM有时会在回答中生成看似合理但实际上是错误的信息
- 数据安全性
  **RAG（Retrieval Augmented Generation）即检索增强生成，为大模型提供了从特定数据源检索到的信息，以此来修正和补充生成的答案。可以总结为一个公式：RAG=检索技术+LLM提示**
  RAG的核心工作原理：![image-20261114092437085](C:\Users\32494\AppData\Roaming\Typora\typora-user-images\image-20261114092437085.png)
  **RAG标准流程：**
  RAG标准流程由索引、检索、生成三个核心阶段组成：
- 索引阶段，通过处理多种来源多种格式的文档提取其中文本，将其切分为标准长度的文本块，并进行嵌入向量化，向量存储在向量数据库
  - 加载文本
  - 内容提取
  - 文本分割，形成chunk
  - 文本向量化
  - 存向量数据库
- 检索阶段，用户输入查询被转化为向量表示，通过相似度匹配从向量数据库中检索出最相关的文本块
  - query向量化
  - 在文本向量中匹配出与问句向量相似的k个
- 生成阶段，检索到的相关文本与原始查询共同构成提示词，输入大语言模型，生成精确且具备上下文关联的回答。
  - 匹配出的文本作为上下文和问题一起添加到prompt中
  - 提交给LLM生成答案  
    **RAG的核心价值**：
- 解决知识实效性问题
- 降低模型幻觉
- 无需重新训练模型

# 4.LangChain组件：Models大语言模型的使用

现在市面上的模型多如牛毛，各种各样的模型不断出现，LangChain 模型组件提供了与各种模型的集成，并为所有模型提供一个精简的统一接口。

LangChain 目前支持三种类型的模型：LLMs（大语言模型）、Chat Models（聊天模型）、Embeddings Models（嵌入模型）。

- LLMs：是技术范畴的统称，指基于大参数量、海量文本训练的 Transformer 架构模型，核心能力是理解和生成自然语言，主要服务于文本生成场景
- 聊天模型：是应用范畴的细分，是专为对话场景优化的 LLMs，核心能力是模拟人类对话的轮次交互，主要服务于聊天场景
- 文本嵌入模型：文本嵌入模型接收文本作为输入，得到文本的向量。

LangChain 支持的三类模型，它们的使用场景不同，输入和输出不同，开发者需要根据项目需要选择相应。

我们所用的阿里云通义千问系列主要来自于：langchain_community 包

**LangChain访问阿里云通义千问大模型**：

```
from langchain_community.llms.tongyi import Tongyi

model=Tongyi(model='qwen-max')
#调用invoke向模型提问
res=model.invoke(input='你是谁呀能做什么？')
print(res)
```

**LangChain访问Ollama本地模型**:

```
from langchain_ollama import OllamaLLM

model=OllamaLLM(model='qwen-max')

res=model.invoke(input='你是谁呀能做什么？')

print(res)
```

#  5.LangChain组件：Models的流式输出

如果需要流失输出结果，需要将模型的invoke方法改为stream方法：

- invoke方法：一次性返回完整结果
- stream方法：逐段返回结果，流式输出

```
from langchain_community.llms.tongyi import Tongyi

model=Tongyi(model='qwen-max')

res=model.stream(input='你是谁呀能做什么？')

for chunk in res:
    print(chunk,end='',flush=True)
```

这两个方法是新版LangChain（1.0版本后）中基于Runnable接口的通用核心方法。

绝大多数组件（如提示词模板、链、向量检索、工具调用等）都支持这两个方法，这也是LangChain设计的核心统一范式。

#  6.LangChain组件：Models聊天模型的使用

聊天消息包含下面几种类型，使用时需要按照约定传入合适的值：

- AIMessage：就是 AI 输出的消息，可以是针对问题的回答。（OpenAI 库中的 assistant 角色）
- HumanMessage：人类消息就是用户信息，由人给出的信息发送给 LLMs 的提示信息，比如 “实现一个快速排序方法”。（OpenAI 库中的 user 角色）
- SystemMessage：可以用于指定模型具体所处的环境和背景，如角色扮演等。你可以在这里给出具体的指示，比如 “作为一个代码专家”，或者 “返回 json 格式”。（OpenAI 库中的 system 角色）

```
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

model=ChatTongyi(model='qwen-max')

messages=[
    SystemMessage(content='你是一个边塞诗人'),
    HumanMessage(content="写一首唐诗"),
    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    HumanMessage(content="按照你上一个回复的格式，再写一首唐诗。")
]

res=model.stream(input=messages)

for chunk in res:
    print(chunk.content,end='',flush=True)
```

#  7.LangChain组件：Models消息的简写形式

messages=[
    SystemMessage(content='你是一个边塞诗人'),
    HumanMessage(content="写一首唐诗"),
    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    HumanMessage(content="按照你上一个回复的格式，再写一首唐诗。")
]

```
messages=[
    ('system','你是一个边塞诗人'),
    ('human',"写一首唐诗"),
    ('ai',"锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    ('human',"按照你上一个回复的格式，再写一首唐诗。")
]
```

```
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

model = ChatTongyi(
    model="qwen-plus-character",
    streaming=True,
)

messages=[
    ('system','你是一个边塞诗人'),
    ('human',"写一首唐诗"),
    ('ai',"锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    ('human',"按照你上一个回复的格式，再写一首唐诗。")
]

res=model.stream(input=messages)

for chunk in res:
    print(chunk.content,end='',flush=True)
```

**总结**：

```
messages = [
    SystemMessage(content="内容..."),
    HumanMessage(content="内容..."),
    AIMessage(content="内容..."),
]
```

是静态的，一步到位 直接就得到了 Message 类的类对象

```
messages = [
    ("system", "内容..."),
    ("human", "内容..."),
    ("ai", "内容..."),
]
```

是动态的，需要在运行时 由 LangChain 内部机制转换为 Message 类对象

```
messages = [
    ("system", "今天的天气是{weather}"),
    ("human", "我的名字是: {name}"),
    ("ai", "欢迎{lastname}先生"),
]
```

由于是动态，需要转换步骤 所以简写形式支持内部填充 {变量} 占位 可在运行时填充具体值 (后续学习提示词模板时用到)

#  8.LangChain组件：Models嵌入模型的使用

**Embeddings Models（文本嵌入模型）**：

Embeddings Models（文本嵌入模型）的特点：将字符串作为输入，返回一个浮点数的列表（向量）

在NLP中，Embedding的作用就是将数据进行文本向量化

```
from langchain_community.embeddings import DashScopeEmbeddings

model=DashScopeEmbeddings()

#不用invoke，stream
#embed_query 单次转换 embed_documents 批量转换

print(model.embed_query("我喜欢你"))
print(model.embed_documents(["我喜欢你","我稀饭你","晚上吃什么"]))
```

目前所掌握的LangChain API如下：

| 方式            | LLMs 大语言模型                                          | 聊天模型                                                     | 文本嵌入模型                                                 |
| --------------- | -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 阿里云千问      | `from langchain_community.llms.tongyi import Tongyi`     | `from langchain_community.chat_models.tongyi import ChatTongyi` | `from langchain_community.embeddings import DashScopeEmbeddings` |
| Ollama 本地模型 | `from langchain_ollama import OllamaLLM`                 | `from langchain_ollama import ChatOllama`                    | `from langchain_ollama import OllamaEmbeddings`              |
| 常用方法        | `invoke` 单次调用 / `batch` 批量调用 / `stream` 流式输出 | `invoke` 单次调用 / `batch` 批量调用 / `stream` 流式输出     | `embed_query` 单条文本 / `embed_documents` 多条文本          |

#  9.LangChain组件：通用Prompts

- 提示词优化在模型应用中非常重要，LangChain提供了PromptTemplate类，用来协助优化提示词
- PromptTemplate表示提示词模板，可以构建一个自定义的基础提示词模板，支持变量的注入，最终生成所需的提示词

**AIMessage对象包含的字段**：

| 字段                 | 保存的内容                                   |
| -------------------- | -------------------------------------------- |
| `content`            | 回答正文，通常是字符串，也可能是内容块列表   |
| `type`               | 消息类型，`AIMessage` 对应 `"ai"`            |
| `id`                 | 消息 ID，可能为空                            |
| `name`               | 消息名称，可选                               |
| `response_metadata`  | 模型返回的附加信息，例如结束原因、模型名称等 |
| `usage_metadata`     | 标准化的 Token 用量信息，可能为空            |
| `tool_calls`         | AI 请求调用的工具及参数，没有时通常为空列表  |
| `invalid_tool_calls` | 未能正确解析的工具调用                       |
| `additional_kwargs`  | 其他由模型或接口提供的附加数据               |

```
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate

# 1. 创建提示词模板
prompts_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了一个{gender}，"
    "请帮孩子起个名字，简单回答。"
)

# 2. 填入变量，生成提示词字符串
prompts_text = prompts_template.format(
    lastname="张",
    gender="女儿",
)

print("生成的提示词：", prompts_text)

# 3. 使用聊天模型封装
model = ChatTongyi(model="qwen-plus-character")

# ChatTongyi 会将字符串转换成人类消息，再发送给模型
res = model.invoke(prompts_text)

# 4. 聊天模型返回消息对象，content 是回答正文
print("模型回答：", res.content)
print(res)
print(type(res))
```

基于 PromptTemplate 类可以得到提示词模板，支持基于模板注入变量得到最终提示词。

- zero-shot 思想下，可以基于 PromptTemplate 直接完成。
- few-shot 思想下，需要更换为 FewShotPromptTemplate（后续学习）

PS：使用 PromptTemplate 还不如自己手动拼接字符串？

- 使用 Template 模板构建提示词，在大型工程中更容易做标准化模板
- Template 模板类，支持 LangChian 框架的链式调用（Runnable 接口，后续学习）
  - PromptTemplate
  - FewShotPromptTemplate 
  - ChatPromptTemplate 

#  10.LangChain组件：FewShotPromptTemplate的使用

```
from langchain_core.prompts import FewShotPromptTemplate

FewShotPromptTemplate(
    examples=None,
    example_prompt=None,
    prefix=None,
    suffix=None,
    input_variables=None
)
```

参数：

- examples：示例数据，list，内套字典
- example_prompt：示例数据的提示词模板
- prefix：组装提示词，示例数据前内容
- suffix：组装提示词，示例数据后内容
- input_variables：列表，注入的变量列表

```
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.llms.tongyi import Tongyi
# 示例的模板
example_template = PromptTemplate.from_template("单词: {word}, 反义词: {antonym}")

# 示例的动态数据注入 要求是list内部套字典
examples_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"}
]

few_shot_template = FewShotPromptTemplate(
    example_prompt=example_template,        # 示例数据的模板
    examples=examples_data,                 # 示例的数据（用来注入动态数据的），list内套字典
    prefix="告知我单词的反义词，我提供如下的示例：",        # 示例之前的提示词
    suffix="基于前面的示例告知我，{input_word}的反义词是？",  # 示例之后的提示词
    input_variables=['input_word']          # 声明在前缀或后缀中所需要注入的变量名
)

prompt_text=few_shot_template.invoke(input={"input_word": "左"}).to_string()
print(prompt_text)

model=Tongyi(model="qwen-max")
print(model.invoke(input=prompt_text))
```

#  11.LangChain组件：模板类format和invoke方法

在 PromptTemplate（通用提示词模板）和 FewShotPromptTemplate（FewShot 提示词模板）的使用中，我们使用了如下：

PromptTemplate、FewShotPromptTemplate、ChatPromptTemplate都拥有 format 和 invoke 这 2 类方法。

继承关系：

- `Runnable` 定义 invoke 方法的规范
- `BasePromptTemplate` 继承自 Runnable，定义 format 方法的规范
- `PromptTemplate` / `FewShotPromptTemplate` / `ChatPromptTemplate` 继承自 BasePromptTemplate

**format和invoke的区别在于：**

| 区别   | format                             | invoke                                            |
| ------ | ---------------------------------- | ------------------------------------------------- |
| 功能   | 纯字符串替换，解析占位符生成提示词 | Runnable 接口标准方法，解析占位符生成提示词       |
| 返回值 | 字符串                             | PromptValue 类对象                                |
| 传参   | `.format(k=v, k=v, ...)`           | `.invoke({"k": v, "k": v, ...})`                  |
| 解析   | 支持解析占位符                     | 支持解析占位符和 MessagesPlaceholder 结构化占位符 |

**format方法：**

```
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}, 刚生了{gender}, 你帮我起个名字，简单回答。"
)
# 调用.format方法注入信息即可
prompt_text = prompt_template.format(lastname="张", gender="女儿")

model = Tongyi(model="qwen-max")
res = model.invoke(input=prompt_text)
print(res)
```

**invoke方法：**

```
few_shot_template = FewShotPromptTemplate(
    example_prompt=example_template,    # 示例数据的模板
    examples=examples_data,             # 示例的数据（用来注入动态数据的），list内套
    prefix="告知我单词的反义词，我提供如下的示例：", # 示例之前的提示
    suffix="基于前面的示例告知我，{input_word}的反义词是？", # 示例之后的提示
    input_variables=['input_word']      # 声明在前缀或后缀中所需要注入的变量名
)

prompt_text = few_shot_template.invoke(input={"input_word": "左"}).to_string()
print(prompt_text)
```

```
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

"""
PromptTemplate -> StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
FewShotPromptTemplate -> StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
ChatPromptTemplate -> BaseChatPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
"""

template = PromptTemplate.from_template("我的邻居是: {lastname}，最喜欢: {hobby}")
res = template.format(lastname="张大明", hobby="钓鱼")
print("format结果：", res, type(res))

res2 = template.invoke({"lastname": "周杰伦", "hobby": "唱歌"})
print("invoke结果：", res2.to_string(), type(res2))
print("invoke原始对象：", res2)
```

#  12.LangChain组件：ChatPromptTemplate的使用

PromptTemplate：通用提示词模板，支持动态注入信息。

 FewShotPromptTemplate：支持基于模板注入任意数量的示例信息。

 ChatPromptTemplate：支持注入任意数量的历史会话信息。

- 通过`from_messages`方法，从列表中获取多轮次会话作为聊天的基础模板
  - PS: 前面 PromptTemplate 类用的`from_template`仅能接入一条消息，而`from_messages`可以接入一个 list 的消息

历史会话信息并不是静态的（固定的），而是随着对话的进行不停地积攒，即动态的。 所以，历史会话信息需要支持动态注入。

```
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "……"),
        ("ai", "……"),
        MessagesPlaceholder("history"),
        ("human", "……")
    ]
)

history_data = [
    ("human", "..."),
    ("ai", "..."),
    ("human", "..."),
    ("ai", "...")
]

prompt_value = chat_template.invoke({"history": history_data})
```

- MessagesPlaceholder 作为占位
  - 提供 history 作为占位的 key
  - 基于 invoke 动态注入历史会话记录
- 必须是 invoke，format 无法注入

```
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system','你是一个边塞诗人'),
        MessagesPlaceholder('history'),
        ('human','请再来一首唐诗'),
    ]
)

history_data = [
    ('human','你来写一首唐诗'),
    ('ai','床前明月光，疑是地上霜，举头望明月，低头思故乡。'),
    ('human','好诗，再来一首。'),
    ('ai','锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。'),
]

prompt_text=chat_prompt_template.invoke({'history':history_data}).to_string()
model=ChatTongyi(model='qwen-max')
res=model.invoke(prompt_text)
print(res.content,type(res))
```

#  13.LangChain组件：chains链的基础使用

「将组件串联，上一个组件的输出作为下一个组件的输入」是 LangChain 链（尤其是管道链）的核心工作原理，这也是链式调用的核心价值：实现数据的自动化流转与组件的协同工作，如下。

```
chain = prompt_template | model
```

核心前提：即 Runnable 子类对象才能入链（以及 Callable、Mapping 接口子类对象也可加入（后续了解用的不多））。 我们目前所学习到的组件，均是 Runnable 接口的子类，如下类的继承关系：

继承关系梳理：

- PromptTemplate、FewShotPromptTemplate → StringPromptTemplate → BasePromptTemplate → Runnable
- ChatPromptTemplate → BaseChatPromptTemplate → BasePromptTemplate → Runnable
- Tongyi → BaseLLM → BaseLanguageModel → RunnableSerializable → Runnable
- ChatTongyi → BaseChatModel → BaseLanguageModel → RunnableSerializable → Runnable

```
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables.base import RunnableSerializable

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗。"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首唐诗，无需额外输出"),
    ]
)

history_data = [
    ("human", "你来写一个唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗再来一个"),
    ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
]

model = ChatTongyi(model="qwen3-max")

chain: RunnableSerializable = chat_prompt_template | model
print(type(chain))

# Runnable接口，invoke执行
res = chain.invoke({"history": history_data})
print(res.content)

# Runnable接口，stream执行
for chunk in chain.stream({"history": history_data}):
    print(chunk.content, end="", flush=True)
```

- 通过 `|` 链接提示词模板对象和模型对象
- 返回值`chain`对象是`RunnableSerializable`对象
  - 是 Runnable 接口的直接子类
  - 也是绝大多数组件的父类
- 通过`invoke`或`stream`进行阻塞执行或流式执行

组成的链在执行上有：上一个组件的输出作为下一个组件的输入的特性。 

执行流程： `{"history": history_data}` `→` `chat_prompt_template` `→` `PromptValue` 提示词文本 → 模型对象 → AIMessage 消息对象

#  14.LangChain组件：Runnable接口

LangChain 中的绝大多数核心组件都继承了 Runnable 抽象基类（位于 `langchain_core.runnables.base`）

代码：

```
chain = prompt | model
```

`chain`变量是`RunnableSequence`（`RunnableSerializable`子类）类型 而得到这个类型的原因就是 Runnable 基类内部对`__or__`魔术方法的改写。 同时，在后面继续使用`|`添加新的组件，依旧会得到`RunnableSequence`，这就是链的基础架构。

```
def __or__(
    self,
    other: Runnable[Any, Other]
    | Callable[[Iterator[Any]], Iterator[Other]]
    | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
    | Callable[[Any], Other]
    | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
) -> RunnableSerializable[Input, Other]:
    """Runnable "or" operator.

    Compose this `Runnable` with another object to create a `RunnableSequence`.

    Args:
        other: Another `Runnable` or a `Runnable`-like object.

    Returns:
        A new `Runnable`.
    """
    return RunnableSequence(steps=[self, coerce_to_runnable(other)])
```

```
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate

prompt=PromptTemplate.from_template('你是一个AI助手')
model=Tongyi(model='qwen-max')

chain=prompt|model
print(chain,type(chain))
```

#  15.LangChain组件：StrOutputParser解析器

StrOutputParser是LangChain内置的简单字符串解析器

- 可以将AIMessage解析为简单的字符串，符合了模型invoke方法要求（可传入字符串，不接收AIMessage类型）
- 是Runnable接口的子类（可以加入链）

```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage

parser = StrOutputParser()
model = ChatTongyi(model="qwen-plus-character")
prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请起名，仅告知我名字无需其它内容。"
)

chain = prompt | model | parser

res= chain.invoke({"lastname": "张", "gender": "女儿"})
print(res)
```

#  16.LangChain组件：JsonOutputParser解析器

```
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate

str_parser = StrOutputParser()
json_parser = JsonOutputParser()

model = ChatTongyi(model='qwen-plus-character')

first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname},刚生了{gender},请帮忙起名字，"
    "并封装为JSON格式返回给我，要求key是name,value就是你起的名字，请严格遵守。"
)

second_prompt = PromptTemplate.from_template(
    "姓名：{name},请帮我解析含义。"
)

chain = first_prompt | model | json_parser | second_prompt | model | str_parser

res = chain.invoke({'lastname': '张', 'gender': '女'})

print(res)
print(type(res))
```

**总结**

在构建链的时候要注意整体兼容性，注意前后组件的输入和输出要求。

- 模型输入：PromptValue 或字符串或序列（BaseMessage、list、tuple、str、dict）。
- 模型输出：AIMessage
- 提示词模板输入：要求是字典
- 提示词模板输出：PromptValue 对象
- StrOutputParser：AIMessage 输入、str 输出
- JsonOutputParser：AIMessage 输入、dict 输出
