import math
import os
from glob import glob
from dataclasses import dataclass, field

import json
import torch
import torch.nn as nn
from datasets import Dataset
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info  # Qwen3-VL专用视觉预处理工具
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer  # 继承LoRA基础层实现自定义MOT LoRA
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import swanlab  # 实验监控工具，替代TensorBoard


def process_func(example):
    """
    多模态数据集预处理函数：适配Qwen3-VL的图文输入格式，生成训练所需的input_ids/labels等
    Args:
        example: 单条数据集样本，格式为{"conversations": [{"value": 图文prompt}, {"value": 目标caption}]}
    Returns:
        格式化后的训练样本，包含input_ids/attention_mask/labels/pixel_values/image_grid_thw
    """
    MAX_LENGTH = 8192  # Qwen3-VL支持的最大上下文长度
    conversation = example["conversations"]
    input_content = conversation[0]["value"]  # 用户侧输入（含图片路径）
    output_content = conversation[1]["value"]  # 模型目标输出（COCO caption）
    
    # 提取图片路径：从<|vision_start|>和<|vision_end|>标签中解析
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    
    # 构造Qwen3-VL标准多模态对话格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,  # 固定图像尺寸，平衡显存和精度
                    "resized_width": 280,
                },
                {"type": "text", "text": "COCO Yes:"},  # 固定prompt前缀，引导模型生成caption
            ],
        }
    ]
    
    # 应用Qwen3-VL的chat template，生成模型可识别的文本格式
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 预处理视觉信息：将图片路径转为模型可接受的pixel_values
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 文本+视觉特征编码，转为tensor格式
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # tensor转list，便于后续拼接输入和输出序列
    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs  # 指令部分（图文prompt）
    
    # 编码目标输出（caption），不添加特殊token（避免重复）
    response = tokenizer(f"{output_content}", add_special_tokens=False)

    # 拼接输入+输出序列：构建自回归训练的完整序列
    input_ids = instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    # 构建labels：指令部分设为-100（不计算损失），仅输出部分计算损失
    labels = (
        [-100] * len(instruction["input_ids"][0])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    
    # 超长序列截断：避免超出模型最大上下文长度
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    # 转回tensor格式，适配模型训练输入
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs["pixel_values"] = torch.tensor(inputs["pixel_values"])
    # 调整image_grid_thw维度：(1, h, w) → (h, w)，适配模型视觉输入要求
    inputs["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
    }


def predict(messages, model):
    """
    多模态推理函数：基于输入的图文消息生成caption
    Args:
        messages: Qwen3-VL标准对话格式的消息列表（含图片路径和文本prompt）
        model: 加载完成的PEFT模型（MOT LoRA微调后）
    Returns:
        模型生成的图片描述文本
    """
    # 应用chat template生成推理文本格式
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 预处理视觉信息
    image_inputs, video_inputs = process_vision_info(messages)
    # 编码输入特征并移至GPU
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出：max_new_tokens=128限制生成长度，避免冗余
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    # 截断输入部分，仅保留模型生成的token
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # 解码生成的token为文本，跳过特殊token
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


# -------------------------- 量化配置：4-bit NF4双量化 --------------------------
# 核心作用：降低显存占用，使4B模型能在8G显存的GPU上训练
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用4-bit量化（相比8-bit进一步降低显存）
    bnb_4bit_use_double_quant=True,  # 双量化：对量化参数再量化，进一步压缩
    bnb_4bit_quant_type="nf4",  # NF4量化类型：适配自然语言的4-bit量化方案（优于FP4）
    bnb_4bit_compute_dtype=torch.float16,  # 计算时用FP16，平衡精度和速度
)

# -------------------------- 模型/处理器加载 --------------------------
# 本地加载Qwen3-VL-4B-Instruct模型（避免重复下载）
model_id = "./qwen3-vl-4b-instruct"
# 加载tokenizer：use_fast=False适配Qwen自定义tokenizer，trust_remote_code=True加载模型自定义代码
tokenizer = AutoTokenizer.from_pretrained("qwen3-vl-4b-instruct", use_fast=False, trust_remote_code=True)
# 加载多模态处理器：整合文本tokenizer和视觉处理器
processor = AutoProcessor.from_pretrained("qwen3-vl-4b-instruct")

# 加载多模态模型：结合量化配置，自动分配设备（GPU优先）
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配模型层到GPU/CPU，适配显存
    dtype=torch.float16,
    trust_remote_code=True,
)

# 梯度检查点相关配置：降低训练时的显存占用
model.enable_input_require_grads()  # 梯度检查点必须启用，否则LoRA梯度无法计算
model.config.use_cache = False  # 训练时禁用cache（推理时启用）
model.gradient_checkpointing_enable()  # 启用梯度检查点，以时间换显存

# -------------------------- 数据集处理 --------------------------
# 读取原始数据集，拆分为训练集（最后4条为测试集）
train_json_path = "data_vl.json"
with open(train_json_path, "r") as f:
    data = json.load(f)
    train_data = data[:-4]  # 训练集：除最后4条外的所有样本
    test_data = data[-4:]   # 测试集：最后4条样本（小批量验证）

# 保存拆分后的数据集，便于复用
with open("data_vl_train.json", "w") as f:
    json.dump(train_data, f)
with open("data_vl_test.json", "w") as f:
    json.dump(test_data, f)

# 加载训练集并应用预处理函数
train_ds = Dataset.from_json("data_vl_train.json")
train_dataset = train_ds.map(process_func)  # 每条样本都会经过process_func处理

# -------------------------- MOT LoRA核心定义 --------------------------
@dataclass
class MotLoraConfig(LoraConfig):
    """
    自定义MOT LoRA配置类：继承LoraConfig，新增Mixture-of-Tokens相关参数
    MOT (Mixture-of-Tokens) LoRA：在普通LoRA基础上增加1D卷积的Token混合模块，增强序列建模能力
    """
    mix_kernel_size: int = field(default=3, metadata={"help": "Mixture-of-Tokens 1D卷积核大小（奇数保证对称 padding）"})
    mix_dropout: float = field(default=0.05, metadata={"help": "token混合后的dropout，防止过拟合"})

    # 重写属性：标识非prompt learning/adaption prompt（PEFT库内部判断用）
    @property
    def is_prompt_learning(self):
        return False

    @property
    def is_adaption_prompt(self):
        return False


class MotLoraLayer(LoraLayer):
    """
    MOT LoRA核心层：继承LoraLayer，在普通LoRA基础上增加Token混合模块
    核心逻辑：普通LoRA输出（delta） + 1D卷积Token混合输出（mixed），叠加到原层输出上
    """
    def __init__(self, base_layer, config: MotLoraConfig):
        super().__init__(base_layer, config)
        # 基础LoRA参数：输入/输出维度、秩r、缩放因子
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = config.r
        self.scaling = config.lora_alpha / config.r  # LoRA缩放因子（alpha/r）
        
        # 基础LoRA的dropout和线性层
        self.dropout = nn.Dropout(config.lora_dropout)
        self.mix_dropout = nn.Dropout(config.mix_dropout)  # MOT模块的dropout
        self.adapter_A = nn.Linear(self.in_features, self.r, bias=False)  # LoRA A层（降维）
        self.adapter_B = nn.Linear(self.r, self.out_features, bias=False)  # LoRA B层（升维）
        
        # 初始化：A层用kaiming_uniform（适合ReLU类激活），B层初始化为0（保证初始等价于原模型）
        nn.init.kaiming_uniform_(self.adapter_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_B.weight)
        
        # MOT核心：1D卷积实现Token混合（跨序列维度的特征融合）
        self.token_mixer = nn.Conv1d(
            in_channels=self.out_features,  # 输入通道=隐藏层维度
            out_channels=self.out_features, # 输出通道=隐藏层维度（通道数不变）
            kernel_size=config.mix_kernel_size,  # 卷积核大小（3表示融合当前+前后1个token）
            padding=config.mix_kernel_size // 2,  # 对称padding，保证序列长度不变
            groups=1,  # 普通卷积（非分组卷积）
            bias=False,
        )
        # 卷积层初始化为0：保证初始状态下MOT模块无贡献，等价于普通LoRA
        nn.init.zeros_(self.token_mixer.weight)

    def forward(self, x):
        """
        前向传播：原层输出 + LoRA增量 + MOT Token混合增量
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, hidden_dim) 或 (batch_size, hidden_dim)
        Returns:
            叠加增量后的输出张量
        """
        # 原层输出（冻结的预训练模型层）
        base_out = self.base_layer(x)
        # 若LoRA合并/禁用，直接返回原层输出
        if self.merged or self.disable_adapters:
            return base_out

        # 普通LoRA增量计算：x → A → dropout → B → 缩放
        delta = self.adapter_B(self.dropout(self.adapter_A(x))) * self.scaling

        # MOT Token混合：适配不同维度的输入（3D/2D/其他）
        if delta.dim() == 3:  # 标准序列输入：(batch, seq_len, hidden)
            # 1D卷积要求输入格式为 (batch, hidden, seq_len)，需转置
            mixed = self.token_mixer(delta.permute(0, 2, 1)).permute(0, 2, 1)
        elif delta.dim() == 2:  # 无序列维度：(batch, hidden)
            # 扩展维度后卷积，再还原
            mixed = self.token_mixer(delta.unsqueeze(-1)).squeeze(-1)
        else:  # 其他维度（如多模态融合后的特殊形状）
            reshaped = delta.view(-1, delta.size(-1)).unsqueeze(-1)
            mixed = self.token_mixer(reshaped).squeeze(-1)
            mixed = mixed.view_as(delta)

        # MOT增量添加dropout，最终输出 = 原层输出 + LoRA增量 + MOT增量
        mixed = self.mix_dropout(mixed)
        return base_out + delta + mixed


def get_mot_peft_model(model, config: MotLoraConfig):
    """
    封装PEFT模型构建函数：将自定义MOT LoRA层绑定到PEFT框架
    Args:
        model: 预训练基座模型
        config: MOT LoRA配置
    Returns:
        加载MOT LoRA的PEFT模型
    """
    config.adapter_layer = MotLoraLayer  # 指定PEFT使用自定义的MOT LoRA层
    return get_peft_model(model, config)

# -------------------------- MOT LoRA配置初始化 --------------------------
config = MotLoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型：自回归语言建模（图文生成）
    # 目标模块：文本+视觉的注意力层（q/k/v/o），仅微调这些层的MOT LoRA
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "visual_q_proj",
        "visual_k_proj",
    ],
    inference_mode=False,  # 训练模式（推理时设为True）
    r=8,  # LoRA秩（越小参数越少，8是平衡参数和效果的常用值）
    lora_alpha=16,  # LoRA alpha（缩放因子，通常设为2*r）
    lora_dropout=0.05,  # 基础LoRA的dropout
    mix_kernel_size=3,  # MOT卷积核大小（3是最小的奇数，保证对称融合）
    mix_dropout=0.05,  # MOT模块的dropout
    bias="none",  # 不训练bias（减少参数，避免过拟合）
)

# 构建MOT LoRA模型，打印可训练参数（面试重点：展示参数效率）
peft_model = get_mot_peft_model(model, config)
peft_model.print_trainable_parameters()  # 输出格式：trainable params / all params / trainable%

# -------------------------- 训练参数配置 --------------------------
args = TrainingArguments(
    output_dir="./output/Qwen3-VL-4Bmotlora",  # 模型保存路径
    per_device_train_batch_size=1,  # 单卡batch size（8G显存下设为1）
    gradient_accumulation_steps=8,  # 梯度累积：有效batch size = 1*8=8
    logging_steps=10,  # 每10步打印一次日志
    logging_first_step=5,  # 第5步开始打印日志
    num_train_epochs=5,  # 训练轮数（小数据集设5轮，避免过拟合）
    save_steps=100,  # 每100步保存一次checkpoint
    learning_rate=1e-4,  # 学习率（LoRA微调常用1e-4，比全量微调大）
    fp16=True,  # 启用FP16训练，加速且降低显存
    save_on_each_node=True,  # 多节点训练时每个节点保存checkpoint（单节点无影响）
    gradient_checkpointing=True,  # 启用梯度检查点，进一步降低显存
    report_to="none",  # 禁用其他日志工具（仅用SwanLab）
)

# -------------------------- SwanLab实验监控 --------------------------
swanlab_callback = SwanLabCallback(
    project="Qwen3-VL-finetune",  # 项目名称
    experiment_name="qwen3-vl-motlora",  # 实验名称
    config={  # 记录关键配置，便于后续对比不同LoRA方案
        "model": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct",
        "dataset": "https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart",
        "github": "https://github.com/datawhalechina/self-llm",
        "prompt": "COCO Yes: ",
        "train_data_number": len(train_data),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "mot_kernel_size": 3,  # 新增MOT参数记录
        "mix_dropout": 0.05,   # 新增MOT参数记录
    },
)

# -------------------------- 训练器初始化与训练 --------------------------
trainer = Trainer(
    model=peft_model,  # MOT LoRA模型
    args=args,  # 训练参数
    train_dataset=train_dataset,  # 训练集
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),  # 数据拼接器（处理padding）
    callbacks=[swanlab_callback],  # 监控回调
)

# 开始训练（核心入口）
trainer.train()

# -------------------------- 推理验证 --------------------------
# 推理时的MOT LoRA配置（inference_mode=True，禁用训练相关逻辑）
val_config = MotLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "visual_q_proj",
        "visual_k_proj",
    ],
    inference_mode=True,  # 推理模式：禁用dropout，固定参数
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    mix_kernel_size=3,
    mix_dropout=0.05,
    bias="none",
)

# 找到最新的checkpoint（优先加载最新保存的模型）
checkpoint_dirs = glob("./output/Qwen3-VL-4Bmotlora/checkpoint-*")
latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime) if checkpoint_dirs else "./output/Qwen3-VL-4Bmotlora"
# 加载微调后的MOT LoRA模型（推理模式）
val_peft_model = PeftModel.from_pretrained(model, model_id=latest_checkpoint, config=val_config)

# 加载测试集，批量推理并记录结果
with open("data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

test_image_list = []
for item in test_dataset:
    # 提取测试集图片路径
    input_image_prompt = item["conversations"][0]["value"]
    origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

    # 构造推理的图文消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": origin_image_path},
                {"type": "text", "text": "COCO Yes:"},
            ],
        }
    ]

    # 调用推理函数生成caption
    response = predict(messages, val_peft_model)
    messages.append({"role": "assistant", "content": f"{response}"})
    print(messages[-1])  # 打印推理结果

    # 记录图片+caption到SwanLab，便于可视化查看效果
    test_image_list.append(swanlab.Image(origin_image_path, caption=response))

# 日志记录推理结果（SwanLab可视化）
swanlab.log({"Prediction": test_image_list})

# 结束SwanLab实验
swanlab.finish()