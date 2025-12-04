import os
import json
from glob import glob

import torch
from datasets import Dataset
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import (
    TaskType,
    get_peft_model,
    PeftModel,
    IA3Config,
)
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import swanlab


# ---------------- 数据预处理与推理辅助 ---------------- #
def process_func(example):
    """将多模态对话样本转换为模型可训练的张量。"""

    MAX_LENGTH = 8192
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]

    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": "COCO Yes:"},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}

    response = tokenizer(output_content, add_special_tokens=False)
    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(inputs["input_ids"][0])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    tensor_inputs = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": torch.tensor(inputs["pixel_values"]),
        "image_grid_thw": torch.tensor(inputs["image_grid_thw"]).squeeze(0),
    }
    return tensor_inputs


def predict(messages, model):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


# ---------------- 模型与数据准备 ---------------- #
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_id = "./qwen3-vl-4b-instruct"
tokenizer = AutoTokenizer.from_pretrained(
    "qwen3-vl-4b-instruct", use_fast=False, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("qwen3-vl-4b-instruct")

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)
model.enable_input_require_grads()
model.config.use_cache = False

train_json_path = "data_vl.json"
with open(train_json_path, "r") as f:
    data = json.load(f)
    train_data = data[:-4]
    test_data = data[-4:]

with open("data_vl_train.json", "w") as f:
    json.dump(train_data, f)
with open("data_vl_test.json", "w") as f:
    json.dump(test_data, f)

train_ds = Dataset.from_json("data_vl_train.json")
train_dataset = train_ds.map(process_func)


# ---------------- Adapter Tuning 配置 ---------------- #
adapter_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        "visual_q_proj",
        "visual_k_proj",
        "visual_v_proj",
        "visual_o_proj",
        "visual_up_proj",
        "visual_down_proj",
        "visual_gate_proj",
    ],
    feedforward_modules=[
        "down_proj",
        "gate_proj",
        "up_proj",
        "visual_down_proj",
        "visual_gate_proj",
        "visual_up_proj",
    ],
)

peft_model = get_peft_model(model, adapter_config)
peft_model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./output/Qwen3-VL-4Badapter",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=5,
    save_steps=100,
    learning_rate=1e-4,
    fp16=True,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen3-VL-finetune",
    experiment_name="qwen3-vl-adapter",
    config={
        "model": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct",
        "dataset": "https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart",
        "prompt": "COCO Yes:",
        "train_data_number": len(train_data),
        "adapter_type": "IA3",
    },
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()


# ---------------- 推理 / 验证 ---------------- #
val_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=adapter_config.target_modules,
    feedforward_modules=adapter_config.feedforward_modules,
    inference_mode=True,
)

checkpoint_dirs = glob("./output/Qwen3-VL-4Badapter/checkpoint-*")
latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime) if checkpoint_dirs else "./output/Qwen3-VL-4Badapter"

val_peft_model = PeftModel.from_pretrained(
    model,
    model_id=latest_checkpoint,
    config=val_config,
)
val_peft_model.base_model.config.use_cache = True

with open("data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

test_image_list = []
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]
    origin_image_path = (
        input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": origin_image_path},
                {"type": "text", "text": "COCO Yes:"},
            ],
        }
    ]

    response = predict(messages, val_peft_model)
    messages.append({"role": "assistant", "content": response})
    print(messages[-1])

    test_image_list.append(swanlab.Image(origin_image_path, caption=response))

swanlab.log({"Prediction": test_image_list})
swanlab.finish()
