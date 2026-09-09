"""
LoRA fine-tuning script for Qwen3 (8B, fits 8-12GB VRAM) on chat-style data.
Produces a merged GGUF model ready to load into LM Studio.

Requirements:
    pip install unsloth --break-system-packages
    pip install trl datasets --break-system-packages

Usage:
    1. Put your chat data in `data.jsonl` (see format below).
    2. Run: python train_qwen_lora.py
    3. Find the output GGUF at ./qwen3_finetuned_gguf/*.gguf
    4. Drag that file into LM Studio's models folder, or use "Import Model".

Data format (data.jsonl, one JSON object per line):
    {"conversations": [
        {"role": "user", "content": "What's the return policy?"},
        {"role": "assistant", "content": "You can return items within 30 days..."}
    ]}
"""

import os
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# 1. Config — tune these if you hit OOM or want faster/slower training
# ---------------------------------------------------------------------------

BASE_MODEL = "unsloth/Qwen3-8B-bnb-4bit"   # 4-bit base, fits 8-12GB VRAM
DATA_PATH = "data.jsonl"                    # your chat dataset
OUTPUT_DIR = "outputs"                      # training checkpoints
MERGED_DIR = "qwen3_finetuned_merged"       # merged fp16 model (optional)
GGUF_DIR = "qwen3_finetuned_gguf"           # final GGUF for LM Studio

MAX_SEQ_LENGTH = 2048        # lower to 1024 if VRAM is tight
LORA_RANK = 16                # 8-64; higher = more capacity, more VRAM
LORA_ALPHA = 16
NUM_EPOCHS = 3                 # adjust based on dataset size
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 2      # drop to 1 if you hit OOM
GRAD_ACCUM_STEPS = 4           # raise if you drop batch size, to keep effective batch ~8
GGUF_QUANTIZATION = "q4_k_m"   # good default for LM Studio; q5_k_m/q8_0 for higher quality

# ---------------------------------------------------------------------------
# 2. Load base model + tokenizer (4-bit quantized for VRAM efficiency)
# ---------------------------------------------------------------------------

print(f"Loading base model: {BASE_MODEL}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# ---------------------------------------------------------------------------
# 3. Attach LoRA adapters
# ---------------------------------------------------------------------------

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # saves VRAM, key for 8-12GB cards
    random_state=3407,
)

# ---------------------------------------------------------------------------
# 4. Load and format the dataset
# ---------------------------------------------------------------------------

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"'{DATA_PATH}' not found. Create it with your chat conversations "
        f"(see the format description at the top of this script)."
    )

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3",   # matches the Qwen3 chat format
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

print(f"Loading dataset from {DATA_PATH}")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"Loaded {len(dataset)} conversations")

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=SFTConfig(
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=10,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",     # lower memory footprint than default adamw
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

print("Starting training...")
trainer.train()
print("Training complete.")

# ---------------------------------------------------------------------------
# 6. Save merged model (fp16) — optional, useful if you want it outside GGUF too
# ---------------------------------------------------------------------------

print(f"Saving merged model to {MERGED_DIR}")
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")

# ---------------------------------------------------------------------------
# 7. Export directly to GGUF for LM Studio
# ---------------------------------------------------------------------------

print(f"Exporting GGUF ({GGUF_QUANTIZATION}) to {GGUF_DIR}")
model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method=GGUF_QUANTIZATION)

print("\nDone!")
print(f"Load the .gguf file from '{GGUF_DIR}/' into LM Studio via 'Import Model'.")
