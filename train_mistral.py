import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset


# Check GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


model_name = "kevin36524/SVDfy_Mistral-7B_r32"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    logging_dir="./logs",
    num_train_epochs=3,
    save_strategy="epoch",
)

# Load the dataset from the JSONL file
dataset = load_dataset("json", data_files="trainingSet.jsonl")

# Preprocess the dataset to create the "text" column
def preprocess_function(examples):
    return {"text": [text[0] for text in examples["text"]]}

# Apply the preprocessing function to the dataset
dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

print(dataset["train"].features)

# Create a list of linear layer names to finetune
linear_layer_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]

# Define a function to freeze all parameters except linear layers and lm_head
def freeze_params(model):
    for name, param in model.named_parameters():
        if not any(linear_name in name for linear_name in linear_layer_names) and "lm_head" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

# Freeze parameters
freeze_params(model)

# Verify that some parameters are trainable
trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
print(f"Trainable parameters: {trainable_params}")

# Create the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_mistral")
