# fine_tune_sikh_en.py

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

MODEL_NAME = "bert-base-uncased"  # or "roberta-base", etc.

def prepare_train_features(examples, tokenizer):
    questions = []
    contexts = []
    start_positions = []
    end_positions = []

    # Adapted for SQuAD-like structure with "paragraphs" field
    for paragraphs in examples["paragraphs"]:
        para = paragraphs[0]
        context = para["context"]
        for qa in para["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]["text"]
            questions.append(question)
            contexts.append(context)
            start_positions.append(0)
            tokenized_answer = tokenizer(answer, add_special_tokens=False)
            end_positions.append(len(tokenized_answer["input_ids"]) - 1)

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    data_files = {
        "train": "data/squad_train_en.json",
        "validation": "data/squad_val_en.json"
    }
    raw_datasets = load_dataset("json", data_files=data_files, field="data")

    train_dataset = raw_datasets["train"].map(
        lambda examples: prepare_train_features(examples, tokenizer),
        batched=True,
        remove_columns=["paragraphs"]
    )
    val_dataset = raw_datasets["validation"].map(
        lambda examples: prepare_train_features(examples, tokenizer),
        batched=True,
        remove_columns=["paragraphs"]
    )

    training_args = TrainingArguments(
        output_dir="./sikh_qa_en_model",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        learning_rate=2e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model("sikh_qa_en_model")
    tokenizer.save_pretrained("sikh_qa_en_model")

    print("✅ English QA model fine-tuned and saved to 'sikh_qa_en_model/'")

if __name__ == "__main__":
    main()
