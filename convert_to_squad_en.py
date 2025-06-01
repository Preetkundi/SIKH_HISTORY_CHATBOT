# convert_to_squad_en.py
import pandas as pd
import json
import random
import os

def build_squad_from_df(df_subset):
    """
    Given a DataFrame with columns `question_en` and `answer_en`,
    build one SQuAD-like dict under a dummy "title" = "sikh_history_en".
    """
    paragraphs = []
    for idx, row in df_subset.iterrows():
        context = row["answer_en"]  # we’ll use the answer itself as a minimal “context”
        qa_entry = {
            "id": f"sikh_{idx}",
            "question": row["question_en"],
            "answers": [
                {
                    "text": row["answer_en"],
                    "answer_start": 0  # since answer_en == context, start=0
                }
            ],
            "is_impossible": False
        }
        paragraphs.append({
            "context": context,
            "qas": [qa_entry]
        })
    return {
        "title": "sikh_history_en",
        "paragraphs": paragraphs
    }

def main():
    df_en = pd.read_csv("data/sikh_history_qa_500_en.csv")
    indices = list(df_en.index)
    random.seed(42)
    random.shuffle(indices)

    split = int(0.8 * len(indices))  # 400 / 100 split
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_df = df_en.loc[train_idx].reset_index(drop=True)
    val_df   = df_en.loc[val_idx].reset_index(drop=True)

    squad_train = {"version": "1.1", "data": [build_squad_from_df(train_df)]}
    squad_val   = {"version": "1.1", "data": [build_squad_from_df(val_df)]}

    os.makedirs("data", exist_ok=True)
    with open("data/squad_train_en.json", "w", encoding="utf-8") as f:
        json.dump(squad_train, f, ensure_ascii=False, indent=2)
    with open("data/squad_val_en.json", "w", encoding="utf-8") as f:
        json.dump(squad_val, f, ensure_ascii=False, indent=2)

    print("Saved:\n - data/squad_train_en.json\n - data/squad_val_en.json")

if __name__ == "__main__":
    main()
