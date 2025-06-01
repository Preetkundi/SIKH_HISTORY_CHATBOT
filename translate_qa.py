# translate_qa.py
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import tqdm

def load_marien_model(src_lang="pa", tgt_lang="en"):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_batch(texts, tokenizer, model):
    """
    texts: list of strings (Punjabi). Returns list of English translations.
    """
    # Tokenize with batch
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    gen = model.generate(**batch)
    out = [tokenizer.decode(t, skip_special_tokens=True) for t in gen]
    return out

def main():
    df_pa = pd.read_csv("data/sikh_history_qa_500.csv")  # contains "question" & "answer" in Punjabi

    # Load MarianMT Punjabi→English
    tokenizer_en, model_en = load_marien_model("pa", "en")

    questions_en = []
    answers_en = []

    for q, a in tqdm.tqdm(zip(df_pa["question"], df_pa["answer"]), total=len(df_pa)):
        # translate question
        q_en = translate_batch([q], tokenizer_en, model_en)[0]
        # translate answer
        a_en = translate_batch([a], tokenizer_en, model_en)[0]

        questions_en.append(q_en)
        answers_en.append(a_en)

    df_pa["question_en"] = questions_en
    df_pa["answer_en"] = answers_en

    df_pa.to_csv("data/sikh_history_qa_500_en.csv", index=False, encoding="utf-8")
    print("Saved English translations to data/sikh_history_qa_500_en.csv")

if __name__ == "__main__":
    main()
