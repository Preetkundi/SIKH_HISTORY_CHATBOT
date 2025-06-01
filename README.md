
---

## 🗂️ Data Collection & Preprocessing

1. **Curated Q&A Dataset:**  
   Collected ~500 questions and answers about Sikh history from trusted sources such as [SikhiWiki](https://www.sikhiwiki.org/) and other educational sites, and saved in `data/sikh_history_qa_500.csv` (in both English and Punjabi).

2. **SQuAD Conversion:**  
   Converted the original CSV Q&A into [SQuAD format](https://rajpurkar.github.io/SQuAD-explorer/) JSON for question-answering fine-tuning:  
   - `squad_train_en.json`
   - `squad_val_en.json`

3. **Translation:**  
   Used Google Translate API and custom scripts (`translate_qa.py`) to create Punjabi and English versions for multi-lingual experiments.

---

## 🤖 Model Fine-Tuning (Optional/For Demonstration)

- **Script:** `fine_tune_sikh_en.py`
- Fine-tunes a pre-trained QA model (`bert-base-uncased`) on Sikh history SQuAD data.
- Saves the model as `sikh_qa_en_model/` (can be loaded in `pipeline.py` for custom serving).
- Demonstrates practical NLP model adaptation for a domain-specific knowledge base.

---

## 🌐 Web App and API Pipeline

- **Frontend:**  
  `templates/index.html` is a modern, mobile-friendly chat UI with internal scroll and Punjabi interface.

- **Backend:**  
  - `app.py` runs a Flask server and connects the web form to the NLP pipeline.
  - `pipeline.py` handles:
    1. **Translating** user questions (Punjabi → English)
    2. **Generating answers** using an LLM API (Gemini, OpenAI, etc.)  
       *(Or, for demo, the fine-tuned model on local SQuAD data)*
    3. **Translating answer** back to Punjabi
    4. **Cleaning and formatting** the answer before sending to the user

---

## 🚀 Running the Project

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Set up your API keys:**  
   - For Gemini: Add your Google API key as an environment variable
   - For OpenAI: Add your OpenAI API key as an environment variable

3. **Run the Flask app:**
    ```bash
    python app.py
    ```
    Then open [http://localhost:5000](http://localhost:5000) in your browser!

---

## 📦 Main Technologies Used

- **Python** (Flask, Transformers, pandas, requests)
- **HuggingFace Transformers** (for fine-tuning, translation, and model serving)
- **Google Gemini/OpenAI API** (for LLM-powered answers)
- **HTML/CSS/JS** (for a modern chat interface)
- **SQuAD format** (for standard QA data)

---

## 📝 Notes

- You can **extend** the project to use only the fine-tuned BERT model for offline/local answering (no API needed) by switching the logic in `pipeline.py`.
- All dataset and conversion scripts are included for transparency and reproducibility.
- The chatbot is tailored for **Punjabi language** and Sikh history domain, but can be adapted for other languages or topics.

---



---

