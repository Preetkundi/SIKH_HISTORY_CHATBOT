import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyAjO-XSTJfOPtj8VogKlYpN2ykfdeFUoec"
genai.configure(api_key=GEMINI_API_KEY)

def answer_punjabi_question(question_pa: str) -> str:
    system_prompt = (
        "You are an expert Sikh historian. "
        "Answer all questions ONLY in Punjabi. "
        "If you do not know the answer, reply with: 'ਮਾਫ ਕਰਨਾ, ਮੈਨੂੰ ਇਸਦਾ ਉੱਤਰ ਨਹੀਂ ਆਉਂਦਾ।'"
        "Give factual and detailed answers for Sikh history topics."
    )
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        chat = model.start_chat(history=[
            {"role": "user", "parts": [system_prompt]},
        ])
        response = chat.send_message(question_pa)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gemini API failed: {e}")
        return "ਮਾਫ ਕਰਨਾ, ਕੁਝ ਗਲਤ ਹੋ ਗਿਆ ਹੈ।"

if __name__ == "__main__":
    q_pa = "ਗੁਰੂ ਨਾਨਕ ਦੇਵ ਜੀ ਕਿੱਥੇ ਜਨਮੇ ਸਨ?"
    print("Q:", q_pa)
    print("A:", answer_punjabi_question(q_pa))
