from transformers import pipeline

pipe = pipeline("text-generation", model="sberbank-ai/rugpt3small_based_on_gpt2")
while True:
    prompt = input("📝 Введи начало текста (или 'exit' для выхода): ")
    if prompt.lower() == "exit":
        break
    result = pipe(prompt, max_length=100, truncation=True)
    print("\n💬 Ответ:\n" + result[0]["generated_text"])
