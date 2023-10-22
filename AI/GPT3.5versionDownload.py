import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
# Model va tokenizer nomi
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)
# Foydalanuvchidan so'rov olish va javob bermoq
while True:
    user_input = input("Siz: ")
    if user_input.lower() == "exit":
        break
    # Tokenlash va modelga o'qitish
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model(**inputs)
    # Javobni chiqarish
    response = tokenizer.decode(outputs.logits[:, -1, :])
    print("Chatbot:", response)