import transformers
import torch

model_id = "aaditya/OpenBioLLM-Llama3-70B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="auto",
)

unstructured_text = """
John is a 45-year-old male with a history of hypertension and type 2 diabetes mellitus. He has been on medication for both conditions for the past 10 years. His current medications include Metformin, Lisinopril, and Atorvastatin.

John presents with a 2-week history of intermittent chest pain that occurs with exertion and is relieved by rest. He describes the pain as a pressure-like sensation in the center of his chest, sometimes radiating to his left arm. He denies any shortness of breath, nausea, or sweating.
"""

messages = [
    {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."},
    {"role": "user", "content": f"extract patient's name and age, disease, medication, treatment from the provided clinical text in a json format. if there is no information for a field put an empty string. the clinical text is: {unstructured_text}"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.0,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
