import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Local Model Path
LOCAL_MODEL_PATH = r"C:\gemmi"
# 2. Reliable Public Model Fallback
FALLBACK_MODEL = "vennify/t5-base-grammar-correction"

print("Loading AI Model...")

# Check if local path exists and contains model files
if os.path.exists(LOCAL_MODEL_PATH) and os.listdir(LOCAL_MODEL_PATH):
    try:
        print(f"Loading from local path: {LOCAL_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH)
    except Exception as e:
        print(f"Local load failed: {e}. Falling back to Hugging Face model...")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(FALLBACK_MODEL)
else:
    print(f"Local folder empty or not found. Loading default model '{FALLBACK_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(FALLBACK_MODEL)

print("AI Model Loaded Successfully!")

class TextRequest(BaseModel):
    sentence: str

@app.post("/correct_grammar")
def correct_grammar(request: TextRequest):
    user_text = request.sentence
    
    input_text = f"grammar: {user_text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    
    outputs = model.generate(**inputs, max_length=128)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    is_correct = user_text.strip().lower() == corrected_text.strip().lower()
    
    return {
        "original": user_text,
        "corrected": corrected_text,
        "is_correct": is_correct
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)