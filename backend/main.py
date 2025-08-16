from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import tiktoken
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Prompt Token Optimizer", version="1.0.0")

# Configure CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}
tokenizers = {}

LANGUAGE_PAIRS = {
    "English": "en",
    "Chinese": "zh",
    "Japanese": "ja", 
    "Spanish": "es"
}

def load_models():
    """Load translation models on startup"""
    try:
        # Load models for each language
        for lang_name, lang_code in LANGUAGE_PAIRS.items():
            if lang_name == "English":
                continue  # English is the source language
            
            model_name = f"Helsinki-NLP/opus-mt-en-{lang_code}"
            logger.info(f"Loading model: {model_name}")
            
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            models[lang_name] = model
            tokenizers[lang_name] = tokenizer
            
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Fallback: continue without models for now

class AnalyzeRequest(BaseModel):
    text: str

class TranslationResult(BaseModel):
    language: str
    translation: str
    token_count: int
    percentage_savings: float

class AnalyzeResponse(BaseModel):
    original_text: str
    original_tokens: int
    translations: List[TranslationResult]
    best_language: str

@app.on_event("startup")
async def startup_event():
    """Load models when the app starts"""
    load_models()

@app.get("/")
async def root():
    return {"message": "Prompt Token Optimizer API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": len(models)}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Analyze text and provide translations with token counts"""
    try:
        # Count tokens in original English text
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        original_tokens = len(encoding.encode(request.text))
        
        if original_tokens == 0:
            raise HTTPException(status_code=400, detail="Input text is empty")
        
        translations = []
        
        # Add English as baseline
        translations.append(TranslationResult(
            language="English",
            translation=request.text,
            token_count=original_tokens,
            percentage_savings=0.0
        ))
        
        # Generate translations for other languages
        for lang_name, model in models.items():
            try:
                tokenizer = tokenizers[lang_name]
                
                # Tokenize and translate
                inputs = tokenizer(request.text, return_tensors="pt", padding=True)
                translated = model.generate(**inputs)
                translation = tokenizer.decode(translated[0], skip_special_tokens=True)
                
                # Count tokens in translation
                translation_tokens = len(encoding.encode(translation))
                
                # Calculate percentage savings
                percentage_savings = ((original_tokens - translation_tokens) / original_tokens) * 100
                
                translations.append(TranslationResult(
                    language=lang_name,
                    translation=translation,
                    token_count=translation_tokens,
                    percentage_savings=percentage_savings
                ))
                
            except Exception as e:
                logger.error(f"Translation error for {lang_name}: {e}")
                # Skip this language if translation fails
                continue
        
        # Find best language (minimum token count)
        best_translation = min(translations, key=lambda x: x.token_count)
        best_language = best_translation.language
        
        return AnalyzeResponse(
            original_text=request.text,
            original_tokens=original_tokens,
            translations=translations,
            best_language=best_language
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 