# -*- coding: utf-8 -*-
"""
LLM Integration Module
Mendukung multiple LLM providers: Google Gemini, OpenAI GPT, dan Groq
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMProvider:
    """Base class untuk LLM providers"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """Google Gemini API Provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv('GEMINI_API_KEY'))
        
        if not self.api_key or self.api_key == 'your_gemini_api_key_here':
            raise ValueError("Gemini API key tidak ditemukan. Set GEMINI_API_KEY di .env")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini API initialized")
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response using Gemini"""
        try:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': max_tokens,
                    'temperature': temperature,
                }
            )
            return response.text
        except Exception as e:
            return f"Error dari Gemini: {str(e)}"


class OpenAIProvider(LLMProvider):
    """OpenAI GPT API Provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model
        
        if not self.api_key or self.api_key == 'your_openai_api_key_here':
            raise ValueError("OpenAI API key tidak ditemukan. Set OPENAI_API_KEY di .env")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            print(f"✅ OpenAI API initialized (model: {self.model})")
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response using OpenAI GPT"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Kamu adalah asisten chatbot film yang membantu dan informatif."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error dari OpenAI: {str(e)}"


class GroqProvider(LLMProvider):
    """Groq API Provider (Fast & Free!)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        super().__init__(api_key or os.getenv('GROQ_API_KEY'))
        self.model = model
        
        if not self.api_key or self.api_key == 'your_groq_api_key_here':
            raise ValueError("Groq API key tidak ditemukan. Set GROQ_API_KEY di .env")
        
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            print(f"✅ Groq API initialized (model: {self.model})")
        except ImportError:
            raise ImportError("Install groq: pip install groq")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response using Groq"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Kamu adalah asisten chatbot film yang membantu dan informatif."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error dari Groq: {str(e)}"


class LLMManager:
    """
    Manager untuk mengelola multiple LLM providers
    """
    
    PROVIDERS = {
        'gemini': GeminiProvider,
        'openai': OpenAIProvider,
        'groq': GroqProvider,
    }
    
    def __init__(self, default_provider: str = 'none'):
        """
        Initialize LLM Manager
        
        Args:
            default_provider: 'gemini', 'openai', 'groq', atau 'none'
        """
        self.default_provider = default_provider
        self.active_provider = None
        
        if default_provider != 'none':
            self.set_provider(default_provider)
    
    def set_provider(self, provider_name: str, **kwargs):
        """
        Set active LLM provider
        
        Args:
            provider_name: 'gemini', 'openai', atau 'groq'
            **kwargs: Additional arguments untuk provider (e.g., model, api_key)
        """
        if provider_name not in self.PROVIDERS:
            raise ValueError(f"Provider '{provider_name}' tidak didukung. Pilih: {list(self.PROVIDERS.keys())}")
        
        try:
            provider_class = self.PROVIDERS[provider_name]
            self.active_provider = provider_class(**kwargs)
            self.default_provider = provider_name
            print(f"🤖 Active LLM: {provider_name}")
        except Exception as e:
            print(f"⚠️ Gagal initialize {provider_name}: {e}")
            self.active_provider = None
    
    def generate_enhanced_response(
        self, 
        query: str, 
        search_results: str, 
        intent: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate enhanced response menggunakan LLM
        
        Args:
            query: User's original query
            search_results: Hasil pencarian dari search engine
            intent: Detected intent (optional)
            **kwargs: Additional generation parameters
        
        Returns:
            Enhanced response string
        """
        if not self.active_provider:
            # Fallback ke hasil pencarian biasa jika LLM tidak aktif
            return search_results
        
        # Buat prompt untuk LLM
        prompt = self._create_prompt(query, search_results, intent)
        
        try:
            response = self.active_provider.generate(prompt, **kwargs)
            return response
        except Exception as e:
            print(f"⚠️ Error generating LLM response: {e}")
            return search_results  # Fallback
    
    def _create_prompt(self, query: str, search_results: str, intent: Optional[str] = None) -> str:
        """Create prompt untuk LLM berdasarkan query dan search results"""
        
        intent_context = f"\nIntent yang terdeteksi: {intent}" if intent else ""
        
        prompt = f"""Kamu adalah asisten chatbot film yang membantu dan informatif.

Pertanyaan User: {query}{intent_context}

Informasi dari Database:
{search_results}

Tugas kamu:
1. Berikan jawaban yang natural, ramah, dan informatif berdasarkan informasi di atas
2. Jika informasi tidak lengkap, katakan dengan jujur
3. Gunakan emoji yang relevan untuk membuat jawaban lebih menarik
4. Jangan menambahkan informasi yang tidak ada di database
5. Jawab dalam bahasa Indonesia yang baik

Jawaban:"""
        
        return prompt
    
    def is_active(self) -> bool:
        """Check apakah LLM provider aktif"""
        return self.active_provider is not None
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get informasi tentang active provider"""
        if not self.active_provider:
            return {'status': 'inactive', 'provider': 'none'}
        
        return {
            'status': 'active',
            'provider': self.default_provider,
            'model': getattr(self.active_provider, 'model', 'gemini-1.5-flash')
        }


# === Testing ===
if __name__ == "__main__":
    # Fix encoding untuk Windows
    import sys
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("🧪 Testing LLM Integration\n")
    
    # Test dengan mode none (fallback)
    print("=" * 50)
    print("Test 1: No LLM (Fallback Mode)")
    print("=" * 50)
    manager = LLMManager(default_provider='none')
    
    test_query = "Siapa sutradara film Inception?"
    test_results = "Film Inception disutradarai oleh Christopher Nolan pada tahun 2010."
    
    response = manager.generate_enhanced_response(test_query, test_results)
    print(f"Query: {test_query}")
    print(f"Response: {response}\n")
    
    # Test dengan LLM (jika API key tersedia)
    print("=" * 50)
    print("Test 2: With LLM (if API key available)")
    print("=" * 50)
    
    # Coba Groq dulu (paling mudah dapat free API key)
    try:
        manager.set_provider('groq')
        response = manager.generate_enhanced_response(
            test_query, 
            test_results,
            intent='sutradara',
            max_tokens=200
        )
        print(f"Query: {test_query}")
        print(f"Enhanced Response:\n{response}\n")
    except Exception as e:
        print(f"⚠️ Groq tidak tersedia: {e}\n")
    
    # Info
    print("=" * 50)
    print("Provider Info:")
    print("=" * 50)
    print(manager.get_provider_info())
    
    print("\n💡 Tips:")
    print("1. Dapatkan Groq API key (GRATIS & CEPAT): https://console.groq.com/keys")
    print("2. Dapatkan Gemini API key (GRATIS): https://makersuite.google.com/app/apikey")
    print("3. Simpan di file .env")
