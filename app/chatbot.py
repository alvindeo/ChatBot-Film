import sys
import os

# Tambahkan path src ke sys.path agar bisa import modul
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from search_engine import chatbot, demo  # Import instance dan UI dari search_engine.py

if __name__ == "__main__":
    demo.launch(share=True)