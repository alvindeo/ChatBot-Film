import sys
import os
import io

# 1. Force Unbuffered Output (Agar progress bar muncul realtime)
# Re-open stdout/stderr dengan buffering=1 (line buffered) atau 0 (unbuffered)
try:
    if sys.platform == 'win32':
        # Windows specific fix
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', write_through=True)
    else:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
except Exception as e:
    print(f"Note: Output buffering setup failed: {e}")

# 2. Setup path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# 3. Import modul (Loading akan berjalan di sini)
print("🚀 Starting Chatbot Application...")
from search_engine import chatbot, demo

if __name__ == "__main__":
    # Removed share=True to prevent hanging
    demo.launch()