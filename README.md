# Sutra

Convert unstructured files (PDF, DOCX, JPG, PNG) to structured data using AI.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Install Tesseract OCR:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `apt-get install tesseract-ocr`

3. Create a `.env` file in the project root:
   ```
   LLM_API_KEY=YOUR_API_KEY_HERE
   ```
   Get an API key from OpenRouter or another LLM provider.

## Running the app

```
streamlit run app.py
```

## Usage

1. Upload one or more files (PDF, DOCX, JPG, PNG)
2. Enter a description of the data structure you want to extract
3. Click "Process Files"
4. View the resulting table and download in your preferred format 