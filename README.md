# pdf2dataset

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b)](https://streamlit.io/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI-powered toolkit that converts unstructured **PDF, DOCX, and image** files into clean, structured datasets (CSV / Excel / JSON).

---

## ✨ Features

• Automatic document-type detection and smart extraction hints  
• Table, text, image, formula & question extraction  
• Interactive Streamlit web interface – no code required  
• Download results instantly in CSV, Excel or JSON  
• Pluggable LLM backend (OpenAI, OpenRouter, …)  
• Pure-Python – runs on Windows, macOS & Linux

---

## 🚀 Quick start

```bash
# 1) Clone the repo
$ git clone https://github.com/<your-username>/pdf2dataset.git
$ cd pdf2dataset

# 2) Create a virtual environment (recommended)
$ python -m venv .venv && source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1

# 3) Install dependencies
$ pip install -r requirements.txt

# 4) Install Tesseract OCR engine (required for image OCR)
#    – Windows: https://github.com/UB-Mannheim/tesseract/wiki
#    – macOS  : brew install tesseract
#    – Linux  : sudo apt-get install tesseract-ocr

# 5) Add your LLM API key (and optional settings) to .env
$ cp .env.example .env  # then edit .env

# 6) Launch the app
$ streamlit run app.py
```

Open the printed local URL in your browser, upload files and watch the magic ✨.

---

## 🔧 Configuration

Environment variables are loaded from **.env**. Minimum:

```ini
LLM_API_KEY=sk-...
```

Optional:

```ini
LLM_API_PROVIDER=openai        # or openrouter
LLM_MODEL=gpt-3.5-turbo        # override default model
PORT=8501                      # Streamlit port
```

---

## 🖥️ Usage workflow

1. Upload one or more PDF/DOCX/images.  
2. Describe the data structure you need (e.g. “extract invoices: invoice number, date, total”).  
3. Click **Process Files**.  
4. Explore the table preview, inspect extracted content in separate tabs.  
5. Download your dataset in the desired format.

---

## 🧐 Troubleshooting

Common setup issues and solutions.

### 1. `streamlit: command not found`

-   **Cause**: The Python packages were not installed, or your shell is not using your virtual environment.
-   **Solution**: Ensure your virtual environment is active. You should see `(.venv)` or a similar prefix in your terminal prompt. If not, activate it:
    ```bash
    # On Windows (Git Bash)
    source .venv/Scripts/activate

    # On Windows (CMD)
    .venv\Scripts\activate.bat

    # On macOS / Linux
    source .venv/bin/activate
    ```
    Then, install the dependencies again:
    ```bash
    pip install -r requirements.txt
    ```

### 2. `Permission denied` / `OSError: [WinError 5]` on `pip install`

-   **Cause**: You are trying to install packages to a system-level Python directory without administrator permissions. This usually happens when a virtual environment is **not active**.
-   **Solution**: Activate your virtual environment as described above. All `pip` commands will then target the local `.venv/` folder, which you have permission to write to.

### 3. Virtual Environment Not Activating Correctly (Windows / Git Bash)

-   **Symptom**: You see `(.venv)` in your prompt, but `which python` still points to your global Python (e.g., `/c/Python311/python`), and installations still fail with permission errors.
-   **Cause**: A shell configuration issue is preventing the `PATH` from updating correctly.
-   **Solution**: Bypass the shell's `PATH` by calling the executables from your virtual environment directly by their full path. This is a foolproof method.

    Instead of `pip install ...`, run:
    ```bash
    # Note the leading ./ which is important in bash
    ./.venv/Scripts/python.exe -m pip install -r requirements.txt
    ```

    Instead of `streamlit run app.py`, run:
    ```bash
    ./.venv/Scripts/python.exe -m streamlit run app.py
    ```

---

## 🤝 Contributing

Pull requests are welcome! If you’d like to add a new feature or fix a bug:

1. Fork the repository & create your branch (`git checkout -b feature/awesome`)
2. Commit your changes with clear messages
3. Push to the branch (`git push origin feature/awesome`)
4. Open a Pull Request

Please ensure linting (`flake8`) passes and add/adjust tests where relevant.

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

Built with [Streamlit](https://streamlit.io/), [pdfplumber](https://github.com/jsvine/pdfplumber), and the power of modern LLMs. 