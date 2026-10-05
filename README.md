# hw3-wevans2024

Homework 3: a command-line LangChain IT utility agent with Python calculations,
DNS lookups, and HTTP/HTTPS connectivity checks.

## Run the agent

From the repository root, create and activate a virtual environment, then
install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r hw3/requirements.txt
```

Set the Google Generative AI credentials and model name in the environment:

```powershell
$env:GOOGLE_API_KEY = "your-api-key"
$env:GOOGLE_MODEL = "gemini-2.0-flash"
```

Start the interactive agent:

```powershell
python hw3/app.py
```

Type `exit` or `quit` to end the session. The Python REPL tool executes Python
code, so only use it with trusted input.
