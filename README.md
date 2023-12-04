## Set Up
```bash
# Create env folder
mkdir llm-py-env
# Create python env for project
python -m venv llm-py-env # might be python3 if your OS not configured for work with python
# Enter env folder
cd llm-py-env
# Clone project
git clone git@github.com:Serhii-M-prime/LLM-SandBox.git
# Activate python env
source bin/activate
# Enter project folder
cd LLM-SandBox
# Install dependencies
pip install -r requirements.txt # might be pip3 if your OS not configured for work with python
# Run chat (waiting for downloading models) (auto open in default browser on host http://localhost:8000/)
chainlit run app.py
```