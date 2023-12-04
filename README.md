## Set Up
```bash
# Create folder
mkdir llm-py-env
# Create python env for project
python -m venv llm-py-env # might be python3 if your OS not configured for work with python
# Clone project
git clone git@github.com:Serhii-M-prime/LLM-SandBox.git
# Enter project folder
cd llm-py-env
# Activate python env
source ../bin/activate
# Install dependencies
pip install -r requirements.txt # might be pip3 if your OS not configured for work with python
# Run chat (waiting for downloading models) (auto open in default browser on host http://localhost:8000/)
chainlit run app.py
```