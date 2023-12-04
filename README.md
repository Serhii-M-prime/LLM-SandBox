# Set Up
### Create env folder
```bash
mkdir llm-py-env
```
### Create python env for project
```bash
# might be python3 if your OS not configured for work with python
python -m venv llm-py-env
```
### Enter env folder
```bash
cd llm-py-env
```
### Clone project
```bash
git clone git@github.com:Serhii-M-prime/LLM-SandBox.git
```
### Activate python env
```bash
source bin/activate
```
### Enter project folder
```bash
cd LLM-SandBox
```
### (OPTIONAL) create .env file for cache control
```bash
echo "HF_HOME=\"$PWD/.cache/huggingface\"
HF_HUB_CACHE=\"$PWD/.cache/huggingface/hub\"
HF_ASSETS_CACHE=\"$PWD/.cache/huggingface/assets\"
HF_DATASETS_CACHE=\"$PWD/.cache/huggingface/datasets\"
TRANSFORMERS_CACHE=\"$PWD/.cache/huggingface/models\"" > .env
```
### Install dependencies
```bash
# might be pip3 if your OS not configured for work with python
pip install -r requirements.txt
```
### Run chat (waiting for downloading models) (auto open in default browser on host http://localhost:8000/)
```bash
chainlit run app.py
```