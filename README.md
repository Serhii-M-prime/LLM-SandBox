# Set Up
### Clone project
```bash
git clone git@github.com:Serhii-M-prime/LLM-SandBox.git
```
### Enter project folder
```bash
cd LLM-SandBox
```
### Create run env folder
```bash
mkdir venv-run
```
### Create python env for project
```bash
# might be python3 if your OS not configured for work with python
python -m venv venv-run
```
### Activate python env for run app
```bash
source venv-run/bin/activate
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
pip install -r requirements.run.txt
```
### Run chat (waiting for downloading models) (auto open in default browser on host http://localhost:8000/)
```bash
chainlit run app.py
```

# Fine tune pretrained model
Training tool https://ludwig.ai/latest/
### Create separate python env to avoid dependencies versions conflicts
```bash
cd mkdir "venv-tune"
```
```bash
# might be python3 if your OS not configured for work with python
python -m venv venv-tune/
```
```bash
source venv-tune/bin/activate
```
Install dependencies
```bash
# might be pip3 if your OS not configured for work with python
pip install -r requirements.tune.txt
```