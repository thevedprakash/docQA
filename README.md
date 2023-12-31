# docQA
LLM powered Physics docQA


## Clone repository

git clone https://github.com/thevedprakash/docQA.git

## Creating environment

conda create -n docQA python=3.8


## Necessary packages only 

### Installing requirements.txt
pip install -r requirements.txt

### Generating  equirements.txt
pip install pipreqs
cd ..
pipreqs docQA


## Download model

python download.py

## Create Vector Embedding of documents
python vector.py

```
You may change pdf and vector strategy as per need.
```


## Run Streamlit app.
streamlit run app.py


