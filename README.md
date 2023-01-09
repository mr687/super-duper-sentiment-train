# Superb Sentiment Analyzation

## Requirements
- Python 3.6

## Getting Started
1. Clone the repository

```bash
git clone https://github.com/mr687/super-duper-sentiment-train.git
cd super-duper-sentiment-train
```

2. Setup a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the requirements

```bash
pip install -r requirements.txt
```

4. Run the application

```bash
# Open this project with vscode
code .
```

## Folder Structure
```
├── dataset 
│   ├── raw 
│   │   ├── *.csv 
│   ├── processed	
│   │   ├── *.csv 
│   ├── wordlist 
│   │   ├── *.{csv,txt} 
├── notebooks 
│   ├── {incremental}-*.ipynb 
├── requirements.txt 
```

- Dataset

The dataset folder contains the raw data, processed data, and wordlist.

> Raw Data: The raw data is the data that is downloaded from the internet. The raw data is stored in the raw folder.

> Processed Data: The processed data / data outputs is the data that is cleaned and ready to be used. The processed data is stored in the processed folder.

> Wordlist: The wordlist is the list of words that are used to determine the sentiment of a sentence. The wordlist is stored in the wordlist folder.

- Notebooks

The notebooks folder contains the notebooks that are used to process the data. The notebooks are stored in the notebooks folder.

> The notebook file should be runned in incremental order. The incremental number is the order of the notebook.

- Requirements.txt

The requirements.txt file contains the list of requirements that are needed to run the application.