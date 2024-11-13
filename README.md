
# BiLSTM NER using CRF

This project performs Named Entity Recognition (NER) using a BiLSTM-CRF model. The model has been trained on the CoNLL-2003 dataset and can recognize entities like names, organizations, and locations.

## Project Setup

To set up the project, follow these steps:

### 1. Clone the Repository

Clone the project repository using Git:

```bash
git clone https://github.com/AnurupMondal/BiLSTM_NER_using_CRF.git
cd BiLSTM_NER_using_CRF
```

### 2. Set Up a Python Virtual Environment

1. Create a virtual environment:
    ```bash
    python -m .venv .
    ```
2. Activate the virtual environment:
    - On Windows:
      ```bash
      .venv/Scripts/activate.bat
      ```
    - On MacOS/Linux:
      ```bash
      source .venv/bin/activate
      ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Run the Streamlit GUI

The model has already been trained for 10 epochs, and the trained weights are saved in `model.pth`. To load the model and run predictions:

```bash
streamlit run gui.py
```

### 4. Retrain the Model

To retrain the model on the CoNLL-2003 dataset:

```bash
python model.py
```

## Dataset

The dataset used for training is in the `conll2003` folder.

## Additional Information

- **Python Version**: 3.10
- **Dependencies**: All dependencies are listed in `requirements.txt`.
