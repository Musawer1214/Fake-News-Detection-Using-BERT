
# Fake News Detection Using BERT

This project focuses on detecting fake news using a pre-trained BERT model. The aim is to classify news articles as either genuine or fake, utilizing Natural Language Processing (NLP) techniques with BERT.

## Project Overview

Fake news is a prevalent problem in today's digital age, with misinformation spreading rapidly across social media and news platforms. This project leverages the power of BERT (Bidirectional Encoder Representations from Transformers) to detect fake news with high accuracy.

## Features

- Utilizes BERT for feature extraction and classification
- Implements data preprocessing techniques for textual data
- Custom PyTorch dataset class for handling data
- Trains and evaluates a model for fake news detection

## Requirements

To run the notebook, you need the following dependencies:

- Python 3.7+
- PyTorch
- Transformers library (HuggingFace)
- Pandas
- NumPy
- Scikit-learn

You can install the dependencies using the following command:

```bash
pip install torch transformers pandas numpy scikit-learn
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Musawer1214/Fake-News-Detection-Using-BERT.git
   cd Fake-News-Detection-Using-BERT
   ```

2. Install the required dependencies.

3. Run the Jupyter notebook:

   ```bash
   jupyter notebook fake-news-detection-bert.ipynb
   ```

## Dataset

The dataset used for training and evaluation should contain labeled news articles with genuine and fake labels. You can use datasets like the [Fake News Challenge](http://www.fakenewschallenge.org/) or [Kaggle's Fake News dataset](https://www.kaggle.com/c/fake-news/data).

## Model Training

The notebook contains the following steps for training the BERT model:

1. **Data Preprocessing**: Cleaning and preparing text data for training.
2. **Dataset Preparation**: Creating a custom dataset class for handling training and validation data.
3. **Model Training**: Using BERT with a classification head to fine-tune on the fake news dataset.
4. **Evaluation**: Evaluating the model's performance using metrics like accuracy, precision, recall, and F1 score.

## Results

The model's performance is evaluated using various metrics to determine its ability to distinguish between real and fake news. Detailed results are available in the notebook.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers) for providing pre-trained models and utilities.
- [PyTorch](https://pytorch.org/) for the deep learning framework.

## Author

- [Musawer1214](https://github.com/Musawer1214)
