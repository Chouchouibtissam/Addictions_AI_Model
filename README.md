# Addictions_AI_Model
# Description of model:
In this project, I developed an AI model that filters positive and negative text and removes the negative one.
# Working with a Clean Dataset:
In this project, I worked with a clean dataset that was obtained from GitHub. The dataset contained information on various , including their names, descriptions, prices, and ratings.

**Data Preprocessing**:
Before I started working with the dataset, I first had to preprocess it to ensure that it was clean and ready for analysis. The preprocessing involved several steps, including:

**Removing duplicates**: I used Python's pandas library to identify and remove any duplicate rows in the dataset.

**Handling missing values**: I also used pandas to identify and handle any missing values in the dataset. In some cases, I simply dropped the rows with missing values, while in other cases, I imputed the missing values using various techniques such as mean imputation.

**Data cleaning**: I cleaned the data by removing any unnecessary characters or symbols that could interfere with the analysis. This included removing whitespace, punctuation marks, and special characters.

**Machine learning models**: I also used a pre-training bert from Hugging Face Transformers library to pretrain a BERT model on a binary text classification task.So first we load and tokenize the training data using the BertTokenizer class from Hugging Face. We then convert the labels to a tensor and create a TensorDataset object containing the encoded texts and labels.
Next, we define the BERT model for sequence classification (BertForSequenceClassification) and an optimizer (AdamW). We then define a DataLoader for training, which samples batches randomly from the training dataset.
Finally, we train the model for 5 epochs, with each batch passed through the model using model(input_ids, attention_mask=attention_mask, labels=labels). The loss is calculated and the model is updated using the optimizer.

The pretrained model is then saved to the pretrained-bert-binary-classification directory using the save_pretrained method of the model object.
# Packages :
pandas, json, sklearn, torch, numpy, transformers, locale,time,random,datetime, pickle
# GitHub Link dataset:
https://github.com/dvaldez44/Reddit_Addiction
