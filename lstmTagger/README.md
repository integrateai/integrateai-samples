## LSTM Tagger Model

This example is based on [Pytorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

### Getting Started
For this example it is suggested to look at pytorch example, if you are not familiar with the basics of NLP

### Using integrate.ai SDK

* To start we need to implement one class for models and one for dataset
* Preprocssing - In this dataset, sentences need to be converted to ids so it is understandable by the model 
  * Ensure your data is preprocessed and create your tokenizer dictionary before using integrate_ai_sdk.
  * For this simple dataset we use the function `create_tokenizer_file` in `dataset.py`.
  * Our tokenizer maps words to the same ids in all silos, so silo users need to share it through the dataset path.

* In the `lstmtagger.py` file you can see our simple LSTM model.
  * Note that to maintain *differential privacy*, LSTM will be replaced with `Opacus.layer.DPLSTM` in model training.
* In the `dataset.py` you can find loading and preprocessing section
* `lstmtagger.json` and `taggerDataset.json` are used for testing and contain the default inputs for custom dataset and custom model.
  * The default configs also enable other users to know how to structure their data to use your custom model
