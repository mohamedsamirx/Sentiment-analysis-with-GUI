from transformers import BertTokenizer
from BERTClassifier import BERTClassifier
import pickle
import torch
import sys

from PyQt5.QtWidgets import QApplication

from GUI.MainWindow import MainWindow
from TextBlobPipe import TextBlobPipeline


if __name__ == "__main__":
    
    # Load model and tokenizer
    bert_model_path = "Models/bert_classifier.pth" 
   
    # Name of the pre-trained BERT model to be used.
    bert_model_name = 'bert-base-uncased'

    # The tokenizer converts raw text into tokens that can be processed by the BERT model.
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Determine the device to run the model on.
    # If a GPU is available, it will use 'cuda'; otherwise, it will default to the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 2

    model = BERTClassifier(bert_model_name, num_classes)
    model.load_state_dict(torch.load(bert_model_path, map_location = device))
    model.eval()





    textblob_pipeline_path = 'Models/textblob_pipeline.pkl'
    with open(textblob_pipeline_path, "rb") as file:
        textblob_pipeline = pickle.load(file)



    vader_path = 'Models/sia_model.pkl'

    # Load the object from the file
    with open(vader_path, 'rb') as file:
        loaded_sia = pickle.load(file)
        
    
    
    
    # Load the vectorizer
    vectorizer_path = 'Models/TfidfVectorizer.pkl'  
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)

    # Load the Logistic Regression model
    log_r_model_path = 'Models/logistic_regression_model_tfidf.pkl'  
    with open(log_r_model_path, 'rb') as file:
        log_r = pickle.load(file)

    # Run the app
    app = QApplication(sys.argv)
    main_window = MainWindow(model, tokenizer, textblob_pipeline, loaded_sia, vectorizer, log_r)
    main_window.show()
    sys.exit(app.exec_())
