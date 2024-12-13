import torch
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QLabel, QWidget
)

from PyQt5.QtCore import Qt


class SentimentWindow(QMainWindow):
    def __init__(self, text_input, model, tokenizer):
        super().__init__()
        self.setWindowTitle("Sentiment Analysis")
        self.setGeometry(300, 300, 400, 200)

        # Preprocess the text
        inputs = tokenizer(text_input, 
                           return_tensors = "pt",
                           truncation = True,
                           padding = True,
                           max_length = 128)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Predict sentiment
        with torch.no_grad():
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim = 1)
            sentiment = "Positive" if probabilities[0, 1] > probabilities[0, 0] else "Negative"
            confidence = probabilities[0, 1].item() if sentiment == "Positive" else probabilities[0, 0].item()

        # Set up layout and labels
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        input_label = QLabel(f"Input Text: {text_input}", self)
        input_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(input_label)

        sentiment_label = QLabel(f"Sentiment: {sentiment}", self)
        sentiment_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(sentiment_label)

        confidence_label = QLabel(f"Confidence Score: {confidence:.2f}", self)
        confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(confidence_label)


