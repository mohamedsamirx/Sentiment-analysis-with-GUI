from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QWidget
)

from PyQt5.QtCore import Qt

from GUI.SentimentWindow import SentimentWindow
from GUI.TextBlobWindow import TextBlobWindow
from GUI.VaderWindow import VaderWindow
from GUI.LogisticRegressionWindow import LogisticRegressionWindow


class MainWindow(QMainWindow):
    def __init__(self, model, tokenizer, textblob_pipeline, vader_analyzer, vectorizer, log_r):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.textblob_pipeline = textblob_pipeline
        self.vader_analyzer = vader_analyzer
        self.vectorizer = vectorizer
        self.log_r = log_r
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Sentiment Analysis App")
        self.setGeometry(100, 100, 800, 500)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Header
        header_label = QLabel("Sentiment Analysis Tool", self)
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; text-align: center;")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Input section
        self.text_box = QLineEdit(self)
        self.text_box.setPlaceholderText("Enter text to analyze sentiment...")
        self.text_box.setStyleSheet("font-size: 16px; padding: 8px;")
        layout.addWidget(self.text_box)

        # Buttons section
        button_layout = QHBoxLayout()

        buttons = [
            ("BERT Analysis", self.open_sentiment_window),
            ("TextBlob Analysis", self.open_textblob_window),
            ("VADER Analysis", self.open_vader_window),
            ("Logistic Regression", self.open_log_reg_window)
        ]

        for text, handler in buttons:
            button = QPushButton(text, self)
            button.setStyleSheet("font-size: 14px; padding: 10px;")
            button.clicked.connect(handler)
            button_layout.addWidget(button)

        layout.addLayout(button_layout)



    def open_sentiment_window(self):
        # Get the text from the text box
        text_input = self.text_box.text()
        if text_input.strip():  # Ensure text is not empty
            self.sentiment_window = SentimentWindow(text_input, self.model, self.tokenizer)
            self.sentiment_window.show()
            
        else:
            error_window = QLabel("Please enter text for analysis.")
            error_window.setWindowTitle("Error")
            error_window.show()
            
    def open_textblob_window(self):
        text_input = self.text_box.text()
        if text_input.strip():  # Ensure text is not empty
            self.textblob_window = TextBlobWindow(text_input, self.textblob_pipeline)
            self.textblob_window.show()
            
        else:
            error_window = QLabel("Please enter text for analysis.")
            error_window.setWindowTitle("Error")
            error_window.show()
    
    def open_vader_window(self):
        text_input = self.text_box.text()
        if text_input.strip():  # Ensure text is not empty
            self.vader_window = VaderWindow(text_input, self.vader_analyzer)
            self.vader_window.show()
            
        else:
            error_window = QLabel("Please enter text for analysis.")
            error_window.setWindowTitle("Error")
            error_window.show()
            
    def open_log_reg_window(self):
        text_input = self.text_box.text()
        if text_input.strip():  # Ensure text is not empty
            self.log_r_ = LogisticRegressionWindow(text_input, self.vectorizer, self.log_r)
            self.log_r_.show()
            
        else:
            error_window = QLabel("Please enter text for analysis.")
            error_window.setWindowTitle("Error")
            error_window.show()

