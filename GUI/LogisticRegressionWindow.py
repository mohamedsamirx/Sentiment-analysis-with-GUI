from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QLabel, QWidget
)

from PyQt5.QtCore import Qt


from log_reg_preprocess import preprocess_text

class LogisticRegressionWindow(QMainWindow):
    def __init__(self, text_input, vectorizer, model):
        super().__init__()
        self.setWindowTitle("Logistic Regression Sentiment Analysis")
        self.setGeometry(300, 300, 400, 200)
        
        preprocessed_text = preprocess_text(text_input)

        # Preprocess the text
        text_vector = vectorizer.transform([preprocessed_text])

        # Predict sentiment
        sentiment_scores = model.predict(text_vector)

        # Determine the sentiment
        sentiment = 'positive' if sentiment_scores == 1 else 'negative'

        # Set up layout and labels
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        input_label = QLabel(f"Input Text: {text_input}", self)
        input_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(input_label)

        sentiment_label = QLabel(f"Overall Sentiment: {sentiment}", self)
        sentiment_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(sentiment_label)


