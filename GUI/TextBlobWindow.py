from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QLabel, QWidget
)

from PyQt5.QtCore import Qt


class TextBlobWindow(QMainWindow):
    def __init__(self, text_input, pipeline):
        super().__init__()
        self.setWindowTitle("TextBlob Analysis")
        self.setGeometry(300, 300, 400, 200)

        # Analyze the input text
        polarity, sentiment = pipeline.analyze_text(text_input)

        # Set up layout and labels
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        input_label = QLabel(f"Input Text: {text_input}", self)
        input_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(input_label)

        polarity_label = QLabel(f"Polarity: {polarity:.2f}", self)
        polarity_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(polarity_label)

        sentiment_label = QLabel(f"Sentiment: {sentiment}", self)
        sentiment_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(sentiment_label)

