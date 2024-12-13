from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QLabel, QWidget
)

from PyQt5.QtCore import Qt

import nltk
nltk.download('vader_lexicon')

class VaderWindow(QMainWindow):
    def __init__(self, text_input, vader_analyzer):
        super().__init__()
        self.setWindowTitle("VADER Sentiment Analysis")
        self.setGeometry(300, 300, 400, 200)

        # Analyze the input text using VADER
        scores = vader_analyzer.polarity_scores(text_input)

        # Set up layout and labels
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        input_label = QLabel(f"Input Text: {text_input}", self)
        input_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(input_label)

        pos_label = QLabel(f"Positive: {scores['pos']:.2f}", self)
        pos_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(pos_label)

        neu_label = QLabel(f"Neutral: {scores['neu']:.2f}", self)
        neu_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(neu_label)

        neg_label = QLabel(f"Negative: {scores['neg']:.2f}", self)
        neg_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(neg_label)

        compound_label = QLabel(f"Compound: {scores['compound']:.2f}", self)
        compound_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(compound_label)

