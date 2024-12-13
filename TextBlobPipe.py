from textblob import TextBlob

class TextBlobPipeline:
    def analyze_text(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiment = 'positive' if polarity > 0 else ('negative' if polarity < 0 else 'neutral')
        return polarity, sentiment

    def process_dataframe(self, df):
        df['polarity'] = df['review'].apply(lambda review: TextBlob(review).sentiment.polarity)
        df['sentiment_pred'] = df['polarity'].apply(
            lambda score: 'positive' if score > 0 else ('negative' if score < 0 else 'neutral')
        )
        return df
