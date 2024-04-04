from .base_analysis import BaseTextAnalyser, TextAnalyserException
from scipy.special import softmax
import numpy as np
from typing import Dict


class SentimentAnalysis(BaseTextAnalyser):
    """
    Text analyzer for sentiment analysis.

    Args:
        BaseTextAnalyser (type): Base class for text analysis.

    Methods:
        sentiment() -> Dict[str, float]:
            Performs sentiment analysis on the provided text and returns a tuple
            containing the most relevant sentiment type ('negative', 'neutral', 'positive')
            and the score assigned to that sentiment.
        analysis() -> Dict[str, float]:
            Performs the analysis by calling the sentiment method and returns the sentiment tuple.
            If the text is empty, raises a TextAnalyserException.

    """

    def sentiment(self) -> Dict[str, float]:
        self.config.id2label = ['negative', 'neutral', 'positive']
        preprocessed_text = self._preprocess(self.text)
        encoded_input = self.tokenizer(preprocessed_text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        response = {
            'sentiment': self.config.id2label[np.argmax(scores)],
            'compound': round((scores[2] - scores[0]) * 1, 3),
            'neg': round(scores[0] * -1, 3),
            'neu': round(scores[1] * 1, 3),
            'pos': round(scores[2] * 1, 3),
        }
        #ranking = np.argsort(scores)
        #ranking = ranking[::-1]
        #label = self.config.id2label[ranking[0]]
        #sentiment_score = scores[ranking[0]] * 10 if label != 'negative' else scores[ranking[0]] * -10
        #response = {'sentiment':label, 'score':round(sentiment_score, 2)}
        return response

    def analysis(self) -> Dict[str, float]:
        if not self.text:
            try:
                raise TextAnalyserException("Text is empty.")
            except TextAnalyserException as e:
                print(f"Error: {str(e)}")
        else:
            return self.sentiment()
