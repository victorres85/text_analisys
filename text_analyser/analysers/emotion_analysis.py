from .base_analysis import BaseTextAnalyser, TextAnalyserException
from scipy.special import softmax
import numpy as np
from typing import Dict


class EmotionAnalysis(BaseTextAnalyser):
    """
    Text analyzer for emotion analysis.

    Args:
        BaseTextAnalyser (type): Base class for text analysis.

    Methods:
        emotion() -> Dict[str, float]:
            Performs emotion analysis on the provided text and returns a dictionary
            containing the most relevant emotion type ['anger', 'joy', 'optimism', 'sadness']
            and the score assigned to that emotion.
        analysis() -> Dict[str, float]:
            Performs the analysis by calling the emotion method and returns the emotion dictionary.
            If the text is empty, raises a TextAnalyserException.

    """

    def emotion(self) -> Dict[str, float]:
        labels = ['anger', 'joy', 'optimism', 'sadness']
        preprocessed_text = self._preprocess(self.text)
        encoded_input = self.tokenizer(preprocessed_text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        label = labels[ranking[0]]
        emotion_score = scores[ranking[0]] * 10
        response = {'emotion':label, 'score': round(emotion_score, 2)}
        return response

    def analysis(self) -> Dict[str, float]:
        if not self.text:
            try:
                raise TextAnalyserException("Text is empty.")
            except TextAnalyserException as e:
                print(f"Error: {str(e)}")
        else:
            return self.emotion()
