from .base_analysis import BaseTextAnalyser, TextAnalyserException
from scipy.special import softmax
import numpy as np
from typing import Dict


class HateAnalysis(BaseTextAnalyser):
    """
    Text analyzer for hate speech analysis (classification).

    Args:
        BaseTextAnalyser (type): Base class for text analysis.

    Methods:
        hate() -> Dict[str, float]:
            Performs hate analysis on the provided text and returns a 
            dictionary with one of the following labes: hate or not-hate
            and the score assigned to that label.
        analysis() -> Dict[str, float]:
            Performs the analysis by calling the hate method and returns the hate tuple.
            If the text is empty, raises a TextAnalyserException.

    """

    def hate(self) -> Dict[str, float]:
        label = ['not-hate', 'hate']
        preprocessed_text = self._preprocess(self.text)
        encoded_input = self.tokenizer(preprocessed_text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        label = label[ranking[0]]
        hate_score = scores[ranking[0]]
        response = {'label': label, 'score': float(round(hate_score, 2))}
        return response

    def analysis(self) -> Dict[str, float]:
        if not self.text:
            try:
                raise TextAnalyserException("Text is empty.")
            except TextAnalyserException as e:
                print(f"Error: {str(e)}")
        else:
            return self.hate()
