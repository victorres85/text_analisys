from .base_analysis import BaseTextAnalyser, TextAnalyserException
from scipy.special import softmax
from typing import List, Dict
import numpy as np


class PassionAnalysis(BaseTextAnalyser):
    """
    Text analyzer for passion analysis (classification).

    Args:
        BaseTextAnalyser (type): Base class for text analysis.

    Methods:
        passion() -> List[Dict[str, float]]:
            Performs passion analysis on the provided text and returns a list of dictionaries.
            Each dictionary contains the most relevant passion type and the score assigned to that passion.
        analysis() -> List[Dict[str, float]]:
            Performs the analysis by calling the passion method and returns the passion list.
            If the text is empty, raises a TextAnalyserException.

    """

    def passion(self) -> List[Dict[str, float]]:
        class_mapping = self.model.config.id2label
        passion_mapping = {
            'arts_&_culture': ['Art/Design'],
            'business_&_entrepreneurs': ['Business'],
            'celebrity_&_pop_culture': [],
            'diaries_&_daily_life': ['Lifestyle'],
            'family': ['Family/Parenting'],
            'fashion_&_style': ['Fashion'],
            'film_tv_&_video': ['Film/Television', 'Entertainment/TV'],
            'fitness_&_health': ['Health/Fitness'],
            'food_&_dining': ['Food/Drink'],
            'gaming': ['Gaming'],
            'learning_&_educational': ['Education'],
            'music': ['Music'],
            'news_&_social_concern': [],
            'other_hobbies': [],
            'relationships': [],
            'science_&_technology': ['Tech', 'Computing'],
            'sports': ['Sports'],
            'travel_&_adventure': ['Travel', 'Adventure'],
            'youth_&_student_life': []
        }
        preprocessed_text = self._preprocess(self.text)
        encoded_input = self.tokenizer(preprocessed_text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        passions = []
        for i in ranking:
            if scores[i] > 0.05:
                passions.append({
                    '_passion': class_mapping[i],
                    'score': round(float(scores[i]), 2),
                    'passions': passion_mapping.get(class_mapping[i], []) # VB passions list
                })
        return passions

    def analysis(self):
        if not self.text:
            try:
                raise TextAnalyserException("Text is empty.")
            except TextAnalyserException as e:
                print(f"Error: {str(e)}")
        else:
            return self.passion()