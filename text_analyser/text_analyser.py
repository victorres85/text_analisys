from text_analyser.analysers.emotion_analysis import EmotionAnalysis
from text_analyser.analysers.hate_analysis import HateAnalysis
from text_analyser.analysers.passion_analysis import PassionAnalysis
from text_analyser.analysers.profanity_analysis import ProfanityAnalysis
from text_analyser.analysers.sentiment_analysis import SentimentAnalysis
from typing import Dict, List, Union

class TextAnalyserException(Exception):
    pass

class TextAnalyser(object):
    """
    A controller used to analyse text.

    Args:
        task (str): The task or analysis type, which can be one of the following:
        sentiment, emotion, passion, hate, profanity.
        text (str): The text to be analysed.

    Attributes:
        text_analyser_registration (dict): A dictionary mapping analysis types to their respective analyser classes.

    Methods:
        analysis() -> Union[Dict[str, float], List[Dict[str, float]]]:
            Routes the analysis request to the appropriate text analyser based on the task and returns the analysis results.
            The return type can be a dictionay or a list of dictionary, see below the a sample result for each type of task:
            sentiment: {'sentiment': 'neutral', 'score': 7.27}
            emotion:   {'emotion': 'joy', 'score': 7.29}
            passion:   [{'passion': 'diaries_&_daily_life', 'score': 0.4583451},
                        {'passion': 'other_hobbies', 'score': 0.25058475},
                        {'passion': 'business_&_entrepreneurs', 'score': 0.10848563}]
            hate:      {'label': 'hate', 'score': 0.63}
            profanity: {'label': 'not-profane'}

    """

    def __init__(self, task: str, text: str):
        super(TextAnalyser, self).__init__()
        self.text = text
        self.task = task

        # register the analyser class for a given task
        self.analyzer_registration = {
            'sentiment': [SentimentAnalysis, "cardiffnlp/twitter-roberta-base-sentiment"],
            'emotion': [EmotionAnalysis, "cardiffnlp/twitter-roberta-base-emotion"],
            'passion': [PassionAnalysis, "cardiffnlp/tweet-topic-21-multi"],
            'hate': [HateAnalysis, "facebook/roberta-hate-speech-dynabench-r4-target"],
            'profanity': [ProfanityAnalysis, ''],
        }

    def analysis(self) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """ routing function for different TextAnalysers functions """
        # look for the registered method
        model = self.analyzer_registration.get(self.task)[1]
        analysisClass = self.analyzer_registration.get(self.task)[0]

        # instantiate the analyser
        if analysisClass:
            result = ''
            analyser = ''
            if self.task != 'profanity':
                result = analysisClass(model_name = model, text = self.text)
            else:
                analyser = analysisClass(self.text)
                result = analyser.analysis()
            return result
        else:
            try:
                raise TextAnalyserException(
                    """Task not found in analyzer_registration. options are:  sentiment, emotion, topic, hate, profanity """)
            except TextAnalyserException as e:
                print(f"Error: {str(e)}")
