from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification
import re


class TextAnalyserException(Exception):
    pass


class BaseTextAnalyser(object):
    """
    Base class for text analysis.

    Args:
        text (str): The text to be analyzed.

    Attributes:
        model_name (str): The model used for the analysis.
        tokenizer: The tokenizer for the model.
        config: The configuration for the model.
        model: The pre-trained model for sequence classification.

    Methods:
        remove_emojis(text: str) -> str:
            Removes emojis from the given text.
        preprocess(text: str) -> str:
            Preprocesses the text by removing social media handles, website links, and emojis.
        analysis():
            Raises a NotImplementedError. Subclasses should implement this method for specific analysis tasks.
    """

    def __init__(self, text: str, model_name:str = ''):
        self.text = text.strip()
        if model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def remove_emojis(self, text: str) -> str:
        """ Removes emojis from the given text. """
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010FFFF"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def _preprocess(self, text: str) -> str:
        """ Preprocesses the text by removing social media handles, website links, emails, and emojis. """
        new_text = []
        punctuation_marks = [",", ":", ";"]
        for t in text.split(" "):
            if t.startswith('@') and len(t) > 1:
                t = '@user' + t[-1] if t[-1] in punctuation_marks else '@user'
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        response = " ".join(new_text)
        response = response.replace('\n', ' ').replace('\t', ' ').replace('[link in bio]', '')
        response = re.sub(r'\S*@\S*\s?', '', response)
        return response.strip()

    def analysis(self):
        """ Raises a NotImplementedError. Subclasses should implement this method for specific analysis tasks. """
        try:
            raise TextAnalyserException("Analysis method not implemented.")
        except TextAnalyserException as e:
            print(f"Error: {str(e)}")

