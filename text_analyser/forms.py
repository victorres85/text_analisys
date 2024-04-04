from django import forms

def get_task_choices():
    return [('sentiment', 'Sentiment'), ('emotion', 'Emotion'), ('passion', 'Passion'), ('hate', 'Hate'), ('profanity', 'Profanity')]


class TextAnalyserForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control'}))
    task = forms.ChoiceField(choices=get_task_choices(), widget=forms.Select(attrs={'class': 'form-control'}))