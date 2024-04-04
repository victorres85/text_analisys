from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
from .forms import TextAnalyserForm
from .text_analyser import TextAnalyser

def text_analyser_view(request):
    if request.method == 'POST':
        form = TextAnalyserForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            task = form.cleaned_data['task']
            analyser = TextAnalyser(task, text)
            form = TextAnalyserForm()
            result = analyser.analysis()
            print(result)
            return render(request, 'result.html', {'result': result, 'form': form})
    else:
        form = TextAnalyserForm()
    return render(request, 'index.html', {'form': form})
