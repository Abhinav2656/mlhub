# core/views.py
from django.shortcuts import render
import pandas as pd
from . import utils

def home(request):
    return render(request, 'home.html')

def predict(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('dataset')
        selected_model = request.POST.get('model')
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if selected_model:
                try:
                    model = utils.load_model(selected_model)
                    predictions = model.predict(df)
                    # Add predictions as a new column for display
                    df['prediction'] = predictions
                    # Optionally, you can store the results in the session or prepare a downloadable CSV later
                    return render(request, 'results.html', {
                        'table': df.head().to_html(classes='table table-striped')
                    })
                except Exception as e:
                    return render(request, 'predict.html', {
                        'error': f"Error during prediction: {e}"
                    })
            else:
                # If no model is selected, just show the data preview
                return render(request, 'preview.html', {
                    'table': df.head().to_html(classes='table table-striped')
                })
    # For GET requests, render the upload form with model selection
    return render(request, 'predict.html')
