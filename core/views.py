# core/views.py
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import io
import os
from . import utils
from .auto_model import auto_train_and_predict

def home(request):
    return render(request, 'home.html')

def predict(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('dataset')
        if uploaded_file:
            try:
                # Read CSV file from the upload
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                return render(request, 'predict.html', {'error': f"Error reading file: {e}"})
            
            try:
                # Call the auto-adaptive model training/prediction function
                context = auto_train_and_predict(df)
            except Exception as e:
                return render(request, 'predict.html', {'error': f"Error during model training: {e}"})
            
            # Render results with the generated context (table preview, visualizations, etc.)
            return render(request, 'results.html', context)
    return render(request, 'predict.html')

def download_csv(request):
    file_path = 'media/predictions.csv'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
            return response
    return HttpResponse("No file found.")
