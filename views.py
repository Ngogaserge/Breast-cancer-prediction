from django.shortcuts import render
from django.http import JsonResponse
from joblib import load
import numpy as np

# Create your views here.
def index(request):
    return render(request, 'myapp/index.html')

def predict(request):
    if request.method == 'POST':
        # Load your machine learning model
        model = load('myapp/model/grade-recommender.joblib')

        # Get the input values from the form and convert them to float
        radius_mean = float(request.POST.get('rainfall'))
        texture_mean = float(request.POST.get('fertilizer'))
        perimeter_mean = float(request.POST.get('temperature'))
      
        # Print the received input data for debugging
        print("Received input data:")
        print(f"radius mean: {radius_mean}")
        print(f"Texture mean: {texture_mean}")
        print(f"Perimeter mean: {perimeter_mean}")
        

        # Prepare input data array and scale it
        input_data = np.array([[radius_mean, texture_mean, perimeter_mean]])
        

        # Make prediction
        prediction = model.predict(input_data)

        # Print the prediction for debugging
        print(f"Breat cancer prediction {prediction[0]}")

        # Convert the prediction ndarray to a Python list
        prediction_list = prediction.tolist()

        # Return the prediction as a JSON response
        return JsonResponse({'prediction': prediction_list})

    return render(request, 'myapp/predict.html')
