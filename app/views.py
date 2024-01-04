# Create your views here.

from django.shortcuts import render
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import *
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response

# Create your views here.
from app.CrimeRegressionAnalysis import perform_regression_analysis, predict_crime_rate_by_year


def index(request):
    return render(request, 'index.html')


@api_view(['GET'])
@renderer_classes((JSONRenderer,))
@csrf_exempt
def train_classifier(request):
    results = {
        "success": True,
        "data": ""
    }

    if request.method == 'GET':
        status = perform_regression_analysis()
        results['success'] = status
        results['data'] = status

    return Response(data=results)


@api_view(['GET'])
@renderer_classes((JSONRenderer,))
@csrf_exempt
def predict_crime_rate(request):
    results = {
        "success": False,
        "data": "Error!!! Please pass right data!"
    }

    if request.method == 'GET' and 'year' in request.GET and 'category' in request.GET and 'algorithm' in request.GET:
        results['data'] = predict_crime_rate_by_year(request.GET['year'], request.GET['category'], request.GET['algorithm'])

    return Response(data=results)
