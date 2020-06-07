from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Snippet
from .serializers import SnippetSerializer
from django.shortcuts import render, redirect
from django.http import HttpResponse
from PIL import Image
from rest_framework import serializers
from . Python_Script import *



@api_view(['GET', 'POST'])
def snippet_list(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        snippets = Snippet.objects.all()
        serializer = SnippetSerializer(snippets, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = SnippetSerializer(data=request.data)

        if serializer.is_valid():
            serializer.save()
            img_path = serializer.data['img']
            res=process_img(img_path)
            return Response(res, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
