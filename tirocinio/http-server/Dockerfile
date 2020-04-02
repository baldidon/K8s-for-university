FROM python:alpine3.7
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python","./http-server.py"]

#docker run -P test-k8s
