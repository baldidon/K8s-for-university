FROM python:alpine3.7
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
CMD ["python3","./modello_capi_abbigliamento.py"]

