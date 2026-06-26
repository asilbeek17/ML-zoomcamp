FROM python:3.13.14-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=1.0.bin" ,"./"]

EXPOSE 1717

ENTRYPOINT ["gunicorn",  "--bind=0.0.0.0:1717", "predict:app"]