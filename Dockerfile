FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
LABEL maintainer="Nkifor"

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 7000

ENTRYPOINT ["python"]
CMD ["app.py"]

