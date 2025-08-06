FROM python:3.11

WORKDIR /app

COPY ./docker/api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# CMD está ok, mas precisa apontar para main.app:app
CMD ["uvicorn", "main.app:app", "--host", "0.0.0.0", "--port", "8000"]
