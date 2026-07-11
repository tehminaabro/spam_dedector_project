
# Python ka official image istemal karein
FROM python:3.10-slim

# Working directory set karein
WORKDIR /code

# Requirements file copy aur install karein
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Baqi sara code copy karein
COPY . .

# FastAPI ko port 7860 par run karein (Hugging Face isi port ko read karta hai)
CMD ["uvicorn", "main:py", "--host", "0.0.0.0", "--port", "7860"]
