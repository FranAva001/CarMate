FROM python:3.11-slim

# Install CarMate dependcies
WORKDIR /carmate
COPY requirements.txt /carmate/
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . /carmate/

# Expose the port
EXPOSE 8501
