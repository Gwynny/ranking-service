FROM python:3.9.7-slim

ENV PYTHONUNBUFFERED 1

EXPOSE 8000
WORKDIR /src


RUN apt-get update && \
    apt-get install -y --no-install-recommends netcat wget && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    
RUN wget -P /usr/local/share/ca-certificates/cacert.org http://www.cacert.org/certs/root.crt http://www.cacert.org/certs/class3.crt
RUN update-ca-certificates

COPY poetry.lock pyproject.toml ./
RUN pip install poetry==1.3.1 && \
    poetry install

COPY . ./

CMD poetry run uvicorn --host=0.0.0.0 app.main:app