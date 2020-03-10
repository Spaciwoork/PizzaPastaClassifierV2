FROM pytorch/pytorch:latest

ENV FLASK_ENV "development"
ENV FLASK_DEBUG True

WORKDIR /app

COPY './requirements.txt' .

RUN pip install --upgrade pip && \
    pip install pipenv

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]