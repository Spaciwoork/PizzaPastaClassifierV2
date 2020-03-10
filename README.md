# PizzaPastaClassifierV2
Pizza and pasta classifier with pytorch

## Run the API using python

```
python api.py
```

## Run the API using docker
Build:
```
docker build -t flask-ai-pizza:latest .
```
Run:
```
docker run -d -p 5000:5000 flask-ai-pizza:latest
```

# Use Ngrok and configure the adress in chatfuel
```
ngrock http 5000
```

To create the model use the notebook (ask the team) or download the model from the [link](https://drive.google.com/file/d/1d21SqGoKD-xAuxzovTlzkWJHDPIv8JRZ/view?usp=sharing).
