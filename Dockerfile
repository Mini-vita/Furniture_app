FROM python:3.8
WORKDIR /app/
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
COPY requirements.txt ./requirements.txt
COPY style_all.csv /style_all.csv
COPY 
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 8080
COPY . /home/daewonng12/apache-dockerfile
CMD streamlit run --server.port 8080 --server.enableCORS false app.py

