# DockerFile
FROM python:3.8
ADD . /app
WORKDIR /app 
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
COPY requirements.txt ./requirements.txt
COPY style_all.csv ./style_all.csv
COPY model_final.pth ./model_final.pth
COPY logo_new.png ./logo_new.png
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip uninstall -y pycocotools
RUN pip install pycocotools --no-binary pycocotools
COPY . /app
EXPOSE 8080
CMD streamlit run --server.port 8080 --server.enableCORS false app.py

