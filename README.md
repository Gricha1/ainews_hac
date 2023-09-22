# preinstalls
Python 3.11.5
# build docker image
docker build -t server_dublicates .
# run docker server(Ubuntu)
docker run --gpus all -v $(pwd)/uploaded_files:/app/uploaded_files -p 8084:8084 server_dublicates
# UI
go to: 
    localhost:8084/docs

load your dataset:
    <your_dataset>.csv

result dataset will be:
    ./uploaded_files/deduplicated_<your_dataset>.csv
    



# TO DELETE 
docker run -it -v $(pwd):/usr/home -p 8083:8083 python:3.11.5 bash
cd usr/home/
pip install -r requirements.txt
python -m nltk.downloader stopwords
uvicorn app.app:app --host 0.0.0.0 --port 8083
docker run -p 8083:8083 server_dublicates
docker run -p 5000:5000 my-server