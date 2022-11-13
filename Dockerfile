FROM tiangolo/uwsgi-nginx-flask:python3.8

# Create a work dir
WORKDIR /compareImage
EXPOSE 5000

# prerequisites
COPY requirements.txt ./requirements.txt

COPY ./compareImage ./compareImage

COPY ./compareImage/compareimagespy.py ./compareImage/compareimagespy.py

RUN python3 -m pip install --default-timeout=100 --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED 1

CMD ["python", "-u", "./compareImage/compareimagespy.py"]
