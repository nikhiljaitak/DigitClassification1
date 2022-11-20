FROM tiangolo/uwsgi-nginx-flask:python3.8

# Create a work dir
WORKDIR ./
EXPOSE 5000

# prerequisites
COPY requirements.txt ./requirements.txt

COPY ./compareImage ./compareImage

COPY ./compareImage/compareimagespy.py ./compareImage/compareimagespy.py

COPY q1.py ./q1.py

COPY q2.py ./q2.py
COPY q3.py ./q3.py
COPY q4.py ./q4.py

COPY results ./results
COPY models ./models


RUN python3 -m pip install --default-timeout=100 --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED 1

CMD ["python", "-u", "./compareImage/compareimagespy.py"]
