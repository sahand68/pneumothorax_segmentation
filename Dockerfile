FROM python:3.7
COPY . /src

RUN pip install Werkzeug numpy Jinja2 pydicom gevent pillow h5py scikit-image opencv-contrib-python tensorflow matplotlib pandas sanic 


EXPOSE 5000
CMD [ "python" , "/src/app.py","serve"]
