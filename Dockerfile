FROM tensorflow/tensorflow:2.9.3-gpu
ENV JOBLIB_TEMP_FOLDER /data/joblib/
MAINTAINER Lucas David <lucasolivdavid@gmail.com>

# RUN curl -o cuda-keyring_1.0-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/* && sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# RUN apt-get update && apt-get install libgl1 -y

ADD requirements.txt .
RUN pip -qq install -r requirements.txt

WORKDIR /workdir

CMD ["python" "run.py"]
