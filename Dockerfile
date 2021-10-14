FROM tensorflow/tensorflow:latest-jupyter

MAINTAINER Lucas David <lucasolivdavid@gmail.com>

ADD requirements.txt .
RUN pip -qq install -r requirements.txt
RUN jt -t grade3

ENV JOBLIB_TEMP_FOLDER /data/joblib/
