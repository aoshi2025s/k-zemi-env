FROM python:3.10 

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y git
RUN apt-get install -y vim
RUN pip install --upgrade pip
RUN pip install poetry

# poetryのpath設定
ENV PATH /root/.local/bin:$PATH

# poetryが仮想環境を生成しないように設定
RUN poetry config virtualenvs.create false
