FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

LABEL maintainer="nyker <nykergoto@gmail.com>"
ARG jupyter_token=dolphin

# update
RUN apt-get update &&\
  apt-get install -y sudo git zsh openssh-server wget gcc libatlas-base-dev curl

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]

RUN mkdir /src

# userの追加
RUN groupadd -g 1000 developer

# user:penguin, password:highway
RUN useradd -g developer -G sudo -m -s /bin/zsh penguin
RUN echo "penguin:highway" | chpasswd

# 以下は penguin での操作
USER penguin
WORKDIR /home/penguin/

# docker build のディレクトリにあるファイル (.) を, カレントディレクトリ (`./`) にコピー
ADD . ./

# pyenv
RUN git clone git://github.com/yyuu/pyenv.git .pyenv
ENV HOME /home/penguin
ENV PYENV_ROOT ${HOME}/.pyenv
ENV PATH ${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}
ENV MINICONDA_VERSION miniconda2-4.3.30

RUN pyenv install ${MINICONDA_VERSION} && \
  pyenv rehash &&\
  pyenv global ${MINICONDA_VERSION}

RUN mkdir ~/.jupyter
RUN echo "c.NotebookApp.open_browser = False\n\ 
c.NotebookApp.ip = '*'\n\
c.NotebookApp.token = '${jupyter_token}'" | tee -a ${HOME}/.jupyter/jupyter_notebook_config.py

RUN conda update conda

WORKDIR /src
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN conda install pygpu jupyter notebook

EXPOSE 8888

CMD [ "jupyter", "notebook", "--ip", "0.0.0.0" ]
