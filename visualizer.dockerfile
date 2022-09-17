FROM --platform=linux/amd64 condaforge/mambaforge

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git vim wget ghostscript
RUN apt-get upgrade -y bash
RUN apt-get clean

EXPOSE 8065

ADD . /raptgen-visualizer
# RUN mkdir /raptgen-visualizer

RUN wget https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTC/NotoSansCJK-Bold.ttc 
RUN mkdir /usr/share/fonts/opentype/noto
RUN mv NotoSansCJK-Bold.ttc /usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc

# create env
COPY ./visualizer/env.yml environment.yml
RUN mamba env create -f environment.yml

# activate env
ENV CONDA_DEFAULT_ENV raptgen-visualizer

# setting
RUN echo "conda activate raptgen-visualizer" >> ~/.bashrc
ENV PATH /opt/conda/envs/raptgen-visualizer/bin:$PATH
ENV PYTHONPATH /raptgen-visualizer

CMD [ "python3", "/raptgen-visualizer/visualizer/app.py" ]