FROM coolsa/pyqt-designer:x64

RUN apt-get install -y wget unzip

WORKDIR /opt/project

RUN pip install --no-cache-dir sentence-transformers
RUN pip install --no-cache-dir torchvision
