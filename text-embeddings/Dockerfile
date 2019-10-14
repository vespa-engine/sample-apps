FROM vespaengine/vespa

RUN echo 'Creating docker setup of .. Vespa doc'

RUN yum -y update && \
    yum install -y \
        python3 \
        python3-pip \
        wget \
        && \
    yum clean all

WORKDIR /opt/vespa/share/ 
COPY . /opt/vespa/share/text-embeddings
RUN pip3 install -r /opt/vespa/share/text-embeddings/py-requirements.txt
RUN python3 -m nltk.downloader punkt 
