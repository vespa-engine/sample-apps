FROM vespaengine/vespa

RUN yum -y update && \
    yum install -y \
        git \
        python3 \
        python3-pip \
        jq \
        wget \
        && \
    yum clean all

WORKDIR /opt/vespa/share/ 
COPY . /opt/vespa/share/qa
RUN pip3 install --upgrade pip
RUN pip3 install -r /opt/vespa/share/qa/py-requirements.txt
RUN python3 -m nltk.downloader punkt 

WORKDIR /opt/vespa/share/ 
RUN git clone https://github.com/google/retrieval-qa-eval.git /opt/vespa/share/qa/bin/retrievalqaeval
