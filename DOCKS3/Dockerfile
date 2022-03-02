
FROM vault.habana.ai/gaudi-docker/1.3.0/ubuntu20.04/habanalabs/pytorch-installer-1.10.1:1.3.0-499
# Image from habana labs

MAINTAINER   Mohamed A. Bencherif <bencherif.research@gmail.com>
     
ENV PYTHON /usr/bin/python3.8

WORKDIR /work3

ENV PYTHONPATH /work3/Model-References

RUN pip install --upgrade pip

#RUN pip install librosa --no-deps
#RUN pip install --no-deps torchaudio

RUN pip install soundfile

RUN pip install jupyter notebook

#CMD echo 'DeepSpeech Managed Docker'
CMD /bin/bash


