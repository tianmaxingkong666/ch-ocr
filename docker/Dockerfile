FROM python:3.6

ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8

RUN pip install easydict opencv-contrib-python==3.4.2.16 Cython h5py lmdb mahotas pandas requests bs4 matplotlib lxml flask sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple/  \
  && pip install -U pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/ \ 
  && pip install web.py==0.40.dev0  \
  && pip install keras==2.1.5 tensorflow==1.8 \
  && pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl \
  && pip install torchvision
    
WORKDIR /opt/chineseocr
CMD ["/bin/bash","-c","python -u app.py"]
