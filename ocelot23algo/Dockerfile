FROM nvcr.io/nvidia/pytorch:22.11-py3 

ENV DEBIAN_FRONTEND=noninteractive

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

USER user

WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONPATH "${PYTHONPATH}:/opt/app/"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY ./ /opt/app/

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

RUN ["/bin/bash", "-c", "pip install openmim"]
RUN ["/bin/bash", "-c", "mim install mmengine"]
RUN ["/bin/bash", "-c", "mim install mmcv>=2.0.0rc4"]
RUN ["/bin/bash", "-c", "pip install mmsegmentation"]
RUN ["/bin/bash", "-c", "pip install -U debugpy"]

ENV GC_CELL_FPATH=/input/images/cell_patches/
ENV GC_TISSUE_FPATH=/input/images/tissue_patches/
ENV GC_METADATA_FPATH=/input/metadata.json
ENV GC_DETECTION_OUTPUT_PATH=/output/cell_classification.json

COPY --chown=user:user process.py /opt/app/

EXPOSE 5678

ENTRYPOINT [ "python", "-m", "process" ]