# A deliberately tiny worker, for telling two failures apart.
#
# When a worker never becomes ready there are two candidates: the platform
# (registry credentials, endpoint settings, placement) or our image (size,
# start-up, a missing dependency). This image removes every variable except the
# platform. If it runs, the problem is ours; if it does not, the problem is not
# something rebuilding the big image will fix.
FROM python:3.11-slim
RUN pip install --no-cache-dir runpod
COPY probe/handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
