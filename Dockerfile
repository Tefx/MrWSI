FROM tefx/mrkt:latest

WORKDIR /app
ADD . /app

RUN apt-get update && apt-get install -y libgraphviz-dev pkg-config && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean -y && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

EXPOSE 8333

ENV LD_LIBRARY_PATH=/app/MrWSI/core
