# SDG Paper Matcher — zero-cost deployment image
# Works on Hugging Face Spaces (free Docker Spaces), Render (free tier),
# or any Docker host. Pure Python standard library: no pip installs.
FROM python:3.11-slim

WORKDIR /app

COPY engine/ engine/
COPY web/ web/
COPY papers/ papers/
COPY LICENSE .

# Materialize the Scopus query ASTs into SQLite at image build time so
# first requests are fast (the app falls back to parsing the SDG*.txt
# files directly if this step is skipped).
RUN python3 engine/sdg2sqlite.py --quiet

# Copy the entrypoint that honors the platform-assigned $PORT
# (Render injects $PORT; Hugging Face Spaces expects 7860 by default).
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

HEALTHCHECK --interval=60s --timeout=5s --start-period=20s \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).status==200 else 1)" || exit 1

CMD ["/app/entrypoint.sh"]
