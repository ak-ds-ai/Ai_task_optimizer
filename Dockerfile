# 1. Python base image
FROM python:3.10

# 2. Non-root user (HF recommendation)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 3. Kaam karne ki directory
WORKDIR /app

# 4. Dependencies install
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Saara project copy
COPY --chown=user . /app

# 6. Hugging Face Spaces default port
ENV PORT=7860
EXPOSE 7860

# 7. Flask app ko gunicorn se run karo
#    "app:app" ka matlab:
#      - file: app.py
#      - variable: app = Flask(__name__)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:app"]