# Base Python
FROM python:3.10

# --- FIX OPENCV & DEEPFACE LIBRARIES ---
# (libgl1 required for cv2, glib for face detectors)
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 libpng-dev libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# --- CREATE NON-ROOT USER (REQUIRED BY HF SPACES) ---
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# --- SET WORKDIR ---
WORKDIR /app

# --- INSTALL PYTHON REQUIREMENTS ---
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- COPY APPLICATION FILES ---
COPY --chown=user . /app

# --- SET PORT FOR HF SPACES ---
ENV PORT=7860
EXPOSE 7860

# --- START FLASK USING GUNICORN ---
# (app.py must contain "app = Flask(__name__)")
CMD ["gunicorn", "--timeout", "200", "--workers", "1", "--bind", "0.0.0.0:7860", "app:app"]
