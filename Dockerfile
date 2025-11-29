FROM python:3.10

# --- system packages needed for OpenCV (fixes libGL.so.1 error) ---
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# Workdir
WORKDIR /app

# Install Python deps
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY --chown=user . /app

# HF default port
ENV PORT=7860
EXPOSE 7860

# Run Flask via gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]

