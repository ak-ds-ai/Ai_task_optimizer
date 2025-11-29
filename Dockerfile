FROM python:3.10

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Install dependencies
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY --chown=user . /app

# Hugging Face Spaces default port
ENV PORT=7860
EXPOSE 7860

# Run Flask using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
