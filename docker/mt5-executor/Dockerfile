# Multi-stage build for MT5 Executor service
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Wine for MT5 terminal (if needed for Linux)
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y wine64 wine32

# Set working directory
WORKDIR /app

# Copy requirements
COPY mt5-executor/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wine64 \
    wine32 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r titanovax && useradd -r -g titanovax titanovax

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY mt5-executor/ ./mt5-executor/
COPY config/ ./config/
COPY utils/ ./utils/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config /app/mt5_terminal && \
    chown -R titanovax:titanovax /app

# Switch to non-root user
USER titanovax

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1
ENV WINEPREFIX=/app/.wine

# Run the application
CMD ["python", "-m", "mt5-executor.server", "--host", "0.0.0.0", "--port", "8002"]