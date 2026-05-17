# Stage 1: Build
FROM python:3.12-alpine AS builder [cite: 20, 40]

WORKDIR /app
RUN apk add --no-cache gcc musl-dev libffi-dev

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime (Hardened)
FROM python:3.12-alpine [cite: 40]
WORKDIR /app

# Copiar solo lo necesario desde el builder
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Sello Y&C: No correr como root para mayor seguridad
USER 1001 

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
