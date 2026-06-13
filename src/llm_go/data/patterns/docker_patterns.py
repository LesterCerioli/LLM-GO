
from __future__ import annotations

from string import Template

GO_VERSIONS = ["1.21", "1.22", "1.23", "1.24"]

_DOCKERFILE_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> dockerfile

# Dockerfile — multi-stage build for a Go Fiber application
# Pattern: golang builder + debian:bookworm-slim runtime
# Observed in: Medical-App-Core/Dockerfile

# ─── Stage 1: Build ────────────────────────────────────────────────────────
FROM golang:${go_version} AS builder

WORKDIR /app

# Copy dependency manifests first for better layer caching.
# Only re-downloads modules when go.mod or go.sum changes.
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build the binary from cmd/
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o app ./cmd

# ─── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM debian:bookworm-slim

WORKDIR /app

# CA certificates are required for outbound HTTPS (e.g., Stripe API).
RUN apt-get update && \\
    apt-get install -y --no-install-recommends ca-certificates curl && \\
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/app .

EXPOSE $port

# Document all required environment variables (values supplied at runtime).
ENV DB_HOST="" \\
    DB_PORT="5432" \\
    DB_USER="" \\
    DB_PASSWORD="" \\
    DB_NAME="" \\
    DB_TIMEZONE="UTC" \\
    JWT_SECRET="" \\
    RABBITMQ_BASE_URL="" \\
    SWAGGER_USER="" \\
    SWAGGER_PASSWORD="" \\
    PORT="$port"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
  CMD curl -f http://localhost:$port/health || exit 1

COPY ./docker/api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /app/app

ENTRYPOINT ["/entrypoint.sh"]
</go_file>
""")

_ENTRYPOINT_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> shell

#!/bin/sh
# docker/api/entrypoint.sh
# Pattern: validate required env vars before exec'ing the binary
# Observed in: Medical-App-Core/docker/api/entrypoint.sh
set -e

echo "Checking required environment variables..."

MISSING=0
for VAR in DB_HOST DB_PORT DB_USER DB_PASSWORD DB_NAME JWT_SECRET; do
  if [ -z "$$(eval echo \\$$$${VAR})" ]; then
    echo "ERROR: $$VAR is not set"
    MISSING=1
  else
    echo "OK: $$VAR is set"
  fi
done

if [ "$$MISSING" -eq 1 ]; then
  echo "One or more required environment variables are missing. Aborting."
  exit 1
fi

echo "All required variables present. Starting application..."
exec ./app
</go_file>
""")

_DOCKER_COMPOSE_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> docker-compose

# docker-compose.yml — local development environment
# Pattern: app + postgres + rabbitmq services with health checks
# Observed from: Medical-App-Core infrastructure patterns
version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "$port:$port"
    environment:
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_USER=postgres
      - DB_PASSWORD=postgres
      - DB_NAME=appdb
      - DB_TIMEZONE=UTC
      - JWT_SECRET=local-dev-secret-change-in-prod
      - RABBITMQ_BASE_URL=amqp://guest:guest@rabbitmq:5672/
      - SWAGGER_USER=admin
      - SWAGGER_PASSWORD=admin
      - PORT=$port
    depends_on:
      postgres:
        condition: service_healthy
      rabbitmq:
        condition: service_healthy
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: appdb
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  rabbitmq:
    image: rabbitmq:3.13-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pg_data:
</go_file>
""")

_JENKINSFILE_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> jenkins

// Jenkinsfile — CI/CD pipeline
// Pattern: multi-stage pipeline: test → build → docker → deploy
// Observed in: Medical-App-Core/Jenkinsfile
pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "$image_name"
        DOCKER_TAG   = "$${BUILD_NUMBER}"
        GO_VERSION   = "$go_version"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            steps {
                sh '''
                    go test ./... -v -coverprofile=coverage.out
                    go tool cover -html=coverage.out -o coverage.html
                '''
            }
            post {
                always {
                    publishHTML(target: [
                        reportName:  'Go Coverage',
                        reportDir:   '.',
                        reportFiles: 'coverage.html',
                    ])
                }
            }
        }

        stage('Lint') {
            steps {
                sh 'go vet ./...'
            }
        }

        stage('Build Binary') {
            steps {
                sh 'CGO_ENABLED=0 GOOS=linux go build -o app ./cmd'
            }
        }

        stage('Docker Build') {
            steps {
                sh 'docker build -t $${DOCKER_IMAGE}:$${DOCKER_TAG} .'
                sh 'docker tag $${DOCKER_IMAGE}:$${DOCKER_TAG} $${DOCKER_IMAGE}:latest'
            }
        }

        stage('Docker Push') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                        echo "$$DOCKER_PASS" | docker login -u "$$DOCKER_USER" --password-stdin
                        docker push $${DOCKER_IMAGE}:$${DOCKER_TAG}
                        docker push $${DOCKER_IMAGE}:latest
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'docker system prune -f || true'
        }
        failure {
            echo "Pipeline failed — check logs above"
        }
    }
}
</go_file>
""")


class DockerPatternGenerator:
    """Generate Docker + CI/CD training examples."""

    PORTS    = ["8080", "3040", "3000"]
    IMAGES   = ["myorg/my-api", "myorg/medical-api", "myorg/ecommerce-api"]

    def all_examples(self) -> list[str]:
        examples: list[str] = []
        for ver in GO_VERSIONS:
            for port in self.PORTS:
                examples.append(_DOCKERFILE_TEMPLATE.substitute(go_version=ver, port=port))
                examples.append(_DOCKER_COMPOSE_TEMPLATE.substitute(go_version=ver, port=port))
            examples.append(_ENTRYPOINT_TEMPLATE.substitute(go_version=ver))
            for img in self.IMAGES:
                examples.append(
                    _JENKINSFILE_TEMPLATE.substitute(go_version=ver, image_name=img)
                )
        return examples
