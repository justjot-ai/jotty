#!/bin/bash
set -e

echo "🚀 Starting Stock Market Analysis System"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Logging functions
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Utility functions
command_exists() {
    command -v "$1" &> /dev/null
}

file_exists() {
    [ -f "$1" ]
}

directory_exists() {
    [ -d "$1" ]
}

get_docker_compose_cmd() {
    if docker compose version &> /dev/null 2>&1; then
        echo "docker compose"
    else
        echo "docker-compose"
    fi
}

# Environment setup
setup_environment() {
    print_status "Setting up environment..."

    if ! file_exists ".env"; then
        print_warning ".env file not found, creating from template..."
        cp .env.example .env
        print_success "Created .env file. Please update with your configuration."
    fi

    if file_exists ".env"; then
        export $(grep -v '^#' .env | grep '=' | xargs)
    fi
}

# Directory creation
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data logs frontend/build
}

# Database initialization
initialize_database() {
    print_status "Initializing database..."

    if ! file_exists "data/stock_market.db"; then
        sqlite3 data/stock_market.db < schema.sql
        print_success "Database initialized"
    else
        print_warning "Database already exists, skipping initialization"
    fi
}

# Python dependencies installation
install_python_dependencies() {
    print_status "Installing Python dependencies..."

    if command_exists poetry; then
        poetry install
    elif file_exists requirements.txt; then
        pip install -r requirements.txt
    else
        print_error "No requirements.txt found and poetry not available"
        exit 1
    fi
}

# Frontend dependencies installation
install_frontend_dependencies() {
    print_status "Installing frontend dependencies..."

    if ! directory_exists frontend; then
        return 0
    fi

    cd frontend || return 1

    if file_exists package.json; then
        if command_exists yarn; then
            yarn install
        else
            npm install
        fi
        print_success "Frontend dependencies installed"
    else
        print_warning "No package.json found in frontend directory"
    fi

    cd .. || return 1
}

# Nginx configuration creation
create_nginx_config() {
    if file_exists nginx.conf; then
        return 0
    fi

    print_status "Creating nginx configuration..."
    cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    server {
        listen 80;

        location /api/ {
            proxy_pass http://backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        location /docs {
            proxy_pass http://backend/docs;
        }

        location /openapi.json {
            proxy_pass http://backend/openapi.json;
        }

        location / {
            proxy_pass http://frontend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
EOF
    print_success "Nginx configuration created"
}

# Docker configuration creation
create_docker_configs() {
    create_backend_dockerfile
    create_frontend_dockerfile
}

create_backend_dockerfile() {
    if file_exists Dockerfile.backend; then
        return 0
    fi

    print_status "Creating backend Dockerfile..."
    cat > Dockerfile.backend << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
}

create_frontend_dockerfile() {
    if file_exists frontend/Dockerfile; then
        return 0
    fi

    print_status "Creating frontend Dockerfile..."
    cat > frontend/Dockerfile << 'EOF'
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY yarn.lock* ./

# Install dependencies
RUN if [ -f yarn.lock ]; then yarn install; else npm install; fi

# Copy source code
COPY . .

EXPOSE 3000

CMD ["npm", "start"]
EOF
}

# Service management functions
start_docker_services() {
    print_status "Starting services with Docker Compose..."

    if ! command_exists docker; then
        print_error "Docker not found. Please install Docker and Docker Compose."
        exit 1
    fi

    local docker_compose_cmd
    docker_compose_cmd=$(get_docker_compose_cmd)

    $docker_compose_cmd down
    $docker_compose_cmd up --build -d

    print_success "Services started with Docker Compose"
    print_service_urls
}

start_local_services() {
    print_status "Starting services locally..."

    start_redis_if_needed
    start_celery_services
    start_backend_service
    start_frontend_service

    print_success "Services started locally"
}

start_production_services() {
    print_status "Starting services in production mode..."
    export API_ENV=production
    export API_RELOAD=false

    build_frontend_for_production

    local docker_compose_cmd
    docker_compose_cmd=$(get_docker_compose_cmd)
    $docker_compose_cmd -f docker-compose.yml -f docker-compose.prod.yml up -d
}

# Helper functions for service startup
start_redis_if_needed() {
    if ! pgrep redis-server > /dev/null; then
        print_status "Starting Redis..."
        redis-server --daemonize yes
    fi
}

start_celery_services() {
    print_status "Starting Celery worker..."
    celery -A app.celery worker --loglevel=info &

    print_status "Starting Celery beat..."
    celery -A app.celery beat --loglevel=info &
}

start_backend_service() {
    print_status "Starting FastAPI backend..."
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
}

start_frontend_service() {
    if ! directory_exists frontend; then
        return 0
    fi

    print_status "Starting React frontend..."
    cd frontend || return 1

    if command_exists yarn && file_exists yarn.lock; then
        yarn start &
    else
        npm start &
    fi

    cd .. || return 1
}

build_frontend_for_production() {
    if ! directory_exists frontend; then
        return 0
    fi

    cd frontend || return 1

    if command_exists yarn && file_exists yarn.lock; then
        yarn build
    else
        npm run build
    fi

    cd .. || return 1
}

print_service_urls() {
    print_status "Backend API: http://localhost:8000"
    print_status "Frontend: http://localhost:3000"
    print_status "API Docs: http://localhost:8000/docs"
}

# Main service control functions
start_services() {
    local mode=${1:-docker}

    case $mode in
        docker)
            start_docker_services
            ;;
        local)
            start_local_services
            ;;
        production)
            start_production_services
            ;;
        *)
            print_error "Unknown mode: $mode"
            exit 1
            ;;
    esac
}

stop_services() {
    print_status "Stopping services..."

    local docker_compose_cmd
    docker_compose_cmd=$(get_docker_compose_cmd)
    $docker_compose_cmd down

    pkill -f celery 2>/dev/null || true
    pkill -f uvicorn 2>/dev/null || true

    print_success "Services stopped"
}

show_logs() {
    local docker_compose_cmd
    docker_compose_cmd=$(get_docker_compose_cmd)
    $docker_compose_cmd logs -f
}

clean_environment() {
    print_status "Cleaning up..."

    local docker_compose_cmd
    docker_compose_cmd=$(get_docker_compose_cmd)
    $docker_compose_cmd down -v
    docker system prune -f

    rm -rf data/*.db logs/*
    print_success "Cleanup completed"
}

# Health check
health_check() {
    print_status "Performing health check..."
    sleep 5

    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Backend is healthy"
    else
        print_warning "Backend health check failed"
    fi

    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        print_success "Frontend is healthy"
    else
        print_warning "Frontend health check failed"
    fi
}

show_usage() {
    echo "Usage: $0 {docker|local|production|stop|logs|clean}"
    echo "  docker     - Start with Docker Compose (default)"
    echo "  local      - Start services locally"
    echo "  production - Start in production mode"
    echo "  stop       - Stop all services"
    echo "  logs       - Show service logs"
    echo "  clean      - Clean up all data and containers"
}

# Main initialization
main() {
    local mode=${1:-docker}

    # Setup phase
    setup_environment
    create_directories
    initialize_database
    install_python_dependencies
    install_frontend_dependencies
    create_nginx_config
    create_docker_configs

    # Execution phase
    case $mode in
        docker|local|production)
            start_services "$mode"
            health_check
            ;;
        stop)
            stop_services
            ;;
        logs)
            show_logs
            ;;
        clean)
            clean_environment
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
