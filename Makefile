.PHONY: help build up down logs restart clean index health test

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start the stack"
	@echo "  make down     - Stop the stack"
	@echo "  make logs     - Show logs"
	@echo "  make restart  - Restart the stack"
	@echo "  make clean    - Clean everything"
	@echo "  make index    - Index documents"
	@echo "  make health   - Check service health"
	@echo "  make test     - Test a query"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f langchain-server

restart:
	docker compose restart langchain-server

clean:
	docker compose down -v
	docker system prune -f

index:
	python index_documents.py

health:
	@echo "Checking RAG Server health..."
	@curl -s http://localhost:8000/health | jq
	@echo "\nChecking Qdrant..."
	@curl -s http://localhost:6333 | jq

test:
	@echo "Testing RAG query..."
	@curl -s -X POST http://localhost:8000/query \
	-H "Content-Type: application/json" \
	-d '{"question": "How to create an EC2 instance in Terraform?"}' | jq

status:
	docker compose ps

