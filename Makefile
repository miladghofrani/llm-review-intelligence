.PHONY: build up down run logs clean

# Build the Docker image
setup:
	docker compose build

# Start the container in the background and wait until healthy
start:
	docker compose up -d
	@echo "Waiting for service to become healthy..."
	@until [ "$$(docker inspect --format='{{.State.Health.Status}}' company_llm_container 2>/dev/null)" = "healthy" ]; do \
		sleep 5; \
	done
	@echo "Started"

# Stop and remove the container
stop:
	docker compose down

console:
	docker exec -it company_llm_container bash

# View container logs in real-time
logs:
	docker compose logs -f

# Clean up Docker images, containers, and volumes
clean:
	docker compose down --rmi all --volumes