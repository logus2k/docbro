docker compose down docbro
docker build --no-cache -t logus2k/docbro:latest .
docker compose up -d docbro
