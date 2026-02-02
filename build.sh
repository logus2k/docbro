docker compose down docbro
docker build --no-cache -t docbro:1.0 .
docker compose up -d docbro
