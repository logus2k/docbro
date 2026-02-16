FROM caddy:2.10.2-alpine

# Copy static site files
COPY index.html /usr/share/caddy/
COPY styles/ /usr/share/caddy/styles/
COPY fonts/ /usr/share/caddy/fonts/
COPY script/ /usr/share/caddy/script/
COPY images/ /usr/share/caddy/images/
COPY libraries/ /usr/share/caddy/libraries/
COPY categories/ /usr/share/caddy/categories/
COPY documents.json /usr/share/caddy/

# Caddy configuration
COPY Caddyfile /etc/caddy/Caddyfile

EXPOSE 8765
