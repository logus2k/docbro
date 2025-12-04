# -------------------------
# Stage 1: Build environment
# -------------------------
FROM node:25.2-alpine3.22 AS builder

# Install serve globally
RUN npm install -g serve

# -------------------------
# Stage 2: Alpine runtime
# -------------------------
FROM node:25.2-alpine3.22

# Copy node_modules from builder
COPY --from=builder /usr/local/lib/node_modules /usr/local/lib/node_modules

# Copy your static site
WORKDIR /app
COPY . .

EXPOSE 8765

# Run serve's main JavaScript file directly with node
ENTRYPOINT ["node", "/usr/local/lib/node_modules/serve/build/main.js", "-s", ".", "-l", "8765"]
