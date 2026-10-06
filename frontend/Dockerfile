# Stage 1: Build the React application
FROM node:20-alpine as builder
WORKDIR /app

ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

# Stage 2: Serve with lightweight Nginx
FROM nginx:alpine

ENV PORT=80

# Copy the built assets to nginx's html directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Replace default config with our own simple config to handle React router fallback
RUN mkdir -p /etc/nginx/templates && \
    echo 'server { \
    listen ${PORT}; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/templates/default.conf.template

# Railway dynamically sets PORT, so we don't hardcode EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
