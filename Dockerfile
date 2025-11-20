# Use Node.js 18 LTS
FROM node:18-slim

# Set working directory
WORKDIR /app

# Copy package files
COPY backend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy backend server
COPY backend/server.js ./

# Copy static website files (index.html, styles.css)
COPY index.html ./
COPY styles.css ./

# Expose port (Cloud Run will set PORT env variable)
EXPOSE 3000

# Use non-root user for security
USER node

# Start the server
CMD ["node", "server.js"]

