FROM golang:1.26.1-alpine AS builder

WORKDIR /build

# Copy dependency files first - this layer is cached unless go.mod/go.sum change
COPY go.mod go.sum ./
RUN go mod download

# Copy source files (CGO_ENABLED=0 ensures no C library dependency)
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o scheduler .

FROM alpine:3.23

# Install CA certificates
RUN apk add --no-cache ca-certificates

# Create a non-root user
RUN adduser -D -h /home/scheduler scheduler
USER scheduler

# Copy the compiled binary from the build stage
COPY --from=builder /build/scheduler /usr/local/bin/scheduler

# Copy the GPU specification database
COPY gpu_database.json /etc/webhook/gpu_database.json

# Expose listener port for HTTPS
EXPOSE 8443

ENTRYPOINT ["scheduler"]