FROM golang:1.24.8-alpine AS builder

ARG MINIO_VERSION=RELEASE.2025-10-15T17-29-55Z
ARG MINIO_COMMIT=9e49d5e

RUN apk add --no-cache git

WORKDIR /src

RUN git clone \
    --depth 1 \
    --branch "${MINIO_VERSION}" \
    https://github.com/minio/minio.git .

# Verifica che il tag punti al commit che abbiamo validato
RUN test "$(git rev-parse --short HEAD)" = "${MINIO_COMMIT}"

RUN CGO_ENABLED=0 \
    GOOS=linux \
    go build \
    -tags kqueue \
    -trimpath \
    -o /out/minio .

FROM alpine:3.22

RUN apk add --no-cache ca-certificates

COPY --from=builder /out/minio /usr/bin/minio

RUN chmod +x /usr/bin/minio

EXPOSE 9000 9001

VOLUME ["/data"]

ENTRYPOINT ["/usr/bin/minio"]

CMD ["server", "/data", "--console-address", ":9001"]