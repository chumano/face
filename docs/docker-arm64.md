# Ghi chú build Docker cho arm64

## 1. Cài đặt QEMU và Docker Buildx
- QEMU giúp giả lập kiến trúc arm64 trên máy x86_64.
- Docker Buildx hỗ trợ build đa nền tảng.

## 2. Tạo builder hỗ trợ đa nền tảng
```sh
docker buildx create --use
docker buildx inspect --bootstrap
```

## 3. Build image cho arm64
### Build 1 tag cho nhiều kiến trúc (multi-arch)
```sh
# Build và push cùng 1 tag cho cả amd64 và arm64:
docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.gpu -t your_image_name:latest --push .
```
- Docker sẽ tạo manifest multi-arch, khi pull sẽ tự động lấy đúng kiến trúc.
- Nếu chỉ build local (không push): thêm `--load` (chỉ hỗ trợ 1 kiến trúc).

### Build riêng cho arm64
```sh
docker buildx build --platform linux/arm64 -f Dockerfile.gpu -t your_image_name:arm64 --push .
```

## 4. Lưu ý về base image
- Đảm bảo base image (ví dụ: `nvidia/cuda`, `python:3.8-slim`) hỗ trợ arm64.
- Nếu base image không hỗ trợ arm64, sẽ gặp lỗi khi build.

## 5. Kiểm tra hỗ trợ arm64 của base image
```sh
docker manifest inspect nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04
```
- Nếu không hỗ trợ arm64, cần tìm base image tương đương có hỗ trợ arm64.

## 6. Tổng kết
- Sử dụng Docker Buildx và QEMU.
- Kiểm tra base image có hỗ trợ arm64.
- Build với `--platform linux/arm64`.

---

## 7. Sử dụng với docker-compose.yml

### Cách 1: Build image arm64 trước, rồi dùng trong docker-compose
1. Build và push multi-arch image:
	```sh
	docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.gpu -t your_image_name:latest --push .
	```
2. Trong docker-compose.yml, chỉ định image (tag chung):
	```yaml
	services:
	  your_service:
		 image: your_image_name:latest
		 # ... các cấu hình khác ...
	```

### Cách 2: Build trực tiếp bằng docker compose V2

Nếu dùng Docker Compose V2 (`docker compose`), có thể build trực tiếp multi-arch:

**Lưu ý:** Docker Compose V2 chưa hỗ trợ build multi-arch và push manifest trực tiếp qua file compose. Cách chuẩn nhất là build multi-arch trước bằng buildx rồi dùng image chung như trên.

Nếu chỉ build 1 kiến trúc (ví dụ arm64):
```yaml
services:
	your_service:
		build:
			context: .
			dockerfile: Dockerfile.gpu
			platform: linux/arm64
		# ... các cấu hình khác ...
```
Sau đó chạy:
```sh
docker compose build
docker compose up
```

### Lưu ý
- Docker Compose V1 (`docker-compose`) không hỗ trợ buildx đa nền tảng trực tiếp.
- Nếu chạy trên máy arm64, Docker sẽ tự động kéo đúng image arm64.
- Nếu chạy trên máy x86_64, cần cài QEMU để giả lập arm64.

---

## 8. Kiểm tra manifest multi-arch
Sau khi build multi-arch, kiểm tra manifest:
```sh
docker manifest inspect your_image_name:latest
```
Kết quả sẽ có cả amd64 và arm64.
