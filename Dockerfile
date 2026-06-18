# ============================================================
#  GomokuMind Docker 镜像 — 多阶段构建
# ============================================================
#  包含: Go 后端 (heuristic/MCTS/AlphaBeta) + React 前端静态文件
#  不含: Python/PyTorch → AlphaZero 自动降级为 heuristic
# ============================================================

# ---- Stage 1: 构建前端 ----
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: 构建 Go 后端 ----
FROM golang:1.26-alpine AS backend-builder
WORKDIR /app
COPY go.mod ./
COPY game-engine/ game-engine/
COPY server/ server/
COPY strategies/ strategies/
RUN go build -ldflags="-s -w" -o /server ./server

# ---- Stage 3: 最终运行镜像 ----
FROM alpine:3.21

# 安全: 以非 root 用户运行
RUN adduser -D -H -h /app appuser

WORKDIR /app

# 复制 Go 二进制
COPY --from=backend-builder /server /app/server

# 复制前端静态文件
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# 复制 AlphaZero 模型（可选 — 若要启用需同时安装 Python+PyTorch）
# COPY strategies/alphazero/checkpoint_11.pth.tar /app/strategies/alphazero/checkpoint_11.pth.tar

# 复制 Python 推理脚本（可选）
# COPY strategies/alphazero/infer.py /app/strategies/alphazero/infer.py
# COPY strategies/alphazero/alphazero_train.py /app/strategies/alphazero/alphazero_train.py
# COPY strategies/requirements.txt /app/strategies/requirements.txt

USER appuser
EXPOSE 8080

CMD ["/app/server"]
