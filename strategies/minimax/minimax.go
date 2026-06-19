package minimax

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	game_engine "gomokumind/game-engine"
)

// ============================================================
//  MinimaxStrategy — 通过持久 Python 子进程调用 Minimax 引擎
// ============================================================
//
// 架构：Go 主进程 ↔ Python 子进程（stdin/stdout JSON Lines 管道）
//
// 与 AlphaZero 不同, Minimax 是纯 CPU 计算（negamax + alpha-beta 剪枝）,
// 不需要 GPU/CUDA, 冷启动时间极短（<1s）.

type MinimaxStrategy struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	mu     sync.Mutex // 保护 stdin/stdout 的并发访问
}

// JSON Lines 协议消息类型 — 与 alphazero 协议一致
type minimaxRequest struct {
	Board  [][]int `json:"board"`  // 15×15: 0=空 1=黑 2=白
	Player int     `json:"player"` // 当前回合: 1=黑 2=白
}

type minimaxResponse struct {
	Row    int    `json:"row"`
	Col    int    `json:"col"`
	Status string `json:"status,omitempty"`
	Error  string `json:"error,omitempty"`
}

type minimaxReady struct {
	Status string `json:"status"`
	Error  string `json:"error,omitempty"`
}

// NewMinimaxStrategy 创建策略实例并启动 Python 推理进程。
//
// 初始化流程:
//  1. 定位 minimax.py 脚本
//  2. 启动 python minimax.py --serve 子进程
//  3. 异步等待 "ready" 信号
//  4. 超时 30s 未就绪则清理并返回错误
//
// depth 和 timeLimit 在 Python 端硬编码（depth=4, time_limit=2s）,
// 未来可扩展为通过命令行参数传递。
func NewMinimaxStrategy() (*MinimaxStrategy, error) {
	// 定位 minimax.py: 优先使用工作目录(支持 go run), 其次用可执行文件路径(支持生产构建)
	pyScript := filepath.Join("strategies", "minimax", "minimax.py")
	if _, err := os.Stat(pyScript); os.IsNotExist(err) {
		if execPath, err2 := os.Executable(); err2 == nil {
			pyScript = filepath.Join(filepath.Dir(execPath), "strategies", "minimax", "minimax.py")
		}
	}

	// 启动 Python 子进程, stderr 重定向到 Go 的 stderr 以便查看错误日志
	cmd := exec.Command("python", pyScript, "--serve")
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("创建 stdin 管道失败: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return nil, fmt.Errorf("创建 stdout 管道失败: %w", err)
	}

	if err := cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		return nil, fmt.Errorf("启动 Python 进程失败: %w", err)
	}

	s := &MinimaxStrategy{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewReader(stdout),
	}

	// 异步等待 Python 就绪信号
	readyCh := make(chan error, 1)
	go func() {
		line, err := s.stdout.ReadString('\n')
		if err != nil {
			readyCh <- fmt.Errorf("读取就绪信号失败: %w", err)
			return
		}
		var rr minimaxReady
		if err := json.Unmarshal([]byte(line), &rr); err != nil {
			readyCh <- fmt.Errorf("解析就绪信号失败: %w (原始数据: %s)", err, line)
			return
		}
		if rr.Error != "" {
			readyCh <- fmt.Errorf("Python 初始化错误: %s", rr.Error)
			return
		}
		if rr.Status != "ready" {
			readyCh <- fmt.Errorf("意外的就绪状态: %s", rr.Status)
			return
		}
		readyCh <- nil
	}()

	select {
	case err := <-readyCh:
		if err != nil {
			s.Close()
			return nil, err
		}
	case <-time.After(30 * time.Second):
		s.Close()
		return nil, fmt.Errorf("等待 Minimax 推理服务超时")
	}

	log.Println("[Minimax] 推理服务就绪")
	return s, nil
}

func (s *MinimaxStrategy) Name() string {
	return "Minimax"
}

// GetMove 获取 Minimax 推荐的落子。
//
// 通信协议 (JSON Lines):
//   写入: {"board": [[...],...], "player": N}\n
//   读取: {"row": N, "col": M, "status": "ok"}\n
//
// 错误处理: 任何通信或推理错误都返回棋盘中心 (7,7) 作为回退,
//   记录错误日志供调试。
func (s *MinimaxStrategy) GetMove(board [game_engine.BoardSize][game_engine.BoardSize]int, player int) game_engine.Move {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 将固定大小数组转为 2D 切片 (JSON 序列化需要)
	board2D := make([][]int, game_engine.BoardSize)
	for i := 0; i < game_engine.BoardSize; i++ {
		board2D[i] = board[i][:]
	}

	req := minimaxRequest{Board: board2D, Player: player}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		log.Printf("[Minimax] 请求序列化失败: %v", err)
		return game_engine.Move{Row: 7, Col: 7}
	}

	if _, err := fmt.Fprintf(s.stdin, "%s\n", reqJSON); err != nil {
		log.Printf("[Minimax] 写入请求失败: %v", err)
		return game_engine.Move{Row: 7, Col: 7}
	}

	line, err := s.stdout.ReadString('\n')
	if err != nil {
		log.Printf("[Minimax] 读取响应失败: %v", err)
		return game_engine.Move{Row: 7, Col: 7}
	}

	var resp minimaxResponse
	if err := json.Unmarshal([]byte(line), &resp); err != nil {
		log.Printf("[Minimax] 解析响应失败: %v (原始数据: %s)", err, line)
		return game_engine.Move{Row: 7, Col: 7}
	}

	if resp.Error != "" {
		log.Printf("[Minimax] 推理错误: %s", resp.Error)
		return game_engine.Move{Row: 7, Col: 7}
	}

	return game_engine.Move{Row: resp.Row, Col: resp.Col}
}

// Close 优雅关闭 Python 推理服务。
func (s *MinimaxStrategy) Close() {
	if s.stdin != nil {
		fmt.Fprintf(s.stdin, "quit\n")
		s.stdin.Close()
	}
	if s.cmd != nil && s.cmd.Process != nil {
		s.cmd.Process.Kill()
		s.cmd.Wait()
	}
	log.Println("[Minimax] 推理服务已关闭")
}
