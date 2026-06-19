package alphazero

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
//  AlphaZeroStrategy — 通过持久 Python 子进程调用 AlphaZero 模型
// ============================================================
//
// 架构：Go 主进程 ↔ Python 子进程（stdin/stdout JSON Lines 管道）
//
// 为什么用持久子进程而非每次重启：
//   模型加载（PyTorch + CUDA 初始化）耗时 5-15s，持久化避免每步重复加载。
//   MCTS 搜索树在 Python 侧每步重建，无需跨进程传输。
//
// 线程安全：所有 stdin/stdout 操作通过 mu 互斥锁保护，
//   因为 Go HTTP server 可能并发调用 GetMove。

type AlphaZeroStrategy struct {
	cmd    *exec.Cmd       // Python 子进程句柄
	stdin  io.WriteCloser  // 写入 JSON 请求
	stdout *bufio.Reader   // 读取 JSON 响应（带缓冲）
	mu     sync.Mutex      // 保护 stdin/stdout 的并发访问
}

// JSON Lines 协议消息类型

type inferRequest struct {
	Board  [][]int `json:"board"`  // 15×15 棋盘: 0=空, 1=黑, 2=白
	Player int     `json:"player"` // 当前回合: 1=黑, 2=白
}

type inferResponse struct {
	Row   int    `json:"row"`            // 落子行 (0-14)
	Col   int    `json:"col"`            // 落子列 (0-14)
	Error string `json:"error,omitempty"` // 错误信息（非空表示失败）
}

type readyResponse struct {
	Status string `json:"status"`         // "ready" 表示初始化完成
	Error  string `json:"error,omitempty"` // 初始化失败时的错误信息
}

// NewAlphaZeroStrategy 创建策略实例并启动 Python 推理进程。
//
// 初始化流程:
//  1. 定位 infer.py 脚本（相对于可执行文件路径）
//  2. 启动 python infer.py <model_path> 子进程
//  3. 通过 goroutine + channel 异步等待 "ready" 信号
//  4. 超时 60s 未就绪则清理并返回错误
//
// 错误处理: 每个步骤失败都会关闭已分配资源，防止泄漏。
func NewAlphaZeroStrategy(modelPath string) (*AlphaZeroStrategy, error) {
	// 从可执行文件位置推导项目根目录（server.exe 位于项目根目录）
	execPath, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("获取可执行文件路径失败: %w", err)
	}
	projectRoot := filepath.Dir(execPath)
	pyScript := filepath.Join(projectRoot, "strategies", "alphazero", "infer.py")

	absModelPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, fmt.Errorf("解析模型路径失败: %w", err)
	}

	// 启动 Python 子进程，stderr 重定向到 Go 的 stderr 以便查看错误日志
	cmd := exec.Command("python", pyScript, absModelPath)
	cmd.Stderr = os.Stderr

	// 创建管道：Go 通过 stdin 发送请求，通过 stdout 读取响应
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("创建 stdin 管道失败: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close() // 清理已创建的管道
		return nil, fmt.Errorf("创建 stdout 管道失败: %w", err)
	}

	if err := cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		return nil, fmt.Errorf("启动 Python 进程失败: %w", err)
	}

	s := &AlphaZeroStrategy{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewReader(stdout), // 带缓冲读取，避免逐字节 syscall
	}

	// 异步等待 Python 就绪信号（避免阻塞构造函数的调用者）
	readyCh := make(chan error, 1)
	go func() {
		// 读取第一行 JSON — 必须是 {"status": "ready"}
		line, err := s.stdout.ReadString('\n')
		if err != nil {
			readyCh <- fmt.Errorf("读取就绪信号失败: %w", err)
			return
		}
		var rr readyResponse
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

	// 等待就绪或超时（模型加载 + CUDA 初始化可能较慢）
	select {
	case err := <-readyCh:
		if err != nil {
			s.Close() // 失败时清理子进程
			return nil, err
		}
	case <-time.After(60 * time.Second):
		s.Close()
		return nil, fmt.Errorf("等待 Python 推理服务超时")
	}

	log.Printf("[AlphaZero] 推理服务就绪，模型: %s", absModelPath)
	return s, nil
}

func (s *AlphaZeroStrategy) Name() string {
	return "AlphaZero"
}

// GetMove 获取 AlphaZero 推荐的落子。
//
// 通信协议（JSON Lines):
//   写入: {"board": [[...],...], "player": N}\n
//   读取: {"row": N, "col": M}\n  或  {"error": "..."}\n
//
// 错误处理策略: 任何通信或推理错误都返回棋盘中心 (7,7) 作为回退，
//   记录错误日志供调试。这种降级策略确保对局不会因单步失败而中断。
func (s *AlphaZeroStrategy) GetMove(board [game_engine.BoardSize][game_engine.BoardSize]int, player int) game_engine.Move {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 将固定大小数组转为 2D 切片（JSON 序列化需要）
	board2D := make([][]int, game_engine.BoardSize)
	for i := 0; i < game_engine.BoardSize; i++ {
		board2D[i] = board[i][:]
	}

	req := inferRequest{Board: board2D, Player: player}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		log.Printf("[AlphaZero] 请求序列化失败: %v", err)
		return game_engine.Move{Row: 7, Col: 7} // 中心回退
	}

	// 发送请求（JSON + 换行符）
	if _, err := fmt.Fprintf(s.stdin, "%s\n", reqJSON); err != nil {
		log.Printf("[AlphaZero] 写入请求失败: %v", err)
		return game_engine.Move{Row: 7, Col: 7}
	}

	// 读取响应（阻塞直到收到完整一行）
	line, err := s.stdout.ReadString('\n')
	if err != nil {
		log.Printf("[AlphaZero] 读取响应失败: %v", err)
		return game_engine.Move{Row: 7, Col: 7}
	}

	var resp inferResponse
	if err := json.Unmarshal([]byte(line), &resp); err != nil {
		log.Printf("[AlphaZero] 解析响应失败: %v (原始数据: %s)", err, line)
		return game_engine.Move{Row: 7, Col: 7}
	}

	if resp.Error != "" {
		log.Printf("[AlphaZero] 推理错误: %s", resp.Error)
		return game_engine.Move{Row: 7, Col: 7}
	}

	return game_engine.Move{Row: resp.Row, Col: resp.Col}
}

// Close 优雅关闭 Python 推理服务。
//
// 步骤:
//   1. 发送 "quit" 指令让 Python 端正常退出
//   2. 关闭 stdin 管道
//   3. 强制 kill 子进程（以防 Python 端未响应 quit）
//   4. Wait 回收子进程资源（防止僵尸进程）
func (s *AlphaZeroStrategy) Close() {
	if s.stdin != nil {
		fmt.Fprintf(s.stdin, "quit\n") // 告知 Python 端退出事件循环
		s.stdin.Close()
	}
	if s.cmd != nil && s.cmd.Process != nil {
		s.cmd.Process.Kill()  // 确保子进程终止
		s.cmd.Wait()          // 回收进程资源
	}
	log.Println("[AlphaZero] 推理服务已关闭")
}
