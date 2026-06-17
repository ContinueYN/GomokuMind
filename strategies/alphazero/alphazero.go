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

type AlphaZeroStrategy struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	mu     sync.Mutex
}

type inferRequest struct {
	Board  [][]int `json:"board"`
	Player int     `json:"player"`
}

type inferResponse struct {
	Row   int    `json:"row"`
	Col   int    `json:"col"`
	Error string `json:"error,omitempty"`
}

type readyResponse struct {
	Status string `json:"status"`
	Error  string `json:"error,omitempty"`
}

// NewAlphaZeroStrategy 创建策略实例并启动 Python 推理进程。
func NewAlphaZeroStrategy(modelPath string) (*AlphaZeroStrategy, error) {
	execDir, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("get working dir: %w", err)
	}
	pyScript := filepath.Join(execDir, "strategies", "alphazero", "infer.py")

	absModelPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, fmt.Errorf("resolve model path: %w", err)
	}

	cmd := exec.Command("python", pyScript, absModelPath)
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("create stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return nil, fmt.Errorf("create stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		return nil, fmt.Errorf("start python process: %w", err)
	}

	s := &AlphaZeroStrategy{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewReader(stdout),
	}

	readyCh := make(chan error, 1)
	go func() {
		line, err := s.stdout.ReadString('\n')
		if err != nil {
			readyCh <- fmt.Errorf("read ready signal: %w", err)
			return
		}
		var rr readyResponse
		if err := json.Unmarshal([]byte(line), &rr); err != nil {
			readyCh <- fmt.Errorf("parse ready signal: %w (got: %s)", err, line)
			return
		}
		if rr.Error != "" {
			readyCh <- fmt.Errorf("python init error: %s", rr.Error)
			return
		}
		if rr.Status != "ready" {
			readyCh <- fmt.Errorf("unexpected ready status: %s", rr.Status)
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
	case <-time.After(60 * time.Second):
		s.Close()
		return nil, fmt.Errorf("timeout waiting for python inference server")
	}

	log.Printf("[AlphaZero] Inference server ready, model: %s", absModelPath)
	return s, nil
}

func (s *AlphaZeroStrategy) Name() string {
	return "AlphaZero"
}

func (s *AlphaZeroStrategy) GetMove(board [game_engine.BoardSize][game_engine.BoardSize]int, player int) game_engine.Move {
	s.mu.Lock()
	defer s.mu.Unlock()

	board2D := make([][]int, game_engine.BoardSize)
	for i := 0; i < game_engine.BoardSize; i++ {
		board2D[i] = board[i][:]
	}

	req := inferRequest{Board: board2D, Player: player}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		log.Printf("[AlphaZero] marshal request: %v", err)
		return game_engine.Move{Row: 7, Col: 7}
	}

	if _, err := fmt.Fprintf(s.stdin, "%s\n", reqJSON); err != nil {
		log.Printf("[AlphaZero] write request: %v", err)
		return game_engine.Move{Row: 7, Col: 7}
	}

	line, err := s.stdout.ReadString('\n')
	if err != nil {
		log.Printf("[AlphaZero] read response: %v", err)
		return game_engine.Move{Row: 7, Col: 7}
	}

	var resp inferResponse
	if err := json.Unmarshal([]byte(line), &resp); err != nil {
		log.Printf("[AlphaZero] parse response: %v (raw: %s)", err, line)
		return game_engine.Move{Row: 7, Col: 7}
	}

	if resp.Error != "" {
		log.Printf("[AlphaZero] inference error: %s", resp.Error)
		return game_engine.Move{Row: 7, Col: 7}
	}

	return game_engine.Move{Row: resp.Row, Col: resp.Col}
}

func (s *AlphaZeroStrategy) Train(episodes int) error {
	return nil
}

func (s *AlphaZeroStrategy) Close() {
	if s.stdin != nil {
		fmt.Fprintf(s.stdin, "quit\n")
		s.stdin.Close()
	}
	if s.cmd != nil && s.cmd.Process != nil {
		s.cmd.Process.Kill()
		s.cmd.Wait()
	}
	log.Println("[AlphaZero] Inference server closed.")
}
