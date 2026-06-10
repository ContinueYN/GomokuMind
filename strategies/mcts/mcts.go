package mcts

import (
	"math"
	"math/rand"

	game_engine "gomokumind/game-engine"
)

// ---- 节点 ----

type MCTSNode struct {
	move   game_engine.Move
	parent *MCTSNode
	wins   float64
	visits int
}

// 子节点信息（预计算启发式评分）
type mctsChild struct {
	move    game_engine.Move
	board   [game_engine.BoardSize][game_engine.BoardSize]game_engine.CellState
	curTurn game_engine.CellState
	score   float64
}

// ---- 策略 ----

type MCTSStrategy struct {
	simulations int
	c           float64 // UCT 探索常数
}

func NewMCTSStrategy(simulations int) *MCTSStrategy {
	return &MCTSStrategy{
		simulations: simulations,
		c:           math.Sqrt2,
	}
}

func (m *MCTSStrategy) Name() string             { return "蒙特卡洛树搜索(MCTS)" }
func (m *MCTSStrategy) Train(episodes int) error { return nil }

// ---- 候选走法（MoveHistory 遍历，避免全棋盘扫描） ----

func (m *MCTSStrategy) getCandidates(state *game_engine.GameEngine) []game_engine.Move {
	if len(state.MoveHistory) == 0 {
		return []game_engine.Move{{Row: 7, Col: 7}}
	}
	seen := make([]bool, game_engine.BoardSize*game_engine.BoardSize)
	var cands []game_engine.Move
	for _, piece := range state.MoveHistory {
		r0, c0 := piece.Row, piece.Col
		for di := -2; di <= 2; di++ {
			for dj := -2; dj <= 2; dj++ {
				r, c := r0+di, c0+dj
				if r >= 0 && r < game_engine.BoardSize &&
					c >= 0 && c < game_engine.BoardSize &&
					state.Board[r][c] == game_engine.Empty {
					key := r*game_engine.BoardSize + c
					if !seen[key] {
						seen[key] = true
						cands = append(cands, game_engine.Move{Row: r, Col: c})
					}
				}
			}
		}
	}
	if len(cands) == 0 {
		return []game_engine.Move{{Row: 7, Col: 7}}
	}
	return cands
}

// ---- 胜负检测 ----

func (m *MCTSStrategy) findWinningMove(state *game_engine.GameEngine, playerCell game_engine.CellState) (game_engine.Move, bool) {
	cands := m.getCandidates(state)
	for _, mv := range cands {
		if m.wouldWin(state.Board, mv.Row, mv.Col, playerCell) {
			return mv, true
		}
	}
	return game_engine.Move{}, false
}

func (m *MCTSStrategy) wouldWin(board [game_engine.BoardSize][game_engine.BoardSize]game_engine.CellState, r, c int, player game_engine.CellState) bool {
	for _, d := range evalDirs {
		cnt := 1
		for i := 1; i < 5; i++ {
			nr, nc := r+d[0]*i, c+d[1]*i
			if nr < 0 || nr >= game_engine.BoardSize ||
				nc < 0 || nc >= game_engine.BoardSize ||
				board[nr][nc] != player {
				break
			}
			cnt++
		}
		for i := 1; i < 5; i++ {
			nr, nc := r-d[0]*i, c-d[1]*i
			if nr < 0 || nr >= game_engine.BoardSize ||
				nc < 0 || nc >= game_engine.BoardSize ||
				board[nr][nc] != player {
				break
			}
			cnt++
		}
		if cnt >= 5 {
			return true
		}
	}
	return false
}

// ---- 主入口（快速评估 + UCT，不做整局 rollout） ----

func (m *MCTSStrategy) GetMove(board [game_engine.BoardSize][game_engine.BoardSize]int, player int) game_engine.Move {
	rootState := m.boardToGame(board)
	cands := m.getCandidates(rootState)
	if len(cands) == 0 {
		return game_engine.Move{Row: 7, Col: 7}
	}

	cur := rootState.CurrentTurn
	opp := game_engine.Black
	if cur == game_engine.Black {
		opp = game_engine.White
	}

	// 优先级1: 自己能赢
	if wm, ok := m.findWinningMove(rootState, cur); ok {
		return wm
	}

	// 优先级2: 对手能赢
	if bm, ok := m.findWinningMove(rootState, opp); ok {
		return bm
	}

	if len(cands) == 1 {
		return cands[0]
	}

	// 对每个候选预计算启发式评分（用于引导选择）
	children := make([]mctsChild, 0, len(cands))

	for _, mv := range cands {
		b := rootState.Board
		b[mv.Row][mv.Col] = cur
		ci := mctsChild{
			move:    mv,
			board:   b,
			curTurn: opp,
			score:   m.normalizedEval(b, mv.Row, mv.Col, cur, opp),
		}
		children = append(children, ci)
	}

	// 构建 MCTS 树（仅根 + 子节点，不展开更深层）
	root := &MCTSNode{move: game_engine.Move{Row: -1, Col: -1}}
	nodes := make([]*MCTSNode, len(children))
	for i, ci := range children {
		nodes[i] = &MCTSNode{
			move:   ci.move,
			parent: root,
		}
	}

	// MCTS 主循环：只选第一层 + 快速评估
	for sim := 0; sim < m.simulations; sim++ {
		node := m.selectChild(root, nodes)
		node.visits++
		// 用启发式评分 + 小随机噪声作为"模拟结果"
		idx := -1
		for i, n := range nodes {
			if n == node {
				idx = i
				break
			}
		}
		if idx >= 0 {
			result := children[idx].score + (rand.Float64()*0.2 - 0.1)
			if result > 1 {
				result = 1
			}
			if result < 0 {
				result = 0
			}
			node.wins += result
		}
	}

	return m.bestMove(root, nodes, children)
}

// UCT 选择第一个子节点
func (m *MCTSStrategy) selectChild(root *MCTSNode, nodes []*MCTSNode) *MCTSNode {
	logN := math.Log(float64(root.visits + 1))
	bestUCT := -math.MaxFloat64
	var best *MCTSNode

	for _, child := range nodes {
		var uct float64
		if child.visits == 0 {
			uct = math.MaxFloat64
		} else {
			uct = child.wins/float64(child.visits) +
				m.c*math.Sqrt(logN/float64(child.visits))
		}
		if uct > bestUCT {
			bestUCT = uct
			best = child
		}
	}
	if best == nil {
		return nodes[0]
	}
	root.visits++
	return best
}

func (m *MCTSStrategy) bestMove(root *MCTSNode, nodes []*MCTSNode, children []mctsChild) game_engine.Move {
	bestVisits := -1
	var best *MCTSNode

	for _, child := range nodes {
		if child.visits > bestVisits {
			bestVisits = child.visits
			best = child
		}
	}
	if best == nil {
		return children[0].move
	}
	return best.move
}

// ---- 启发式评分 ----

var evalDirs = [][2]int{{0, 1}, {1, 0}, {1, 1}, {1, -1}}

func (m *MCTSStrategy) normalizedEval(board [game_engine.BoardSize][game_engine.BoardSize]game_engine.CellState, row, col int, self, opp game_engine.CellState) float64 {
	raw := 0
	for _, d := range evalDirs {
		cnt, op := m.fastCount(board, row, col, self, d[0], d[1])
		raw += m.fastScore(cnt, op)
	}
	for _, d := range evalDirs {
		cnt, op := m.fastCount(board, row, col, opp, d[0], d[1])
		s := m.fastScore(cnt, op)
		if cnt >= 4 {
			s *= 2
		}
		raw += s
	}
	// sigmoid 归一化到 [0,1]
	return 1.0 / (1.0 + math.Exp(-float64(raw)/50000.0))
}

func (m *MCTSStrategy) fastCount(board [game_engine.BoardSize][game_engine.BoardSize]game_engine.CellState, row, col int, player game_engine.CellState, dr, dc int) (int, int) {
	cnt, openEnds := 1, 0
	for i := 1; i < 5; i++ {
		r, c := row+dr*i, col+dc*i
		if r < 0 || r >= game_engine.BoardSize || c < 0 || c >= game_engine.BoardSize {
			break
		}
		if board[r][c] == player {
			cnt++
		} else if board[r][c] == game_engine.Empty {
			openEnds++
			break
		} else {
			break
		}
	}
	for i := 1; i < 5; i++ {
		r, c := row-dr*i, col-dc*i
		if r < 0 || r >= game_engine.BoardSize || c < 0 || c >= game_engine.BoardSize {
			break
		}
		if board[r][c] == player {
			cnt++
		} else if board[r][c] == game_engine.Empty {
			openEnds++
			break
		} else {
			break
		}
	}
	return cnt, openEnds
}

func (m *MCTSStrategy) fastScore(cnt, openEnds int) int {
	if cnt >= 5 {
		return 1000000
	}
	switch {
	case cnt == 4 && openEnds == 2:
		return 100000
	case cnt == 4 && openEnds == 1:
		return 10000
	case cnt == 3 && openEnds == 2:
		return 5000
	case cnt == 3 && openEnds == 1:
		return 500
	case cnt == 2 && openEnds == 2:
		return 200
	case cnt == 2 && openEnds == 1:
		return 50
	}
	return 10
}

// ---- 工具 ----

func (m *MCTSStrategy) boardToGame(board [game_engine.BoardSize][game_engine.BoardSize]int) *game_engine.GameEngine {
	g := game_engine.NewGame()
	mc := 0
	for i := 0; i < game_engine.BoardSize; i++ {
		for j := 0; j < game_engine.BoardSize; j++ {
			if board[i][j] != 0 {
				p := game_engine.Black
				if board[i][j] == -1 {
					p = game_engine.White
				}
				g.Board[i][j] = p
				g.MoveHistory = append(g.MoveHistory, game_engine.Move{Row: i, Col: j})
				mc++
			}
		}
	}
	if mc%2 == 0 {
		g.CurrentTurn = game_engine.Black
	} else {
		g.CurrentTurn = game_engine.White
	}
	return g
}
