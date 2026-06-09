package heuristic

import (
	"math"

	game_engine "gomokumind/game-engine"
)

// 棋型评分表
var patternScores = map[string]int{
	"活四": 100000, // 两边都开放的四连
	"冲四": 10000,  // 一边被堵的四连
	"活三": 5000,   // 两边都开放的三连
	"眠三": 500,    // 一边被堵的三连
	"活二": 200,    // 两边都开放的二连
	"眠二": 50,     // 一边被堵的二连
	"活一": 20,     // 两边都开放的一连
}

// 启发式策略
type HeuristicStrategy struct{}

func NewHeuristicStrategy() *HeuristicStrategy {
	return &HeuristicStrategy{}
}

func (h *HeuristicStrategy) Name() string {
	return "启发式规则"
}

func (h *HeuristicStrategy) GetMove(board [game_engine.BoardSize][game_engine.BoardSize]int, player int) game_engine.Move {
	bestScore := -math.MaxInt64
	bestMove := game_engine.Move{Row: 7, Col: 7} // 默认中心

	// 只考虑已有棋子周围的位置（优化搜索）
	candidates := h.getCandidates(board)

	for _, move := range candidates {
		// 评估自己落子后的得分
		attackScore := h.evaluatePosition(board, move.Row, move.Col, player)
		// 评估对手落子在此位置的得分（防守）
		defenseScore := h.evaluatePosition(board, move.Row, move.Col, -player)

		// 综合评分：进攻+防守
		score := attackScore + int(float64(defenseScore)*0.9) // 防守略低于进攻

		if score > bestScore {
			bestScore = score
			bestMove = move
		}
	}

	return bestMove
}

func (h *HeuristicStrategy) Train(episodes int) error {
	// 启发式策略无需训练
	return nil
}

// 获取候选位置（已有棋子周围2格内）
func (h *HeuristicStrategy) getCandidates(board [game_engine.BoardSize][game_engine.BoardSize]int) []game_engine.Move {
	var candidates []game_engine.Move
	hasPiece := false

	for i := 0; i < game_engine.BoardSize; i++ {
		for j := 0; j < game_engine.BoardSize; j++ {
			if board[i][j] != 0 {
				hasPiece = true
				// 检查周围2格
				for di := -2; di <= 2; di++ {
					for dj := -2; dj <= 2; dj++ {
						r, c := i+di, j+dj
						if r >= 0 && r < game_engine.BoardSize && c >= 0 && c < game_engine.BoardSize && board[r][c] == 0 {
							candidates = append(candidates, game_engine.Move{Row: r, Col: c})
						}
					}
				}
			}
		}
	}

	if !hasPiece {
		// 第一手棋下中心
		candidates = append(candidates, game_engine.Move{Row: 7, Col: 7})
	}

	return candidates
}

// 评估某个位置的得分
func (h *HeuristicStrategy) evaluatePosition(board [game_engine.BoardSize][game_engine.BoardSize]int, row, col, player int) int {
	totalScore := 0

	// 四个方向
	directions := [][2]int{
		{0, 1},  // 水平
		{1, 0},  // 垂直
		{1, 1},  // 对角线
		{1, -1}, // 反对角线
	}

	for _, dir := range directions {
		// 计算该方向上的连子数和开放端
		count, openEnds := h.countInDirection(board, row, col, player, dir[0], dir[1])

		// 根据连子数和开放端数评分
		score := h.getPatternScore(count, openEnds)
		totalScore += score
	}

	return totalScore
}

// 计算某个方向上的连子数和开放端
func (h *HeuristicStrategy) countInDirection(board [game_engine.BoardSize][game_engine.BoardSize]int, row, col, player, dr, dc int) (int, int) {
	count := 1
	openEnds := 0

	// 正方向
	for i := 1; i < 5; i++ {
		r, c := row+dr*i, col+dc*i
		if r < 0 || r >= game_engine.BoardSize || c < 0 || c >= game_engine.BoardSize {
			break
		}
		if board[r][c] == player {
			count++
		} else if board[r][c] == 0 {
			openEnds++
			break
		} else {
			break
		}
	}

	// 反方向
	for i := 1; i < 5; i++ {
		r, c := row-dr*i, col-dc*i
		if r < 0 || r >= game_engine.BoardSize || c < 0 || c >= game_engine.BoardSize {
			break
		}
		if board[r][c] == player {
			count++
		} else if board[r][c] == 0 {
			openEnds++
			break
		} else {
			break
		}
	}

	return count, openEnds
}

// 根据连子数和开放端数获取评分
func (h *HeuristicStrategy) getPatternScore(count, openEnds int) int {
	if count >= 5 {
		return 1000000 // 成五
	}

	switch {
	case count == 4 && openEnds == 2:
		return patternScores["活四"]
	case count == 4 && openEnds == 1:
		return patternScores["冲四"]
	case count == 3 && openEnds == 2:
		return patternScores["活三"]
	case count == 3 && openEnds == 1:
		return patternScores["眠三"]
	case count == 2 && openEnds == 2:
		return patternScores["活二"]
	case count == 2 && openEnds == 1:
		return patternScores["眠二"]
	case count == 1 && openEnds == 2:
		return patternScores["活一"]
	default:
		return 0
	}
}
