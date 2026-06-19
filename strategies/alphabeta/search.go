package alphabeta

import (
	"math"
	"sort"
	"time"
)

// ============================================================
//  abSearcher —— Alpha-Beta 搜索引擎
// ============================================================
//
// 核心特性：
//   - 迭代加深：从 depth=1 逐步加深到目标深度，浅层结果用于走法排序
//   - 杀手启发式：记录触发 beta 剪枝的走法，下次搜索优先尝试
//   - 历史启发式：记录每步棋"引发剪枝"的频率×深度²，高分走法优先搜索
//   - 走法排序：位置权重 + 历史启发式 + 杀手走法 → 大幅提升剪枝效率
//   - 2 秒超时保护：每 512 节点检查一次时间，超时立即返回当前最佳
//   - 紧急局面自适应加深：分数 |score| > 8000 时额外搜索一层
//
// ============================================================

type abSearcher struct {
	eval      *boardEval
	board     boardIntl
	bestRow   int
	bestCol   int
	maxDepth  int
	killers   [20][2]*[2]int // 杀手走法表 [深度][主杀手/副杀手]
	history   map[moveKey]int
	timeLimit time.Duration
	startTime time.Time
	nodes     int
	timedOut  bool
	bestScore int
}

// moveKey 用于历史启发式表 —— 记录 (行,列,执子方) 三元组
type moveKey struct {
	row, col, turn int
}

// search 执行迭代加深 Alpha-Beta 搜索。
// 返回 (分数, 行, 列)。分数正=己方优，负=对方优。
func (s *abSearcher) search(turn int, depth int) (score int, row int, col int) {
	s.maxDepth = depth
	s.timedOut = false
	s.startTime = time.Now()
	s.nodes = 0
	s.history = make(map[moveKey]int)

	// 初始化杀手走法表
	for i := range s.killers {
		s.killers[i] = [2]*[2]int{}
	}

	bestScore := math.MinInt64
	bestRow, bestCol := -1, -1

	// 迭代加深：从浅到深逐步搜索
	// 浅层结果虽不精确，但积累的杀手/历史信息大幅加速深层搜索
	for d := 1; d <= depth; d++ {
		if s.timedOut {
			break
		}
		s.maxDepth = d
		s.bestRow, s.bestCol = -1, -1
		sc := s.alphaBeta(turn, d, math.MinInt64, math.MaxInt64)
		if !s.timedOut && s.bestRow >= 0 {
			bestScore = sc
			bestRow, bestCol = s.bestRow, s.bestCol
		}
	}

	// 紧急局面加深：接近必胜/必败时再多搜一层确认
	if bestRow >= 0 && intAbs(bestScore) > 8000 && !s.timedOut {
		s.maxDepth = depth + 1
		sc := s.alphaBeta(turn, 1, math.MinInt64, math.MaxInt64)
		if !s.timedOut && s.bestRow >= 0 {
			return sc, s.bestRow, s.bestCol
		}
	}

	// 兜底：极端情况（所有走法都超时或被剪枝）
	if bestRow < 0 {
		for i := 0; i < boardSize; i++ {
			for j := 0; j < boardSize; j++ {
				if s.board[i][j] == 0 {
					return 0, i, j
				}
			}
		}
		return 0, 7, 7
	}

	return bestScore, bestRow, bestCol
}

// ============================================================
//  Alpha-Beta 核心递归（带 negamax 框架）
// ============================================================
//
// 标准 Alpha-Beta 剪枝算法：
//   - alpha: 当前方至少能获得的分数下界
//   - beta:  对方能容忍的分数上界
//   - alpha >= beta → 剪枝（对方不会让这个分支发生）
//
// 采用 negamax 视角：每层取反，alpha/beta 交换并取反。
//
// ============================================================

func (s *abSearcher) alphaBeta(turn int, depth int, alpha int, beta int) int {
	// 超时检查（每 512 节点检查一次，避免高频 syscall）
	if s.nodes%512 == 0 && time.Since(s.startTime) > s.timeLimit {
		s.timedOut = true
		return 0
	}
	s.nodes++

	// 叶子节点：直接静态评估
	if depth <= 0 {
		return s.eval.evaluate(&s.board, turn)
	}

	// 终局检测：任一方已成五 → 不再深入
	sc := s.eval.evaluate(&s.board, turn)
	if intAbs(sc) >= 9999 && depth < s.maxDepth {
		return sc
	}

	// 生成并排序走法
	moves := s.genMoves(turn)
	if len(moves) == 0 {
		return sc
	}

	opp := opponentOf(turn)

	for _, mv := range moves {
		// 落子
		s.board[mv.row][mv.col] = turn

		// 递归搜索（negamax：取反并交换 alpha/beta）
		val := -s.alphaBeta(opp, depth-1, -beta, -alpha)

		// 撤子
		s.board[mv.row][mv.col] = 0
		if s.timedOut {
			return 0
		}

		if val > alpha {
			alpha = val
			// 根节点更新最佳走法
			if depth == s.maxDepth {
				s.bestRow = mv.row
				s.bestCol = mv.col
				s.bestScore = val
			}

			// Beta 剪枝
			if alpha >= beta {
				// 记录杀手走法（触发剪枝的走法）
				if s.killers[depth][0] == nil ||
					(*s.killers[depth][0])[0] != mv.row ||
					(*s.killers[depth][0])[1] != mv.col {
					// 主杀手 → 副杀手，新杀手 → 主杀手
					s.killers[depth][1] = s.killers[depth][0]
					rc := &[2]int{mv.row, mv.col}
					s.killers[depth][0] = rc
				}
				// 更新历史启发式：深度² 加权（深层剪枝更有价值）
				mk := moveKey{mv.row, mv.col, turn}
				s.history[mk] = s.history[mk] + depth*depth
				break
			}
		}
	}

	return alpha
}

// ============================================================
//  走法生成 + 排序
// ============================================================

type scoredMove struct {
	row, col int
	score    int
}

// genMoves 生成候选走法并按启发式评分降序排列。
//
// 候选范围：已有棋子周围 2 格（大幅减少分支因子）。
// 排序依据：位置权重 + 历史启发式 + 杀手走法加成。
// 排序好的走法能大幅提升剪枝效率（好走法先搜 → 更多 beta 剪枝）。
func (s *abSearcher) genMoves(turn int) []scoredMove {
	// 仅生成已有棋子周围 2 格内的候选
	seen := make([]bool, boardSize*boardSize)
	hasPiece := false
	for i := 0; i < boardSize; i++ {
		for j := 0; j < boardSize; j++ {
			if s.board[i][j] != 0 {
				hasPiece = true
				for di := -2; di <= 2; di++ {
					for dj := -2; dj <= 2; dj++ {
						r, c := i+di, j+dj
						if r >= 0 && r < boardSize && c >= 0 && c < boardSize && s.board[r][c] == 0 {
							key := r*boardSize + c
							if !seen[key] {
								seen[key] = true
							}
						}
					}
				}
			}
		}
	}

	var moves []scoredMove
	if !hasPiece {
		// 空棋盘：只考虑中心
		return []scoredMove{{7, 7, posWeight[7][7]}}
	}
	for k := range seen {
		if !seen[k] {
			continue
		}
		i, j := k/boardSize, k%boardSize
		sc := posWeight[i][j]

		// 历史启发式加分
		mk := moveKey{i, j, turn}
		if h, ok := s.history[mk]; ok {
			sc += h
		}
		// 杀手走法加分（主杀手 +500，副杀手 +300）
		for d := range s.maxDepth + 1 {
			if s.killers[d][0] != nil {
				kr := *s.killers[d][0]
				if kr[0] == i && kr[1] == j {
					sc += 500
					break
				}
			}
			if s.killers[d][1] != nil {
				kr := *s.killers[d][1]
				if kr[0] == i && kr[1] == j {
					sc += 300
					break
				}
			}
		}
		moves = append(moves, scoredMove{i, j, sc})
	}
	// 按评分降序排列：高分先搜 → 更好的 alpha-beta 剪枝
	sort.Slice(moves, func(a, b int) bool {
		return moves[a].score > moves[b].score
	})
	return moves
}
