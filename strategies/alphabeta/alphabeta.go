package alphabeta

// ============================================================
//  Alpha-Beta 增强搜索策略 —— Go 原生实现
// ============================================================
//
// 基于 skywind/gobang (2011-2024) 的经典棋型分析引擎，Go 语言移植。
// 核心特性：
//   - 11 种棋型精确识别（五连/活四/冲四/活三/冲三/活二/冲二/双四/四三/双三）
//   - 迭代加深 Alpha-Beta 搜索 + 杀手走法启发式
//   - 走法排序优化（历史启发式 + 位置权重）
//   - 跳棋（单 gap）检测（如 X_XXX, XX_XX）
//   - 必胜/必防速查
//   - depth=4 默认，紧急局面自适应加深
//   - 2 秒超时保护
//
// 与 Python 版相比：无子进程 IPC 开销，棋盘值拷贝，性能提升 5-10×。
//
// 文件结构：
//   alphabeta.go — 策略入口 + 公共类型 + 辅助函数
//   evaluate.go  — boardEval 评估器 + analysisLine 棋型识别
//   search.go    — abSearcher 搜索引擎 + 走法生成/排序
//
// ============================================================

import (
	"time"

	game_engine "gomokumind/game-engine"
)

// ============================================================
//  棋型枚举
// ============================================================

const (
	notype   = 0   // 无类型
	stwo     = 1   // 冲二
	two      = 2   // 活二
	sthree   = 3   // 冲三
	three    = 4   // 活三
	sfour    = 5   // 冲四
	four     = 6   // 活四
	five     = 7   // 五连
	dfour    = 8   // 双四
	fourt    = 9   // 四三
	dthree   = 10  // 双三
	analysed = 255 // 已分析标记
	todo     = 0   // 待分析标记
)

const boardSize = 15

// 四个方向：水平、垂直、主对角线、反对角线
var dirs = [4][2]int{{0, 1}, {1, 0}, {1, 1}, {1, -1}}

// 位置权重表 —— 中心高、边缘低（切比雪夫距离）
var posWeight [boardSize][boardSize]int

func init() {
	for i := 0; i < boardSize; i++ {
		for j := 0; j < boardSize; j++ {
			d := 7 - i
			if d < 0 {
				d = -d
			}
			d2 := 7 - j
			if d2 < 0 {
				d2 = -d2
			}
			if d2 > d {
				d = d2
			}
			posWeight[i][j] = (7 - d) * 3
		}
	}
}

// ============================================================
//  策略
// ============================================================

type AlphaBetaStrategy struct {
	depth     int
	timeLimit time.Duration
}

func NewAlphaBetaStrategy(depth int) *AlphaBetaStrategy {
	if depth <= 0 {
		depth = 4
	}
	return &AlphaBetaStrategy{
		depth:     depth,
		timeLimit: 2 * time.Second,
	}
}

func (a *AlphaBetaStrategy) Name() string {
	return "Alpha-Beta 增强搜索"
}

// ============================================================
//  GetMove — 主入口
// ============================================================

func (a *AlphaBetaStrategy) GetMove(
	boardInt [game_engine.BoardSize][game_engine.BoardSize]int,
	player int,
) game_engine.Move {

	// 1. 棋盘格式转换（interface: 1/2/0 → 内部: 1/2/0，已统一）
	var b boardIntl
	turn := 1
	for i := 0; i < boardSize; i++ {
		for j := 0; j < boardSize; j++ {
			switch boardInt[i][j] {
			case 1:
				b[i][j] = 1
			case 2:
				b[i][j] = 2
			}
		}
	}
	if player == 2 {
		turn = 2
	}

	// 2. 候选生成
	cands := getCandidates(b)
	if len(cands) == 0 {
		return game_engine.Move{Row: 7, Col: 7}
	}

	// 3. 快速检测：己方直接获胜
	if m, ok := findWinningMove(b, cands, turn); ok {
		return m
	}
	// 4. 堵对手获胜
	opp := 2
	if turn == 2 {
		opp = 1
	}
	if m, ok := findWinningMove(b, cands, opp); ok {
		return m
	}

	// 5. 空棋盘 → 天元
	if len(cands) == boardSize*boardSize {
		return game_engine.Move{Row: 7, Col: 7}
	}

	// 6. Alpha-Beta 搜索
	searcher := &abSearcher{
		eval:      &boardEval{},
		board:     b,
		timeLimit: a.timeLimit,
	}
	score, row, col := searcher.search(turn, a.depth)
	_ = score

	return game_engine.Move{Row: row, Col: col}
}

// ============================================================
//  内部棋盘类型（0=空, 1=黑, 2=白）
// ============================================================

type boardIntl [boardSize][boardSize]int

// ============================================================
//  辅助函数
// ============================================================

func opponentOf(turn int) int {
	if turn == 1 {
		return 2
	}
	return 1
}

func intAbs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// getCandidates 生成候选落子：已有棋子周围 2 格内的空位。
// 使用 seen 数组去重；空棋盘返回天元 (7,7)。
func getCandidates(b boardIntl) []game_engine.Move {
	seen := make([]bool, boardSize*boardSize)
	var cands []game_engine.Move
	hasPiece := false
	for i := 0; i < boardSize; i++ {
		for j := 0; j < boardSize; j++ {
			if b[i][j] != 0 {
				hasPiece = true
				for di := -2; di <= 2; di++ {
					for dj := -2; dj <= 2; dj++ {
						r, c := i+di, j+dj
						if r >= 0 && r < boardSize && c >= 0 && c < boardSize && b[r][c] == 0 {
							key := r*boardSize + c
							if !seen[key] {
								seen[key] = true
								cands = append(cands, game_engine.Move{Row: r, Col: c})
							}
						}
					}
				}
			}
		}
	}
	if !hasPiece {
		return []game_engine.Move{{Row: 7, Col: 7}}
	}
	return cands
}

// findWinningMove 在候选列表中搜索能直接成五的走法。
func findWinningMove(b boardIntl, cands []game_engine.Move, player int) (game_engine.Move, bool) {
	for _, mv := range cands {
		if wouldWin(b, mv.Row, mv.Col, player) {
			return mv, true
		}
	}
	return game_engine.Move{}, false
}

// wouldWin 判断在 (r,c) 落 player 后是否形成五连（不计跳跃，仅实连检测）。
func wouldWin(b boardIntl, r, c int, player int) bool {
	for _, d := range dirs {
		dr, dc := d[0], d[1]
		cnt := 1
		for i := 1; i < 5; i++ {
			nr, nc := r+dr*i, c+dc*i
			if nr < 0 || nr >= boardSize || nc < 0 || nc >= boardSize || b[nr][nc] != player {
				break
			}
			cnt++
		}
		for i := 1; i < 5; i++ {
			nr, nc := r-dr*i, c-dc*i
			if nr < 0 || nr >= boardSize || nc < 0 || nc >= boardSize || b[nr][nc] != player {
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
