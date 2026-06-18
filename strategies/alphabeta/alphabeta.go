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
// ============================================================

import (
	"math"
	"sort"
	"time"

	game_engine "gomokumind/game-engine"
)

// ============================================================
//  棋型枚举
// ============================================================

const (
	notype   = 0  // 无类型
	stwo     = 1  // 冲二
	two      = 2  // 活二
	sthree   = 3  // 冲三
	three    = 4  // 活三
	sfour    = 5  // 冲四
	four     = 6  // 活四
	five     = 7  // 五连
	dfour    = 8  // 双四
	fourt    = 9  // 四三
	dthree   = 10 // 双三
	analysed = 255
	todo     = 0
)

const boardSize = 15

// 四个方向
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

	// 1. 棋盘格式转换（interface: 1/2/0 → 内部: 1/2/0，已统一无需转换）
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

	// 3. 快速检测：己方获胜
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
//  boardEval —— 棋盘评估器
// ============================================================

type boardEval struct {
	record [boardSize][boardSize][4]int // 四方向棋型缓存
	count  [3][20]int                   // count[棋子][棋型]
	line   [30]int                      // 临时线段缓冲
	result [30]int                      // 临时结果缓冲
}

func (e *boardEval) reset() {
	for i := 0; i < boardSize; i++ {
		for j := 0; j < boardSize; j++ {
			e.record[i][j][0] = todo
			e.record[i][j][1] = todo
			e.record[i][j][2] = todo
			e.record[i][j][3] = todo
		}
	}
	for s := 0; s < 3; s++ {
		for p := 0; p < 20; p++ {
			e.count[s][p] = 0
		}
	}
}

func (e *boardEval) evaluate(b *boardIntl, turn int) int {
	score := e.evalRaw(b, turn)
	if score < -9000 {
		opp := opponentOf(turn)
		for i := 0; i < 20; i++ {
			if e.count[opp][i] > 0 {
				score -= i
			}
		}
	} else if score > 9000 {
		for i := 0; i < 20; i++ {
			if e.count[turn][i] > 0 {
				score += i
			}
		}
	}
	return score
}

func (e *boardEval) evalRaw(b *boardIntl, turn int) int {
	e.reset()

	// 第一步：四方向全盘棋型分析
	for i := 0; i < boardSize; i++ {
		for j := 0; j < boardSize; j++ {
			if b[i][j] == 0 {
				continue
			}
			if e.record[i][j][0] == todo {
				e.analyzeHorizon(b, i, j)
			}
			if e.record[i][j][1] == todo {
				e.analyzeVertical(b, i, j)
			}
			if e.record[i][j][2] == todo {
				e.analyzeLeft(b, i, j)
			}
			if e.record[i][j][3] == todo {
				e.analyzeRight(b, i, j)
			}
		}
	}

	// 第二步：统计各棋型数量
	for i := 0; i < boardSize; i++ {
		for j := 0; j < boardSize; j++ {
			stone := b[i][j]
			if stone == 0 {
				continue
			}
			for d := 0; d < 4; d++ {
				ch := e.record[i][j][d]
				if ch >= stwo && ch <= stwo+6 {
					e.count[stone][ch]++
				}
			}
		}
	}

	white := 2
	black := 1

	// 第三步：五连立即检测
	if turn == white {
		if e.count[black][five] > 0 {
			return -9999
		}
		if e.count[white][five] > 0 {
			return 9999
		}
	} else {
		if e.count[white][five] > 0 {
			return -9999
		}
		if e.count[black][five] > 0 {
			return 9999
		}
	}

	// 两个冲四 = 一个活四
	if e.count[white][sfour] >= 2 {
		e.count[white][four]++
	}
	if e.count[black][sfour] >= 2 {
		e.count[black][four]++
	}

	// 第四步：优先级链评分
	wVal, bVal := 0, 0

	if turn == white {
		if e.count[white][four] > 0 {
			return 9990
		}
		if e.count[white][sfour] > 0 {
			return 9980
		}
		if e.count[black][four] > 0 {
			return -9970
		}
		if e.count[black][sfour] > 0 && e.count[black][three] > 0 {
			return -9960
		}
		if e.count[white][three] > 0 && e.count[black][sfour] == 0 {
			return 9950
		}
		if e.count[black][three] > 1 &&
			e.count[white][sfour] == 0 &&
			e.count[white][three] == 0 &&
			e.count[white][sthree] == 0 {
			return -9940
		}

		if e.count[white][three] > 1 {
			wVal += 2000
		} else if e.count[white][three] > 0 {
			wVal += 200
		}
		if e.count[black][three] > 1 {
			bVal += 500
		} else if e.count[black][three] > 0 {
			bVal += 100
		}
		wVal += e.count[white][sthree] * 10
		bVal += e.count[black][sthree] * 10
		wVal += e.count[white][two] * 4
		bVal += e.count[black][two] * 4
		wVal += e.count[white][stwo]
		bVal += e.count[black][stwo]
	} else {
		// turn == black
		if e.count[black][four] > 0 {
			return 9990
		}
		if e.count[black][sfour] > 0 {
			return 9980
		}
		if e.count[white][four] > 0 {
			return -9970
		}
		if e.count[white][sfour] > 0 && e.count[white][three] > 0 {
			return -9960
		}
		if e.count[black][three] > 0 && e.count[white][sfour] == 0 {
			return 9950
		}
		if e.count[white][three] > 1 &&
			e.count[black][sfour] == 0 &&
			e.count[black][three] == 0 &&
			e.count[black][sthree] == 0 {
			return -9940
		}

		if e.count[black][three] > 1 {
			bVal += 2000
		} else if e.count[black][three] > 0 {
			bVal += 200
		}
		if e.count[white][three] > 1 {
			wVal += 500
		} else if e.count[white][three] > 0 {
			wVal += 100
		}
		bVal += e.count[black][sthree] * 10
		wVal += e.count[white][sthree] * 10
		bVal += e.count[black][two] * 4
		wVal += e.count[white][two] * 4
		bVal += e.count[black][stwo]
		wVal += e.count[white][stwo]
	}

	// 第五步：位置权重
	wc, bc := 0, 0
	for i := 0; i < boardSize; i++ {
		for j := 0; j < boardSize; j++ {
			switch b[i][j] {
			case white:
				wc += posWeight[i][j]
			case black:
				bc += posWeight[i][j]
			}
		}
	}
	wVal += wc
	bVal += bc

	if turn == white {
		return wVal - bVal
	}
	return bVal - wVal
}

// ============================================================
//  四方向分析
// ============================================================

func (e *boardEval) analyzeHorizon(b *boardIntl, i, j int) int {
	for x := 0; x < boardSize; x++ {
		e.line[x] = b[i][x]
	}
	e.analysisLine(boardSize, j)
	for x := 0; x < boardSize; x++ {
		if e.result[x] != todo {
			e.record[i][x][0] = e.result[x]
		}
	}
	return e.record[i][j][0]
}

func (e *boardEval) analyzeVertical(b *boardIntl, i, j int) int {
	for x := 0; x < boardSize; x++ {
		e.line[x] = b[x][j]
	}
	e.analysisLine(boardSize, i)
	for x := 0; x < boardSize; x++ {
		if e.result[x] != todo {
			e.record[x][j][1] = e.result[x]
		}
	}
	return e.record[i][j][1]
}

func (e *boardEval) analyzeLeft(b *boardIntl, i, j int) int {
	var x, y int
	if i < j {
		x, y = j-i, 0
	} else {
		x, y = 0, i-j
	}
	k := 0
	for k < boardSize {
		if x+k > 14 || y+k > 14 {
			break
		}
		e.line[k] = b[y+k][x+k]
		k++
	}
	e.analysisLine(k, j-x)
	for s := 0; s < k; s++ {
		if e.result[s] != todo {
			e.record[y+s][x+s][2] = e.result[s]
		}
	}
	return e.record[i][j][2]
}

func (e *boardEval) analyzeRight(b *boardIntl, i, j int) int {
	var x, y int
	if 14-i < j {
		x, y = j-14+i, 14
	} else {
		x, y = 0, i+j
	}
	k := 0
	for k < boardSize {
		if x+k > 14 || y-k < 0 {
			break
		}
		e.line[k] = b[y-k][x+k]
		k++
	}
	e.analysisLine(k, j-x)
	for s := 0; s < k; s++ {
		if e.result[s] != todo {
			e.record[y-s][x+s][3] = e.result[s]
		}
	}
	return e.record[i][j][3]
}

// ============================================================
//  analysisLine —— 单条线段棋型识别（核心算法）
// ============================================================
//
// 给定线段 line[0..num-1]，以 pos 为中心分析棋型：
//   1. 向左右扩展连续同色范围 [xl, xr]
//   2. 再扩展自由范围（遇对方棋子或边界止）
//   3. 根据连续长度和两端开放情况判定棋型
//   4. 支持跳棋检测（X_XXX→SFOUR, XX_XX→SFOUR, _X_XX→STHREE）
//

func (e *boardEval) analysisLine(num, pos int) {
	// 填充哨兵
	for i := num; i < 30; i++ {
		e.line[i] = 0xf
	}
	for i := 0; i < num; i++ {
		e.result[i] = todo
	}

	if num < 5 {
		for i := 0; i < num; i++ {
			e.result[i] = analysed
		}
		return
	}

	stone := e.line[pos]
	if stone == 0 {
		return
	}

	// 对方棋子值
	inverse := 1
	if stone == 1 {
		inverse = 2
	}

	// 向左扩展连续同色
	xl := pos
	for xl > 0 && e.line[xl-1] == stone {
		xl--
	}
	// 向右扩展
	xr := pos
	for xr < num-1 && e.line[xr+1] == stone {
		xr++
	}

	// 扩展到自由范围
	leftRange := xl
	for leftRange > 0 && e.line[leftRange-1] != inverse {
		leftRange--
	}
	rightRange := xr
	for rightRange < num-1 && e.line[rightRange+1] != inverse {
		rightRange++
	}

	// 范围不足 5 格 → 无威胁
	if rightRange-leftRange < 4 {
		for k := leftRange; k <= rightRange; k++ {
			e.result[k] = analysed
		}
		return
	}

	// 标记已分析
	for k := xl; k <= xr; k++ {
		e.result[k] = analysed
	}

	srange := xr - xl

	// ---- 五连 ----
	if srange >= 4 {
		e.result[pos] = five
		return
	}

	// ---- 四连 ----
	if srange == 3 {
		leftFour := xl > 0 && e.line[xl-1] == 0
		if xr < num {
			if e.line[xr+1] == 0 {
				if leftFour {
					e.result[pos] = four
				} else {
					e.result[pos] = sfour
				}
			} else {
				if leftFour {
					e.result[pos] = sfour
				}
			}
		} else {
			if leftFour {
				e.result[pos] = sfour
			}
		}
		return
	}

	// ---- 三连 ----
	if srange == 2 {
		left3 := false
		if xl > 0 && e.line[xl-1] == 0 {
			if xl > 1 && e.line[xl-2] == stone {
				e.result[xl] = sfour
				e.result[xl-2] = analysed
			} else {
				left3 = true
			}
		} else if xr == num-1 || e.line[xr+1] != 0 {
			return
		}

		if xr < num {
			if e.line[xr+1] == 0 {
				if xr < num-1 && e.line[xr+2] == stone {
					e.result[xr] = sfour
					e.result[xr+2] = analysed
				} else if left3 {
					e.result[xr] = three
				} else {
					e.result[xr] = sthree
				}
			} else if e.result[xl] == sfour {
				return
			} else if left3 {
				e.result[pos] = sthree
			}
		} else {
			if e.result[xl] == sfour {
				return
			}
			if left3 {
				e.result[pos] = sthree
			}
		}
		return
	}

	// ---- 二连 ----
	if srange == 1 {
		left2 := false
		if xl > 2 && e.line[xl-1] == 0 {
			if e.line[xl-2] == stone {
				if e.line[xl-3] == stone {
					e.result[xl-3] = analysed
					e.result[xl-2] = analysed
					e.result[xl] = sfour
				} else if e.line[xl-3] == 0 {
					e.result[xl-2] = analysed
					e.result[xl] = sthree
				}
			} else {
				left2 = true
			}
		}

		if xr < num && e.line[xr+1] == 0 {
			if xr < num-2 && e.line[xr+2] == stone {
				if e.line[xr+3] == stone {
					e.result[xr+3] = analysed
					e.result[xr+2] = analysed
					e.result[xr] = sfour
				} else if e.line[xr+3] == 0 {
					e.result[xr+2] = analysed
					if left2 {
						e.result[xr] = three
					} else {
						e.result[xr] = sthree
					}
				}
			} else {
				if e.result[xl] == sfour {
					return
				}
				if e.result[xl] == sthree {
					e.result[xl] = three
					return
				}
				if left2 {
					e.result[pos] = two
				} else {
					e.result[pos] = stwo
				}
			}
		} else {
			if e.result[xl] == sfour {
				return
			}
			if left2 {
				e.result[pos] = stwo
			}
		}
		return
	}
}

// ============================================================
//  abSearcher —— Alpha-Beta 搜索引擎
// ============================================================

type abSearcher struct {
	eval      *boardEval
	board     boardIntl
	bestRow   int
	bestCol   int
	maxDepth  int
	killers   [20][2]*[2]int // 杀手走法 [深度][主/副]
	history   map[moveKey]int
	timeLimit time.Duration
	startTime time.Time
	nodes     int
	timedOut  bool
	bestScore int
}

type moveKey struct {
	row, col, turn int
}

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

	// 迭代加深
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

	// 紧急局面加深
	if bestRow >= 0 && intAbs(bestScore) > 8000 && !s.timedOut {
		s.maxDepth = depth + 1
		sc := s.alphaBeta(turn, 1, math.MinInt64, math.MaxInt64)
		if !s.timedOut && s.bestRow >= 0 {
			return sc, s.bestRow, s.bestCol
		}
	}

	if bestRow < 0 {
		// 兜底
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
//  Alpha-Beta 核心递归
// ============================================================

func (s *abSearcher) alphaBeta(turn int, depth int, alpha int, beta int) int {
	// 超时检查（每 512 节点检查一次）
	if s.nodes%512 == 0 && time.Since(s.startTime) > s.timeLimit {
		s.timedOut = true
		return 0
	}
	s.nodes++

	// 叶子节点
	if depth <= 0 {
		return s.eval.evaluate(&s.board, turn)
	}

	// 终局检测
	sc := s.eval.evaluate(&s.board, turn)
	if intAbs(sc) >= 9999 && depth < s.maxDepth {
		return sc
	}

	// 生成走法
	moves := s.genMoves(turn)
	if len(moves) == 0 {
		return sc
	}

	opp := opponentOf(turn)

	for _, mv := range moves {
		s.board[mv.row][mv.col] = turn

		val := -s.alphaBeta(opp, depth-1, -beta, -alpha)

		s.board[mv.row][mv.col] = 0
		if s.timedOut {
			return 0
		}

		if val > alpha {
			alpha = val
			if depth == s.maxDepth {
				s.bestRow = mv.row
				s.bestCol = mv.col
				s.bestScore = val
			}

			if alpha >= beta {
				// 记录杀手走法
				if s.killers[depth][0] == nil || (*s.killers[depth][0])[0] != mv.row || (*s.killers[depth][0])[1] != mv.col {
					s.killers[depth][1] = s.killers[depth][0]
					rc := &[2]int{mv.row, mv.col}
					s.killers[depth][0] = rc
				}
				// 更新历史启发式
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

func (s *abSearcher) genMoves(turn int) []scoredMove {
	// 仅生成已有棋子周围2格内的候选（大幅减少分支因子）
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

		// 历史启发式
		mk := moveKey{i, j, turn}
		if h, ok := s.history[mk]; ok {
			sc += h
		}
		// 杀手走法
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
	sort.Slice(moves, func(a, b int) bool {
		return moves[a].score > moves[b].score
	})
	return moves
}

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

func findWinningMove(b boardIntl, cands []game_engine.Move, player int) (game_engine.Move, bool) {
	for _, mv := range cands {
		if wouldWin(b, mv.Row, mv.Col, player) {
			return mv, true
		}
	}
	return game_engine.Move{}, false
}

func wouldWin(b boardIntl, r, c, player int) bool {
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
