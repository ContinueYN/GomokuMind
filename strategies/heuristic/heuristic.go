package heuristic

import (
	"math"

	game_engine "gomokumind/game-engine"
)

// ============================================================
//  启发式策略 —— 算法概述
// ============================================================
//
// 本策略基于"棋型评估 + 动态攻防权重"的静态评估方法：
//
// 1. 候选生成：只考虑已有棋子周围 2 格内的空位（大幅剪枝）
// 2. 两轮评估：
//    a) 第一轮 —— 检测己方是否可直接成五（立即返回），同时记录每个候选位的
//       攻击分（己方棋型）和防守分（对手在此落子的棋型）
//    b) 第二轮 —— 综合评分 = 攻击分 + 防守分 × 动态防守权重 + 位置权重
//       动态防守权重根据对手威胁级别自适应调整（活四→10, 冲四→1.5, 活三→1.1）
// 3. 棋型检测：4 方向扫描，支持单跳棋（gap）模式（如 X_XXX → 识别为冲四）
// 4. 双重威胁：2+ 方向同时形成活三以上棋型时给予 ×1.5 加成
//
// 分值设计原则：各级棋型之间呈指数级差距（约 10×），确保策略不会因
// 小分值累积而错误选择。例如：无论多少个活三(5000)，总和也不会超过一个活四(100000)。
//
// ============================================================
//  棋型分值常量
// ============================================================

const (
	scoreFive       = 10000000 // 成五 / 必胜
	scoreLiveFour   = 100000   // 活四（双端开放）—— 必胜，对手无法同时堵两端
	scoreRushFour   = 10000    // 冲四（单端开放）—— 对手必须堵，但能防住
	scoreLiveThree  = 5000     // 活三（双端开放）—— 有潜力发展成活四
	scoreSleepThree = 500      // 眠三（单端开放）—— 威胁有限
	scoreLiveTwo    = 200      // 活二 —— 需要多步发展
	scoreSleepTwo   = 50       // 眠二 —— 威胁很小
	scoreLiveOne    = 20       // 活一 —— 基础连接
)

// ============================================================
//  位置权重矩阵 —— 距离棋盘中心越近，基础权重越高
// ============================================================
//
// 计算方式：取每个位置到中心点的切比雪夫距离 max(|dx|, |dy|)，
// 距离越小（越靠近中心），权重越高。
//
// 中心(7,7)权重=21，角落(0,0)权重=0，最大值单格贡献约 21 分。
// 这个权重作为综合评分中的细微调节项，在攻防分值相近时打破平局，
// 优先选择靠近中心的落子（中心位置通常有更多发展可能性）。
//
// ============================================================

var positionWeight [game_engine.BoardSize][game_engine.BoardSize]int

func init() {
	center := game_engine.BoardSize / 2 // 15/2 = 7
	for i := 0; i < game_engine.BoardSize; i++ {
		for j := 0; j < game_engine.BoardSize; j++ {
			di := center - i
			if di < 0 {
				di = -di
			}
			dj := center - j
			if dj < 0 {
				dj = -dj
			}
			// 切比雪夫距离：max(|dx|, |dy|)，使权重呈同心正方形分布
			d := di + dj
			if di < dj {
				d = dj
			}
			positionWeight[i][j] = (center - d) * 3
		}
	}
}

// ============================================================
//  四个扫描方向
// ============================================================

var dirs = [4][2]int{
	{0, 1},  // 水平
	{1, 0},  // 垂直
	{1, 1},  // 正对角线
	{1, -1}, // 反对角线
}

// ============================================================
//  策略
// ============================================================

type HeuristicStrategy struct{}

func NewHeuristicStrategy() *HeuristicStrategy {
	return &HeuristicStrategy{}
}

func (h *HeuristicStrategy) Name() string {
	return "启发式规则"
}

func (h *HeuristicStrategy) Train(episodes int) error {
	return nil
}

// ============================================================
//  GetMove — 主入口：两轮评估选出最优落子
// ============================================================
//
// 两轮评估设计：
//
// 第一轮（紧急检测）：
//   - 己方能直接成五 → 立即返回，不做任何计算
//   - 同时计算每个候选位的攻击分和防守分，存入 evals 数组
//   - 检测对手是否能在某位置成五 → 必须堵
//
// 第二轮（综合评分）：
//   总分 = 攻击分 + 防守分 × 动态防守权重 + 位置权重
//   附加 ×1.05 攻击加成（当攻击分 ≥ 活三×2 时）
//
// 动态防守权重设计：
//   对手活四 → 权重 10.0（必须堵，不惜代价）
//   对手冲四 → 权重 1.5
//   对手活三 → 权重 1.1
//   其他     → 权重 0.9（正常偏向进攻）
//   默认 0.9 < 1.0 意味着同等棋型下优先进攻而非防守
//
// ============================================================

func (h *HeuristicStrategy) GetMove(
	board [game_engine.BoardSize][game_engine.BoardSize]int,
	player int,
) game_engine.Move {

	candidates := h.getCandidates(board)
	if len(candidates) == 0 {
		return game_engine.Move{Row: 7, Col: 7}
	}

	opp := -player

	// ---- 第一轮扫描：检查是否有必胜 / 必防位置 ----
	type eval struct {
		move         game_engine.Move
		attackScore  int
		defenseScore int
	}

	evals := make([]eval, 0, len(candidates))

	for _, mv := range candidates {
		// 优先检测：自己能立即成五 → 直接返回，不做无谓计算
		if h.isWinningMove(board, mv.Row, mv.Col, player) {
			return mv
		}

		// 攻击分：为己方评估此落子形成的棋型
		attackScore := h.evaluateMove(board, mv.Row, mv.Col, player)
		// 防守分：为对手评估同样的落子（对手在此下棋会形成多大的威胁）
		defenseScore := h.evaluateMove(board, mv.Row, mv.Col, opp)

		evals = append(evals, eval{
			move:         mv,
			attackScore:  attackScore,
			defenseScore: defenseScore,
		})
	}

	// 对手有必胜点位（成五）→ 必须堵
	for _, e := range evals {
		if e.defenseScore >= scoreFive {
			return e.move
		}
	}

	// ---- 第二轮：综合评分选最优 ----
	bestScore := -math.MaxInt64
	bestMove := evals[0].move

	for _, e := range evals {
		// 动态防守权重：对手威胁越大，防守越重要
		defenseWeight := 0.9
		if e.defenseScore >= scoreLiveFour {
			defenseWeight = 10.0 // 对手活四必须赌，不惜牺牲进攻
		} else if e.defenseScore >= scoreRushFour {
			defenseWeight = 1.5
		} else if e.defenseScore >= scoreLiveThree {
			defenseWeight = 1.1
		}

		score := e.attackScore + int(float64(e.defenseScore)*defenseWeight)
		score += positionWeight[e.move.Row][e.move.Col]

		// 双重威胁加成：如果攻击分本身已经很高（说明形成了强力棋型），
		// 再乘 ×1.05 让真正有威胁的走法脱颖而出
		if e.attackScore >= scoreLiveThree*2 {
			score = score * 105 / 100
		}

		if score > bestScore {
			bestScore = score
			bestMove = e.move
		}
	}

	return bestMove
}

// ============================================================
//  候选位置生成 — 已有棋子周围 2 格
// ============================================================
//
// 剪枝策略：只考虑已有棋子周围曼哈顿距离 ≤ 2 的空位。
// 在 15×15=225 个格子的棋盘上，此策略将候选数从 225 大幅缩减，
// 且几乎不会漏掉有意义的走法（远离所有棋子的孤立空格几乎没有价值）。
// 使用 seen 数组去重，避免同一空位被多个棋子重复添加。
//
// ============================================================

func (h *HeuristicStrategy) getCandidates(
	board [game_engine.BoardSize][game_engine.BoardSize]int,
) []game_engine.Move {

	seen := make([]bool, game_engine.BoardSize*game_engine.BoardSize)
	var candidates []game_engine.Move
	hasPiece := false

	for i := 0; i < game_engine.BoardSize; i++ {
		for j := 0; j < game_engine.BoardSize; j++ {
			if board[i][j] != 0 {
				hasPiece = true
				for di := -2; di <= 2; di++ {
					for dj := -2; dj <= 2; dj++ {
						r, c := i+di, j+dj
						if r >= 0 && r < game_engine.BoardSize &&
							c >= 0 && c < game_engine.BoardSize &&
							board[r][c] == 0 {
							key := r*game_engine.BoardSize + c
							if !seen[key] {
								seen[key] = true
								candidates = append(candidates, game_engine.Move{Row: r, Col: c})
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

	return candidates
}

// ============================================================
//  evaluateMove — 模拟放置棋子后评估位置价值
// ============================================================
//
// 核心思路：临时放置棋子 → 扫描 4 方向棋型 → 还原棋盘。
// 每个方向返回 (分数, 是否强威胁)。如果 2+ 方向同时是强威胁
//（活三以上），给予 ×1.5 额外加成 —— 即"双重威胁"或"冲四活三"，
// 对手无法同时防守两个方向。
//
// ============================================================

func (h *HeuristicStrategy) evaluateMove(
	board [game_engine.BoardSize][game_engine.BoardSize]int,
	row, col, player int,
) int {
	// 临时放置棋子
	board[row][col] = player
	defer func() { board[row][col] = 0 }()

	totalScore := 0
	threatCount := 0 // 强威胁方向数（活三及以上）

	for _, d := range dirs {
		dirScore, isThreat := h.evaluateDirection(board, row, col, player, d[0], d[1])
		totalScore += dirScore
		if isThreat {
			threatCount++
		}
	}

	// 双重威胁或冲四活三加成（对手无法同时防守）
	if threatCount >= 2 {
		// 两个方向都有强威胁 → 基本必胜
		totalScore = totalScore * 3 / 2
	}

	return totalScore
}

// ============================================================
//  evaluateDirection — 评估单个方向上的棋型
// ============================================================
//
// 调用 countWithJump 获取 (连子数, 开放端数, 是否含跳棋)，
// 然后通过 patternScore 映射到具体分值。
// isThreat 标识该方向是否形成活三及以上级别的威胁。
//
// ============================================================

func (h *HeuristicStrategy) evaluateDirection(
	board [game_engine.BoardSize][game_engine.BoardSize]int,
	row, col, player int,
	dr, dc int,
) (score int, isThreat bool) {

	count, openEnds, hasGap := h.countWithJump(board, row, col, player, dr, dc)

	score = h.patternScore(count, openEnds, hasGap)

	isThreat = score >= scoreLiveThree
	return
}

// ============================================================
//  countWithJump — 带跳棋型（单 gap）检测的连子计数
// ============================================================
//
// 这是启发式引擎最核心的函数之一。
//
// 跳棋（gap）指的是中间空一格但两边有子的棋型，例如：
//   X _ X X X  （落子后，正向遇到 _ 再遇到 X → 跳冲四）
//   X X _ X X  （同理，跳活三）
//
// 算法流程（以正向扫描为例）：
//   1. 逐个扫描下一格
//   2. 遇到己方棋子 → count++
//   3. 遇到空格（且尚未使用 gap）→ 检查再下一格是否为己方棋子
//      - 是 → gap 成立，count++（只计 gap 后第一个棋子），检查 gap 后是否开放
//      - 否 → 当前空格算作开放端，停止
//   4. 遇到对方棋子 → 停止（被堵）
//   5. 越界 → 停止（被边界堵）
//
// 注意：每个方向最多使用一个 gap，且 gap 只算一个额外棋子。
//
// ============================================================

func (h *HeuristicStrategy) countWithJump(
	board [game_engine.BoardSize][game_engine.BoardSize]int,
	row, col, player int,
	dr, dc int,
) (count int, openEnds int, hasGap bool) {

	count = 1 // 落子本身
	openEnds = 0
	gapUsed := false

	// 正向扫描
	for i := 1; i < 9; i++ {
		r, c := row+dr*i, col+dc*i
		if r < 0 || r >= game_engine.BoardSize || c < 0 || c >= game_engine.BoardSize {
			break
		}
		if board[r][c] == player {
			count++
		} else if board[r][c] == 0 && !gapUsed {
			// 尝试跳过 1 格 gap
			nr, nc := row+dr*(i+1), col+dc*(i+1)
			if nr >= 0 && nr < game_engine.BoardSize &&
				nc >= 0 && nc < game_engine.BoardSize &&
				board[nr][nc] == player {
				gapUsed = true
				hasGap = true
				count++ // 只数 gap 后第一个棋子
				// 检查再后面一格是否为空（算开放端）
				nnr, nnc := row+dr*(i+2), col+dc*(i+2)
				if nnr >= 0 && nnr < game_engine.BoardSize &&
					nnc >= 0 && nnc < game_engine.BoardSize &&
					board[nnr][nnc] == 0 {
					openEnds++
				}
				break // 跳棋方向到此为止，不再续数
			}
			openEnds++
			break
		} else {
			break // 遇到对方棋子
		}
	}

	// 反向扫描（同理）
	for i := 1; i < 9; i++ {
		r, c := row-dr*i, col-dc*i
		if r < 0 || r >= game_engine.BoardSize || c < 0 || c >= game_engine.BoardSize {
			break
		}
		if board[r][c] == player {
			count++
		} else if board[r][c] == 0 && !gapUsed {
			nr, nc := row-dr*(i+1), col-dc*(i+1)
			if nr >= 0 && nr < game_engine.BoardSize &&
				nc >= 0 && nc < game_engine.BoardSize &&
				board[nr][nc] == player {
				gapUsed = true
				hasGap = true
				count++
				nnr, nnc := row-dr*(i+2), col-dc*(i+2)
				if nnr >= 0 && nnr < game_engine.BoardSize &&
					nnc >= 0 && nnc < game_engine.BoardSize &&
					board[nnr][nnc] == 0 {
					openEnds++
				}
				break
			}
			openEnds++
			break
		} else {
			break
		}
	}

	return
}

// ============================================================
//  patternScore — 棋型评分映射表
// ============================================================
//
// 输入：连子数 count、开放端数 openEnds、是否含跳棋 hasGap
//
// 核心映射（无 gap 情况）：
//   count=5 任意     → 成五    (10000000)  必胜
//   count=4 OE=2     → 活四    (100000)   双端开放，对手无法堵
//   count=4 OE=1     → 冲四    (10000)    单端开放，对手可堵
//   count=3 OE=2     → 活三    (5000)     双端开放，有潜力
//   count=3 OE=1     → 眠三    (500)      单端开放，威胁有限
//   count=2 OE=2     → 活二    (200)
//   count=2 OE=1     → 眠二    (50)
//   count=1 OE=2     → 活一    (20)
//   其他              → 0
//
// 含 gap 的棋型价值略低于同等无 gap 棋型（因为 gap 意味着连接不紧密），
// 但仍给予接近分值 + 小量附加（如跳冲四 = scoreRushFour + scoreLiveTwo）。
//
// ============================================================

func (h *HeuristicStrategy) patternScore(count, openEnds int, hasGap bool) int {
	if count >= 5 {
		if hasGap {
			// 有 gap 的 count>=5 不是真五连，按跳四处理
			if openEnds >= 2 {
				return scoreLiveFour
			}
			return scoreRushFour + scoreLiveTwo
		}
		return scoreFive
	}

	switch {
	// ---- 四连 ----
	case count == 4 && openEnds == 2:
		return scoreLiveFour
	case count == 4 && openEnds == 1:
		if hasGap {
			return scoreRushFour + scoreLiveTwo
		}
		return scoreRushFour

	// ---- 三连 ----
	case count == 3 && openEnds == 2:
		if hasGap {
			return scoreLiveThree + scoreLiveOne
		}
		return scoreLiveThree
	case count == 3 && openEnds == 1:
		if hasGap {
			return scoreSleepThree + scoreSleepTwo
		}
		return scoreSleepThree

	// ---- 二连 ----
	case count == 2 && openEnds == 2:
		if hasGap {
			return scoreLiveTwo + scoreLiveOne
		}
		return scoreLiveTwo
	case count == 2 && openEnds == 1:
		if hasGap {
			return scoreSleepTwo + scoreLiveOne
		}
		return scoreSleepTwo

	// ---- 一连 ----
	case count == 1 && openEnds == 2:
		return scoreLiveOne

	default:
		return 0
	}
}

// ============================================================
//  isWinningMove — 快速必胜检测
// ============================================================
//
// 直接沿 4 个方向统计连续同色子数（不含间隙/跳棋），
// 一旦某方向 ≥ 5 连即返回 true。
// 与 countWithJump 不同，此函数不做跳棋检测，只检查"实连"五连。
//
// ============================================================

func (h *HeuristicStrategy) isWinningMove(
	board [game_engine.BoardSize][game_engine.BoardSize]int,
	row, col, player int,
) bool {
	for _, d := range dirs {
		dr, dc := d[0], d[1]
		cnt := 1
		for i := 1; i < 5; i++ {
			r, c := row+dr*i, col+dc*i
			if r >= 0 && r < game_engine.BoardSize &&
				c >= 0 && c < game_engine.BoardSize &&
				board[r][c] == player {
				cnt++
			} else {
				break
			}
		}
		for i := 1; i < 5; i++ {
			r, c := row-dr*i, col-dc*i
			if r >= 0 && r < game_engine.BoardSize &&
				c >= 0 && c < game_engine.BoardSize &&
				board[r][c] == player {
				cnt++
			} else {
				break
			}
		}
		if cnt >= 5 {
			return true
		}
	}
	return false
}
