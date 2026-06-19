package mcts

import (
	game_engine "gomokumind/game-engine"
)

// ============================================================
//  MCTS 策略 —— 算法概述
// ============================================================
//
// 本策略实现"启发式引导的蒙特卡洛树搜索"，核心流程：
//
//   Selection（选择）
//     → Expansion（展开）
//       → Rollout（模拟/推演）
//         → Backpropagation（回传）
//
// 详细：
//
// 1. 候选预筛：启发式预评分 + softmax 归一化 → 生成先验概率（prior）。
//    根节点保留 top-30 候选，子节点保留 top-12，大幅缩小搜索空间。
//
// 2. Selection：使用 PUCT 公式选择子节点。
//    PUCT = Q(s,a) + c_puct × prior × √(parent_visits) / (1 + visits)
//    Q 从己方视角计算：如果子节点代表对手走棋，则 Q = 1 - wins/visits。
//
// 3. Expansion：从 untried 列表中取一个走法展开。若此走法直接获胜，
//    直接标记为终局并回传；否则生成子节点候选并进入 Rollout。
//
// 4. Rollout（启发式推演）：使用优先级链模拟对局至多 24 步。
//    优先级：己方成五 > 堵对手成五 > 堵冲四/活四 > 己方冲四/活四
//            > 堵活三 > 己方活三 > 指数加权随机
//    若 24 步内未分出胜负，调用全盘 scanBoard 评估判定胜者。
//
// 5. Backpropagation：沿路径向上更新 visits 和 wins。
//    平局时节点 +0.5 wins；节点 player 为胜者时 +1 wins。
//
// 6. 最终决策：选择 visits 最多的子节点作为最优走法。
//
// 关键设计决策：
//   - fastBoard 是值类型（数组而非切片），复制开销小，避免堆分配
//   - 先验概率由启发式评估 + softmax 生成，引导搜索走向有意义的区域
//   - 双重威胁检测（dualThreatScore）在评估中给予大幅加成
//   - Rollout 使用与启发式策略同源的评估函数，保持一致性
//
// 文件结构：
//   mcts.go    — 策略入口 + 数据类型 + 回传 + 辅助函数
//   tree.go    — 候选生成 + softmax 先验 + PUCT 选择
//   rollout.go — 启发式推演 + 威胁检测 + 全盘评估
//
// ============================================================
//  棋盘快照 —— 值类型数组，每轮模拟复制一份，避免堆分配
// ============================================================

type fastBoard [game_engine.BoardSize][game_engine.BoardSize]game_engine.CellState

// ============================================================
//  untriedMove — 未展开走法 + 先验概率
// ============================================================
//
// prior 由 genUntried/genChildUntried 中的 softmax 归一化生成，
// 取值范围 (0, 1]，所有 untried 的 prior 之和 ≈ 1。
// 用于 PUCT 公式中的探索项，引导搜索优先探索启发式评分高的节点。
//
// ============================================================

type untriedMove struct {
	move  game_engine.Move
	prior float64 // softmax 归一化后的先验概率
}

// ============================================================
//  mctsNode — MCTS 树节点
// ============================================================
//
// 字段说明：
//   move:   此节点代表哪一步棋（根节点为 (-1,-1)）
//   player: 此步棋由谁下（Black/White），根节点为 Empty
//   wins:   从此节点的 player 视角的获胜次数（含 0.5 平局）
//   visits: 此节点被访问的总次数
//   untried: 尚未展开的走法列表（按启发式评分降序排列）
//   prior:  PUCT 先验概率
//   children: 已展开的子节点
//
// ============================================================

type mctsNode struct {
	move     game_engine.Move
	player   game_engine.CellState
	parent   *mctsNode
	children []*mctsNode
	wins     float64
	visits   int
	untried  []untriedMove
	prior    float64 // PUCT 先验概率
}

// ============================================================
//  策略
// ============================================================

type MCTSStrategy struct {
	simulations int     // 每次 GetMove 执行的模拟次数
	cpuct       float64 // PUCT 探索常数，值越大越倾向探索（1.5 为常用值）
}

func NewMCTSStrategy(simulations int) *MCTSStrategy {
	return &MCTSStrategy{
		simulations: simulations,
		cpuct:       1.5,
	}
}

func (m *MCTSStrategy) Name() string { return "蒙特卡洛树搜索(MCTS)" }

// ============================================================
//  四个方向
// ============================================================

var dirs = [4][2]int{{0, 1}, {1, 0}, {1, 1}, {1, -1}}

// ============================================================
//  棋型分值（与启发式策略同源，值域略有差异）
// ============================================================
//
// 各棋型间呈约 10× 指数差距，保证优先级正确：
//   成五(100000) >> 活四(50000) >> 冲四(5000) >> 活三(3000)
//   >> 眠三(500) >> 活二(200) >> 眠二(50) >> 活一(15) >> 眠一(5) >> 基值(1)
//
// ============================================================

const (
	vFive       = 100000.0
	vLiveFour   = 50000.0
	vRushFour   = 5000.0
	vLiveThree  = 3000.0
	vSleepThree = 500.0
	vLiveTwo    = 200.0
	vSleepTwo   = 50.0
	vLiveOne    = 15.0
	vSleepOne   = 5.0
	vBase       = 1.0
)

// lineValue 根据连续棋子数和开放端数返回棋型分值。
// count: 连续棋子数，openEnds: 两端开放数 (0/1/2)。
func lineValue(count, openEnds int) float64 {
	if count >= 5 {
		return vFive
	}
	switch {
	case count == 4 && openEnds == 2:
		return vLiveFour
	case count == 4 && openEnds == 1:
		return vRushFour
	case count == 3 && openEnds == 2:
		return vLiveThree
	case count == 3 && openEnds == 1:
		return vSleepThree
	case count == 2 && openEnds == 2:
		return vLiveTwo
	case count == 2 && openEnds == 1:
		return vSleepTwo
	case count == 1 && openEnds == 2:
		return vLiveOne
	case count == 1 && openEnds == 1:
		return vSleepOne
	}
	return vBase
}

// ============================================================
//  GetMove — MCTS 主入口
// ============================================================
//
// 流程概览：
//   1. 棋盘格式转换（int → CellState）
//   2. 候选走法生成（周围 2 格剪枝）
//   3. 必胜/必防速查（直接成五 → 立即返回）
//   4. 预评分 + softmax 归一化 → 生成根节点 prior，限缩到 top-30
//   5. MCTS 主循环（selection → expansion → rollout → backprop）× N 次
//   6. 选择 visits 最多的子节点
//
// ============================================================

func (m *MCTSStrategy) GetMove(
	boardInt [game_engine.BoardSize][game_engine.BoardSize]int,
	player int,
) game_engine.Move {

	// ---- 1. 转换棋盘 ----
	var rootBoard fastBoard
	for i := 0; i < game_engine.BoardSize; i++ {
		for j := 0; j < game_engine.BoardSize; j++ {
			switch boardInt[i][j] {
			case 1:
				rootBoard[i][j] = game_engine.Black
			case 2:
				rootBoard[i][j] = game_engine.White
			}
		}
	}

	self := game_engine.Black
	opp := game_engine.White
	if player == 2 {
		self, opp = opp, self
	}

	// ---- 2. 候选走法 ----
	cands := getCandidates(rootBoard)
	if len(cands) == 0 {
		return game_engine.Move{Row: 7, Col: 7}
	}

	// ---- 3. 必胜必防 ----
	if wm, ok := findWinningMove(rootBoard, cands, self); ok {
		return wm
	}
	if bm, ok := findWinningMove(rootBoard, cands, opp); ok {
		return bm
	}
	if len(cands) == 1 {
		return cands[0]
	}

	// ---- 4. 预评分 + 限缩候选到前 N 个 + 生成先验 ----
	const maxRootCands = 30
	untried := genUntried(rootBoard, cands, self, opp, maxRootCands)

	// ---- 5. MCTS 主循环 ----
	//
	// 每轮模拟：
	//   Selection: 沿树用 PUCT 选择到可展开或终端节点
	//   Expansion: 从 untried 取一个走法展开为新节点
	//   Rollout:   启发式推演至多 24 步，返回胜者
	//   Backprop:  沿路径回传胜负结果
	//
	root := &mctsNode{
		move:    game_engine.Move{Row: -1, Col: -1},
		player:  game_engine.Empty,
		untried: untried,
	}

	for sim := 0; sim < m.simulations; sim++ {
		// 每轮模拟使用独立的棋盘副本（值拷贝，开销很低）
		simBoard := rootBoard
		node := root
		terminal := false

		// ---- Selection：沿树走到可展开或终端节点 ----
		// 选择过程中同时更新 simBoard（沿路径落子）
		for len(node.untried) == 0 && len(node.children) > 0 {
			node = m.selectChild(node, self)
			simBoard[node.move.Row][node.move.Col] = node.player
			if isWinAt(&simBoard, node.move) {
				terminal = true
				break
			}
		}

		if terminal {
			m.backprop(node, node.player)
			continue
		}

		// ---- Expansion：展开一个未尝试的走法 ----
		if len(node.untried) > 0 {
			// 取优先级最高的未尝试走法（untried 按启发式评分降序排列）
			um := node.untried[0]
			node.untried = node.untried[1:]

			// 确定下一步走棋方
			nextPlayer := opponentOf(node.player, self, opp)
			if node.player == game_engine.Empty {
				nextPlayer = self
			}

			child := &mctsNode{
				move:   um.move,
				player: nextPlayer,
				parent: node,
				prior:  um.prior,
			}
			simBoard[um.move.Row][um.move.Col] = nextPlayer

			// 如果此走法直接获胜，不需要展开和 rollout，直接标记为胜利
			if isWinAt(&simBoard, um.move) {
				child.visits = 1
				child.wins = 1
				node.children = append(node.children, child)
				m.backprop(child, nextPlayer)
				continue
			}

			// 为子节点生成候选走法（带先验），限缩到 top-12
			child.untried = genChildUntried(simBoard, opponentOf(nextPlayer, self, opp), 12)
			node.children = append(node.children, child)

			// ---- Rollout：启发式推演 ----
			rolloutPlayer := opponentOf(nextPlayer, self, opp)
			winner := m.heuristicRollout(&simBoard, rolloutPlayer)
			m.backprop(child, winner)
		}
	}

	return m.bestChild(root, cands).move
}

// ============================================================
//  backprop — 沿路径回传模拟结果
// ============================================================
//
// 对路径上每个节点：
//   - visits++（访问计数）
//   - winner == Empty → wins += 0.5（平局，各算半胜）
//   - node.player == winner → wins++（此节点的棋手获胜）
//   - 否则不加（此节点的棋手落败）
//
// 注意：wins 是从 node.player 视角记录的。
// 这意味着在 PUCT 中，需要根据 child.player 是否等于 self 来转换视角。
//
// ============================================================

func (m *MCTSStrategy) backprop(node *mctsNode, winner game_engine.CellState) {
	for node != nil {
		node.visits++
		if winner == game_engine.Empty {
			node.wins += 0.5
		} else if node.player == winner {
			node.wins++
		}
		node = node.parent
	}
}

// bestChild 选择访问次数最多的子节点作为最终决策。
//
// 在 MCTS 中，最终决策通常有两种方式：
//  1. max visits（最稳健，访问越多 = 越可信）
//  2. max Q value（理论上最优但更噪声敏感）
//
// 本实现采用方式 1，是业界通用做法。
//
// fallback 用于根节点没有子节点时的兜底（返回候选列表中第一个）。
func (m *MCTSStrategy) bestChild(root *mctsNode, fallback []game_engine.Move) *mctsNode {
	var best *mctsNode
	bestVisits := -1
	for _, child := range root.children {
		if child.visits > bestVisits {
			bestVisits = child.visits
			best = child
		}
	}
	if best == nil {
		return &mctsNode{move: fallback[0]}
	}
	return best
}

// ============================================================
//  辅助函数
// ============================================================

// getCandidates 生成候选落子：已有棋子周围 2 格内的空位。
// 使用 seen 数组去重；空棋盘时返回中心点 (7,7)。
func getCandidates(board fastBoard) []game_engine.Move {
	seen := make([]bool, game_engine.BoardSize*game_engine.BoardSize)
	var cands []game_engine.Move
	hasPiece := false
	for i := 0; i < game_engine.BoardSize; i++ {
		for j := 0; j < game_engine.BoardSize; j++ {
			if board[i][j] != game_engine.Empty {
				hasPiece = true
				for di := -2; di <= 2; di++ {
					for dj := -2; dj <= 2; dj++ {
						r, c := i+di, j+dj
						if r >= 0 && r < game_engine.BoardSize &&
							c >= 0 && c < game_engine.BoardSize &&
							board[r][c] == game_engine.Empty {
							key := r*game_engine.BoardSize + c
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

// findWinningMove 在候选列表中搜索能直接成五的走法（不模拟落子，仅检测现有棋子）。
func findWinningMove(
	board fastBoard,
	cands []game_engine.Move,
	player game_engine.CellState,
) (game_engine.Move, bool) {
	for _, mv := range cands {
		if wouldWin(board, mv.Row, mv.Col, player) {
			return mv, true
		}
	}
	return game_engine.Move{}, false
}

// wouldWin 判断在 (r,c) 落 player 后是否形成五连（不计跳跃，仅实连检测）。
func wouldWin(
	board fastBoard,
	r, c int,
	player game_engine.CellState,
) bool {
	for _, d := range dirs {
		dr, dc := d[0], d[1]
		cnt := 1
		for i := 1; i < 5; i++ {
			nr, nc := r+dr*i, c+dc*i
			if nr < 0 || nr >= game_engine.BoardSize ||
				nc < 0 || nc >= game_engine.BoardSize ||
				board[nr][nc] != player {
				break
			}
			cnt++
		}
		for i := 1; i < 5; i++ {
			nr, nc := r-dr*i, c-dc*i
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

// isWinAt 判断棋盘上 mv 位置的棋子是否形成五连（需要该位置已有棋子）。
func isWinAt(board *fastBoard, mv game_engine.Move) bool {
	if mv.Row < 0 {
		return false
	}
	p := board[mv.Row][mv.Col]
	return p != game_engine.Empty && wouldWin(*board, mv.Row, mv.Col, p)
}

// opponentOf 返回对手颜色：player==a 则返回 b，否则返回 a。
func opponentOf(player, a, b game_engine.CellState) game_engine.CellState {
	if player == a {
		return b
	}
	return a
}
