package mcts

import (
	"math"
	"math/rand"
	"sort"

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
	prior float64 // 软最大值归一化后的先验概率
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

func (m *MCTSStrategy) Name() string             { return "蒙特卡洛树搜索(MCTS)" }
func (m *MCTSStrategy) Train(episodes int) error { return nil }

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
			case -1:
				rootBoard[i][j] = game_engine.White
			}
		}
	}

	self := game_engine.Black
	opp := game_engine.White
	if player == -1 {
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
	//   Selection: 沿树用 PUCT 选择到可展开节点或终端节点
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

			// 确定下一步走棋方：如果当前节点 player 是 Empty（根节点），
			// 则用 self；否则用 opponentOf 切换
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
//  genUntried — 根节点候选 + 软最大值先验
// ============================================================
//
// 与 genChildUntried 的区别：
//   - 根节点同时评估攻击分和防守分（atk + def × 0.9），子节点仅评估攻击分
//   - 根节点限缩到 top-30，子节点限缩到 top-12
//   - 根节点 temperature=5000，子节点 temperature=3000（更"尖锐"的先验）
//
// softmax 温度参数的作用：
//   温度越高 → 先验分布越平滑（探索更多）
//   温度越低 → 先验分布越尖锐（集中利用高分支）
//   子节点 temperature 更低，意味着深层搜索更倾向于利用而非探索。
//
// ============================================================

func genUntried(
	board fastBoard,
	cands []game_engine.Move,
	self, opp game_engine.CellState,
	maxN int,
) []untriedMove {

	type scored struct {
		move  game_engine.Move
		score float64
	}
	items := make([]scored, len(cands))
	for i, mv := range cands {
		board[mv.Row][mv.Col] = self
		atk := evalPos(&board, mv.Row, mv.Col, self)
		board[mv.Row][mv.Col] = game_engine.Empty

		board[mv.Row][mv.Col] = opp
		def := evalPos(&board, mv.Row, mv.Col, opp)
		board[mv.Row][mv.Col] = game_engine.Empty

		items[i] = scored{move: mv, score: atk + def*0.9}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].score > items[j].score
	})

	n := len(items)
	if n > maxN {
		n = maxN
	}

	// 软最大值归一化
	scores := make([]float64, n)
	maxScore := items[0].score
	if maxScore <= 0 {
		maxScore = 1
	}
	for i := 0; i < n; i++ {
		scores[i] = items[i].score
	}

	const temperature = 5000.0
	sum := 0.0
	probs := make([]float64, n)
	for i := 0; i < n; i++ {
		probs[i] = math.Exp((scores[i] - maxScore) / temperature)
		sum += probs[i]
	}

	result := make([]untriedMove, n)
	for i := 0; i < n; i++ {
		result[i] = untriedMove{
			move:  items[i].move,
			prior: probs[i] / sum,
		}
	}
	return result
}

// ============================================================
//  genChildUntried — 子节点候选 + 软最大值先验
// ============================================================

func genChildUntried(
	board fastBoard,
	nextPlayer game_engine.CellState,
	topN int,
) []untriedMove {
	all := getCandidates(board)
	if len(all) == 0 {
		return nil
	}

	type scored struct {
		move  game_engine.Move
		score float64
	}
	items := make([]scored, len(all))
	for i, mv := range all {
		board[mv.Row][mv.Col] = nextPlayer
		items[i] = scored{move: mv, score: evalPos(&board, mv.Row, mv.Col, nextPlayer)}
		board[mv.Row][mv.Col] = game_engine.Empty
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].score > items[j].score
	})

	n := len(items)
	if n > topN {
		n = topN
	}
	if n == 0 {
		return nil
	}

	// 软最大值归一化
	maxScore := items[0].score
	if maxScore <= 0 {
		maxScore = 1
	}
	const temperature = 3000.0
	sum := 0.0
	probs := make([]float64, n)
	for i := 0; i < n; i++ {
		probs[i] = math.Exp((items[i].score - maxScore) / temperature)
		sum += probs[i]
	}

	result := make([]untriedMove, n)
	for i := 0; i < n; i++ {
		result[i] = untriedMove{
			move:  items[i].move,
			prior: probs[i] / sum,
		}
	}
	return result
}

// ============================================================
//  PUCT 选择公式
// ============================================================
//
//  PUCT(s,a) = Q(s,a) + c_puct × prior × √(parent_visits) / (1 + visits)
//
//  其中：
//    Q(s,a)  = 从 self（己方）视角的胜率
//              - 若 child.player == self：Q = wins / visits（高胜率 = 好）
//              - 若 child.player != self：Q = 1 - wins/visits（对手高胜率 = 坏）
//    prior   = 归一化后的先验概率（来自启发式预评分）
//    explore = UCT 探索项，鼓励访问较少的节点
//
//  特殊情况：child.visits == 0 时直接返回该节点（保证所有节点至少被访问一次）。
//  prior <= 0 时用均匀分布作为 fallback。
//
// ============================================================

func (m *MCTSStrategy) selectChild(node *mctsNode, self game_engine.CellState) *mctsNode {
	parentVisits := float64(node.visits + 1)
	bestScore := -math.MaxFloat64
	var best *mctsNode

	for _, child := range node.children {
		if child.visits == 0 {
			return child
		}

		q := child.wins / float64(child.visits)
		if child.player != self {
			q = 1.0 - q
		}

		prior := child.prior
		if prior <= 0 {
			prior = 1.0 / float64(len(node.children))
		}
		explore := m.cpuct * prior * math.Sqrt(parentVisits) / (1.0 + float64(child.visits))

		score := q + explore
		if score > bestScore {
			bestScore = score
			best = child
		}
	}
	if best == nil {
		return node.children[0]
	}
	return best
}

// ============================================================
//  heuristicRollout — 启发式推演（截断 24 步 → 全盘评估判胜负）
// ============================================================
//
// Rollout 是 MCTS 中"快速模拟对局到结束"的阶段。
// 本实现使用启发式策略而非纯随机推演，大幅提高模拟质量。
//
// 优先级链（从高到低）：
//   1. 己方能直接成五 → 立即获胜，返回当前玩家
//   2. 堵对手成五     → 必须堵，否则对手下一手下赢
//   3. 堵对手冲四/活四 → 高威胁，必须应对
//   4. 己方冲四/活四   → 主动进攻
//   5. 堵对手活三      → 应对中等威胁
//   6. 己方活三        → 主动发展
//   7. 指数加权随机    → 软随机，保持一定探索性
//
// 最多模拟 24 步（12 回合），若未分胜负则调用全盘 scanBoard 评估。
// 24 步截断是性能与精度的折中：太短则评估不准，太长则耗时过多。
//
// ============================================================

func (m *MCTSStrategy) heuristicRollout(
	board *fastBoard,
	startPlayer game_engine.CellState,
) game_engine.CellState {

	cur := startPlayer
	opp := opponentOf(cur, game_engine.Black, game_engine.White)
	const maxSteps = 24
	steps := 0

	for steps < maxSteps {
		cands := getCandidates(*board)
		if len(cands) == 0 {
			return game_engine.Empty
		}

		// 1) 直接赢
		if _, ok := findWinningMove(*board, cands, cur); ok {
			return cur
		}
		// 2) 堵对手成五
		if block, ok := findWinningMove(*board, cands, opp); ok {
			board[block.Row][block.Col] = cur
			steps++
			cur, opp = opp, cur
			continue
		}
		// 3) 堵对手冲四/活四
		if block, ok := findThreat(*board, cands, opp, vRushFour); ok {
			board[block.Row][block.Col] = cur
			steps++
			cur, opp = opp, cur
			continue
		}
		// 4) 己方冲四/活四
		if attack, ok := findThreat(*board, cands, cur, vRushFour); ok {
			board[attack.Row][attack.Col] = cur
			steps++
			if isWinAt(board, attack) {
				return cur
			}
			cur, opp = opp, cur
			continue
		}
		// 5) 堵对手活三
		if block, ok := findThreat(*board, cands, opp, vLiveThree); ok {
			board[block.Row][block.Col] = cur
			steps++
			cur, opp = opp, cur
			continue
		}
		// 6) 己方活三
		if attack, ok := findThreat(*board, cands, cur, vLiveThree); ok {
			board[attack.Row][attack.Col] = cur
			steps++
			if isWinAt(board, attack) {
				return cur
			}
			cur, opp = opp, cur
			continue
		}
		// 7) 指数加权随机
		mv := weightedPick(*board, cands, cur)
		board[mv.Row][mv.Col] = cur
		steps++
		if isWinAt(board, mv) {
			return cur
		}
		cur, opp = opp, cur
	}

	return evalWinner(board, startPlayer)
}

// ============================================================
//  findThreat — 某玩家在此是否形成 ≥ threshold 的威胁
// ============================================================

func findThreat(
	board fastBoard,
	cands []game_engine.Move,
	player game_engine.CellState,
	threshold float64,
) (game_engine.Move, bool) {
	var best game_engine.Move
	bestScore := 0.0
	found := false
	for _, mv := range cands {
		board[mv.Row][mv.Col] = player
		s := evalPos(&board, mv.Row, mv.Col, player)
		board[mv.Row][mv.Col] = game_engine.Empty
		if s >= threshold && s > bestScore {
			bestScore = s
			best = mv
			found = true
		}
	}
	return best, found
}

// ============================================================
//  weightedPick — 指数权重选择
// ============================================================

func weightedPick(
	board fastBoard,
	cands []game_engine.Move,
	player game_engine.CellState,
) game_engine.Move {
	opp := opponentOf(player, game_engine.Black, game_engine.White)
	type item struct {
		move   game_engine.Move
		weight float64
	}
	items := make([]item, len(cands))
	sum := 0.0
	for i, mv := range cands {
		board[mv.Row][mv.Col] = player
		atk := evalPos(&board, mv.Row, mv.Col, player)
		board[mv.Row][mv.Col] = game_engine.Empty

		board[mv.Row][mv.Col] = opp
		def := evalPos(&board, mv.Row, mv.Col, opp)
		board[mv.Row][mv.Col] = game_engine.Empty

		defWeight := 0.9
		if def >= vRushFour {
			defWeight = 1.1
		}
		if def >= vLiveFour {
			defWeight = 1.2
		}
		raw := atk + def*defWeight
		w := math.Exp(raw / 5000.0)
		items[i] = item{move: mv, weight: w}
		sum += w
	}
	r := rand.Float64() * sum
	for _, it := range items {
		r -= it.weight
		if r <= 0 {
			return it.move
		}
	}
	return items[len(items)-1].move
}

// ============================================================
//  dualThreatScore — 双重威胁检测与评分
// ============================================================
//
// 这是 MCTS 引擎评估的核心。不是简单回传各方向分值之和，
// 而是在此基础上检测是否同时形成多个方向的威胁。
//
// 威胁判定：
//   - lv >= vLiveFour → 活四及以上，记为威胁方向
//   - lv >= vLiveThree → 活三，记为威胁方向
//
// 加成系数：
//   threats >= 2 且 同时有活四和活三 → ×20（必胜级别）
//   threats >= 2 仅有其他组合       → ×5
//
// 直觉解释：
//   一步棋同时在两个方向形成活三 → 对手只能堵一个方向
//   → 另一个方向下一步即可冲四/活四 → 必胜
//   这是五子棋最核心的战术模式。
//
// ============================================================

func dualThreatScore(
	board *fastBoard,
	row, col int,
	player game_engine.CellState,
) float64 {
	// 计算各方向分值，统计 ≥ vLiveThree 的方向数
	threats := 0
	hasLiveThree := false
	hasLiveFour := false
	totalScore := 0.0

	for _, d := range dirs {
		cnt, oe := countLine(board, row, col, player, d[0], d[1])
		lv := lineValue(cnt, oe)
		totalScore += lv
		if lv >= vLiveFour {
			hasLiveFour = true
			threats++
		} else if lv >= vLiveThree {
			hasLiveThree = true
			threats++
		}
	}

	// 双威胁以上：大幅加成
	if threats >= 2 {
		bonus := totalScore * 5.0 // 双威胁 = 基本必胜
		// 活三 + 冲四（或活四）的双威胁 → 直接接近必胜
		if hasLiveFour && hasLiveThree {
			bonus = totalScore * 20.0
		}
		return bonus
	}
	return totalScore
}

// ============================================================
//  evalPos — 评估 (row,col) 落子后 player 的棋型分（含双威胁）
// ============================================================

func evalPos(board *fastBoard, row, col int, player game_engine.CellState) float64 {
	return dualThreatScore(board, row, col, player)
}

// ============================================================
//  countLine — 沿 (dr,dc) 方向计连续棋子数 + 开放端数（含跳棋）
// ============================================================
//
// 正向和反向各扫描一次，合并计数。
// 这与启发式策略的 countWithJump 功能等价，但实现风格略有不同。
//
// ============================================================

func countLine(
	board *fastBoard,
	row, col int,
	player game_engine.CellState,
	dr, dc int,
) (count int, openEnds int) {
	count = 1

	cnt, oe := countDir(board, row, col, player, dr, dc)
	count += cnt
	openEnds += oe

	cnt, oe = countDir(board, row, col, player, -dr, -dc)
	count += cnt
	openEnds += oe

	return
}

// countDir 从 (row,col) 沿 (dr,dc) 方向计数连续同色棋子，
// 支持单跳棋模式（如 X_XXX 中的间隙，跳过空格后继续数）。
//
// 跳棋逻辑：
//
//	遇到空格 → 检查下一格是否为己方棋子
//	  - 是 → jumped=true，跳过当前空格，后续不再尝试跳棋
//	  - 否 → 当前空格算开放端，停止扫描
func countDir(
	board *fastBoard,
	row, col int,
	player game_engine.CellState,
	dr, dc int,
) (cnt int, openEnds int) {
	jumped := false
	for i := 1; i <= 5; i++ {
		r, c := row+dr*i, col+dc*i
		if r < 0 || r >= game_engine.BoardSize || c < 0 || c >= game_engine.BoardSize {
			break
		}
		if board[r][c] == player {
			cnt++
		} else if board[r][c] == game_engine.Empty && !jumped {
			nr, nc := r+dr, c+dc
			if nr >= 0 && nr < game_engine.BoardSize &&
				nc >= 0 && nc < game_engine.BoardSize &&
				board[nr][nc] == player {
				jumped = true
				continue
			}
			openEnds++
			break
		} else {
			if board[r][c] == game_engine.Empty {
				openEnds++
			}
			break
		}
	}
	return
}

// ============================================================
//  evalWinner — 全盘评估判定哪方优势
// ============================================================
//
// 分别在棋盘上全面扫描双方所有连续棋段，累加其棋型分值。
// 判定规则：一方分值超过另一方 ×1.15 时判定该方胜。
// 1.15 的阈值避免了噪声导致误判（双方分值接近时判平局）。
//
// ============================================================

func evalWinner(board *fastBoard, perspective game_engine.CellState) game_engine.CellState {
	opp := opponentOf(perspective, game_engine.Black, game_engine.White)
	selfScore := scanBoard(board, perspective)
	oppScore := scanBoard(board, opp)
	if selfScore > oppScore*1.15 {
		return perspective
	}
	if oppScore > selfScore*1.15 {
		return opp
	}
	return game_engine.Empty
}

// scanBoard 逐行逐列逐对角线扫描所有连续同色棋段，
// 每个棋段只计一次分（count, openEnds → lineValue）。
// 不检测跳棋，只计实连。
func scanBoard(board *fastBoard, player game_engine.CellState) float64 {
	at := func(r, c int) game_engine.CellState {
		if r < 0 || r >= game_engine.BoardSize || c < 0 || c >= game_engine.BoardSize {
			return game_engine.Empty
		}
		return board[r][c]
	}
	B := game_engine.BoardSize
	score := 0.0

	// 水平
	for r := 0; r < B; r++ {
		for c := 0; c < B; {
			if at(r, c) != player {
				c++
				continue
			}
			start := c
			for c < B && at(r, c) == player {
				c++
			}
			cnt := c - start
			if cnt > 5 {
				cnt = 5
			}
			oe := 0
			if at(r, start-1) == game_engine.Empty {
				oe++
			}
			if at(r, start+cnt) == game_engine.Empty {
				oe++
			}
			score += lineValue(cnt, oe)
		}
	}
	// 垂直
	for c := 0; c < B; c++ {
		for r := 0; r < B; {
			if at(r, c) != player {
				r++
				continue
			}
			start := r
			for r < B && at(r, c) == player {
				r++
			}
			cnt := r - start
			if cnt > 5 {
				cnt = 5
			}
			oe := 0
			if at(start-1, c) == game_engine.Empty {
				oe++
			}
			if at(start+cnt, c) == game_engine.Empty {
				oe++
			}
			score += lineValue(cnt, oe)
		}
	}
	// 对角线 \
	for r := 0; r < B; r++ {
		for c := 0; c < B; {
			if at(r, c) != player {
				c++
				continue
			}
			startC := c
			k := 0
			for r+k < B && startC+k < B && at(r+k, startC+k) == player {
				k++
			}
			cnt := k
			if cnt > 5 {
				cnt = 5
			}
			oe := 0
			if at(r-1, startC-1) == game_engine.Empty {
				oe++
			}
			if at(r+cnt, startC+cnt) == game_engine.Empty {
				oe++
			}
			score += lineValue(cnt, oe)
			c = startC + k
		}
	}
	// 反对角线 /
	for r := 0; r < B; r++ {
		for c := 0; c < B; {
			if at(r, c) != player {
				c++
				continue
			}
			startC := c
			k := 0
			for r+k < B && startC-k >= 0 && at(r+k, startC-k) == player {
				k++
			}
			cnt := k
			if cnt > 5 {
				cnt = 5
			}
			oe := 0
			if at(r-1, startC+1) == game_engine.Empty {
				oe++
			}
			if at(r+cnt, startC-cnt) == game_engine.Empty {
				oe++
			}
			score += lineValue(cnt, oe)
			c = startC + 1
		}
	}
	return score
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

// getCandidates 生成候选落子：已有棋子周围 2 格内的空位
// 使用 seen 数组去重，空棋盘时返回中心点 (7,7)
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

// findWinningMove 在候选列表中搜索能直接成五的走法（不模拟落子，仅检测现有棋子）
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

// wouldWin 判断在 (r,c) 落 player 后是否形成五连（不计跳跃，仅实连检测）
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

// isWinAt 判断棋盘上 mv 位置的棋子是否形成五连（需要该位置已有棋子）
func isWinAt(board *fastBoard, mv game_engine.Move) bool {
	if mv.Row < 0 {
		return false
	}
	p := board[mv.Row][mv.Col]
	return p != game_engine.Empty && wouldWin(*board, mv.Row, mv.Col, p)
}

// opponentOf 返回对手颜色：player==a 则返回 b，否则返回 a
func opponentOf(player, a, b game_engine.CellState) game_engine.CellState {
	if player == a {
		return b
	}
	return a
}
