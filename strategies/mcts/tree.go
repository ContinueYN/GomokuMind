package mcts

import (
	"math"
	"sort"

	game_engine "gomokumind/game-engine"
)

// ============================================================
//  genUntried — 根节点候选生成 + softmax 先验概率
// ============================================================
//
// 与 genChildUntried 的区别：
//   - 根节点同时评估攻击分和防守分（atk + def × 0.9），子节点仅评估攻击分
//   - 根节点限缩到 top-30，子节点限缩到 top-12（深层分支更少 → 搜索更集中）
//   - 根节点 temperature=5000，子节点 temperature=3000（子节点先验更"尖锐"）
//
// softmax 温度参数的作用：
//   温度越高 → 先验分布越平滑（鼓励探索）
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
		// 攻击分：己方在此落子后的棋型评分
		board[mv.Row][mv.Col] = self
		atk := evalPos(&board, mv.Row, mv.Col, self)
		board[mv.Row][mv.Col] = game_engine.Empty

		// 防守分：假设对手落子后的评分（即此位置的防守价值）
		board[mv.Row][mv.Col] = opp
		def := evalPos(&board, mv.Row, mv.Col, opp)
		board[mv.Row][mv.Col] = game_engine.Empty

		// 综合评分：攻击 + 防守 × 0.9（攻击略重于防守）
		items[i] = scored{move: mv, score: atk + def*0.9}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].score > items[j].score
	})

	// 限缩到前 N 个
	n := len(items)
	if n > maxN {
		n = maxN
	}

	// softmax 归一化生成先验概率
	scores := make([]float64, n)
	maxScore := items[0].score
	if maxScore <= 0 {
		maxScore = 1 // 防止除零
	}
	for i := 0; i < n; i++ {
		scores[i] = items[i].score
	}

	const temperature = 5000.0 // 根节点温度 → 较平滑的先验
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
//  genChildUntried — 子节点候选生成 + softmax 先验概率
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
		// 子节点只评估攻击分（不混合防守分）以保持评估一致性
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

	// softmax 归一化
	maxScore := items[0].score
	if maxScore <= 0 {
		maxScore = 1
	}
	const temperature = 3000.0 // 子节点温度更低 → 更尖锐的先验分布
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
//  selectChild — PUCT 子节点选择
// ============================================================
//
//  PUCT(s,a) = Q(s,a) + c_puct × prior × √(parent_visits) / (1 + visits)
//
//  其中：
//    Q(s,a)  = 从 self（己方）视角的胜率
//              - 若 child.player == self：Q = wins / visits（高胜率 = 好）
//              - 若 child.player != self：Q = 1 - wins/visits（对手高胜率 = 坏）
//              MCTS 中 wins 从 node.player 视角记录，所以需要视角转换。
//    prior   = softmax 归一化后的先验概率（来自启发式预评分）
//    explore = UCT 探索项，鼓励访问较少的节点
//
//  特殊情况：child.visits == 0 时直接返回该节点（保证所有节点至少被访问一次）。
//  prior <= 0 时用均匀分布作为 fallback。
//
// ============================================================

func (m *MCTSStrategy) selectChild(node *mctsNode, self game_engine.CellState) *mctsNode {
	parentVisits := float64(node.visits + 1) // +1 避免根节点 visits=0 时除零
	bestScore := -math.MaxFloat64
	var best *mctsNode

	for _, child := range node.children {
		// 未访问过的节点优先：确保探索完整性
		if child.visits == 0 {
			return child
		}

		// 从己方视角计算 Q 值
		q := child.wins / float64(child.visits)
		if child.player != self {
			// 子节点是对手下棋 → 对手胜率高 = 己方不利
			q = 1.0 - q
		}

		prior := child.prior
		if prior <= 0 {
			prior = 1.0 / float64(len(node.children)) // fallback：均匀分布
		}
		// UCT 探索项：鼓励访问次数少的节点
		explore := m.cpuct * prior * math.Sqrt(parentVisits) / (1.0 + float64(child.visits))

		score := q + explore
		if score > bestScore {
			bestScore = score
			best = child
		}
	}
	if best == nil {
		return node.children[0] // 极端情况回退
	}
	return best
}
