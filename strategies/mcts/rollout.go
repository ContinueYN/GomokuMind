package mcts

import (
	"math"
	"math/rand"

	game_engine "gomokumind/game-engine"
)

// ============================================================
//  heuristicRollout — 启发式推演（截断 24 步 → 全盘评估判胜负）
// ============================================================
//
// Rollout 是 MCTS 中"快速模拟对局到结束"的阶段。
// 本实现使用启发式策略而非纯随机推演，大幅提高模拟质量。
//
// 优先级链（从高到低）：
//   1. 己方能直接成五 → 立即获胜，返回当前玩家
//   2. 堵对手成五     → 必须堵，否则对手下一手即赢
//   3. 堵对手冲四/活四 → 高威胁，必须应对
//   4. 己方冲四/活四   → 主动进攻
//   5. 堵对手活三      → 应对中等威胁
//   6. 己方活三        → 主动发展
//   7. 指数加权随机    → 软随机，保持探索性
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
		// 2) 堵对手成五（优先级高于己方进攻，因为不堵就输了）
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
		// 7) 指数加权随机（无明确威胁时，按攻击/防守分加权随机选择）
		mv := weightedPick(*board, cands, cur)
		board[mv.Row][mv.Col] = cur
		steps++
		if isWinAt(board, mv) {
			return cur
		}
		cur, opp = opp, cur
	}

	// 截断后全盘评估判定胜者
	return evalWinner(board, startPlayer)
}

// ============================================================
//  findThreat — 在候选走法中查找某玩家是否存在 ≥ threshold 的威胁
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
//  weightedPick — 指数加权随机选择
// ============================================================
//
// 对候选走法计算"攻击分 + 防守分 × 防守权重"，经 softmax-ish 指数变换后随机采样。
// 防守权重在对手有强威胁（冲四/活四）时动态提升，确保关键防守优先。
//
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
		// 攻击分：己方落子后的棋型评分
		board[mv.Row][mv.Col] = player
		atk := evalPos(&board, mv.Row, mv.Col, player)
		board[mv.Row][mv.Col] = game_engine.Empty

		// 防守分：假设对手落子后的评分（即此位置的防守价值）
		board[mv.Row][mv.Col] = opp
		def := evalPos(&board, mv.Row, mv.Col, opp)
		board[mv.Row][mv.Col] = game_engine.Empty

		// 防守权重动态调整：对手威胁越大，防守越重要
		defWeight := 0.9
		if def >= vRushFour {
			defWeight = 1.1 // 对手冲四 → 防守优先
		}
		if def >= vLiveFour {
			defWeight = 1.2 // 对手活四 → 必须防守
		}
		raw := atk + def*defWeight
		w := math.Exp(raw / 5000.0) // 指数变换使高分走法权重显著更大
		items[i] = item{move: mv, weight: w}
		sum += w
	}
	// 轮盘赌采样
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
// 这是 MCTS 评估的核心增强。不是简单累加四方向分值，
// 而是检测一步棋是否同时在多个方向形成威胁。
//
// 威胁判定：
//   - lv >= vLiveFour → 活四及以上，记为威胁方向
//   - lv >= vLiveThree → 活三及以上，记为威胁方向
//
// 加成系数：
//   threats >= 2 且同时有活四和活三 → ×20（必胜级别）
//   threats >= 2 仅有其他组合       → ×5
//
// 直觉解释：
//   一步棋同时在两个方向形成活三 → 对手只能堵一个方向
//   → 另一个方向下一步即可冲四/活四 → 必胜
//   这是五子棋最核心的战术模式（双杀）。
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
		// 活三 + 活四（或冲四）的双威胁 → 直接接近必胜
		if hasLiveFour && hasLiveThree {
			bonus = totalScore * 20.0
		}
		return bonus
	}
	return totalScore
}

// ============================================================
//  evalPos — 评估 (row,col) 落子后 player 的棋型分（含双威胁检测）
// ============================================================

func evalPos(board *fastBoard, row, col int, player game_engine.CellState) float64 {
	return dualThreatScore(board, row, col, player)
}

// ============================================================
//  countLine — 沿 (dr,dc) 方向计连续棋子数 + 开放端数（含跳棋）
// ============================================================
//
// 正向和反向各扫描一次，合并计数。
// 与启发式策略的 countWithJump 功能等价。
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
			// 跳棋检测：空格后紧跟己方棋子 → 跳过此空格
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
//  evalWinner — 全盘评估判定哪方占优
// ============================================================
//
// 分别在棋盘上全面扫描双方所有连续棋段，累加棋型分值。
// 判定规则：一方分值超过另一方 ×1.15 时判定该方胜。
// 1.15 的阈值避免噪声误判（双方分值接近时判平局）。
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

	// 水平方向
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
				cnt = 5 // 超过五连按五连算
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
	// 垂直方向
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
	// 主对角线 (\)
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
	// 反对角线 (/)
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
