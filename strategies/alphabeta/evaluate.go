package alphabeta

// ============================================================
//  boardEval —— 棋盘评估器
// ============================================================
//
// 核心思路：
//   1. 四方向全盘扫描 → 缓存每格在各方向的棋型
//   2. 统计各棋型出现次数
//   3. 优先级链评分：五连 > 活四 > 冲四 > 四三 > 活三 > ...
//   4. 跳棋（单 gap）检测：X_XXX → 冲四, XX_XX → 冲四
//
// 11 种棋型：
//   notype(0) / stwo(1) / two(2) / sthree(3) / three(4) /
//   sfour(5) / four(6) / five(7) / dfour(8) / fourt(9) / dthree(10)

// ============================================================

type boardEval struct {
	record [boardSize][boardSize][4]int // 四方向棋型缓存: [row][col][dir]
	count  [3][20]int                   // count[棋子色][棋型] = 出现次数
	line   [30]int                      // 临时线段缓冲（单条线最多15格+哨兵）
	result [30]int                      // 临时结果缓冲（与 line 对应）
}

func (e *boardEval) reset() {
	// 标记所有格子的所有方向为"待分析"
	for i := 0; i < boardSize; i++ {
		for j := 0; j < boardSize; j++ {
			e.record[i][j][0] = todo
			e.record[i][j][1] = todo
			e.record[i][j][2] = todo
			e.record[i][j][3] = todo
		}
	}
	// 清零棋型计数器
	for s := 0; s < 3; s++ {
		for p := 0; p < 20; p++ {
			e.count[s][p] = 0
		}
	}
}

// evaluate 对棋盘做完整评估，返回 turn 视角的分数。
// 正分 = turn 优势，负分 = turn 劣势。
// 绝对值 ≥ 9000 表示接近必胜/必败局面。
func (e *boardEval) evaluate(b *boardIntl, turn int) int {
	score := e.evalRaw(b, turn)
	// 极端局面微调：用对方棋型数量做偏移，打破平局
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

// evalRaw 执行完整的全盘评估流程：
//   1. 四方向全盘棋型分析（缓存到 record）
//   2. 统计各棋型数量
//   3. 优先级链评分 + 位置权重
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

	// 第三步：五连立即检测（一方成五 → 游戏结束）
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

	// 两个冲四 ≈ 一个活四（双冲四几乎必胜）
	if e.count[white][sfour] >= 2 {
		e.count[white][four]++
	}
	if e.count[black][sfour] >= 2 {
		e.count[black][four]++
	}

	// 第四步：优先级链评分
	// 规则：先手的活四/冲四几乎必胜；活三在无对方反击时也是必胜。
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
		// turn == black — 镜像逻辑
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

	// 第五步：位置权重（中心高、边缘低）
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
//  四方向分析 — 将棋盘某条线提取到 line[] 后调用 analysisLine
// ============================================================

// analyzeHorizon 分析第 i 行，返回 (i,j) 处的水平方向棋型。
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

// analyzeVertical 分析第 j 列，返回 (i,j) 处的垂直方向棋型。
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

// analyzeLeft 分析主对角线方向 (↘)，返回 (i,j) 处的棋型。
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

// analyzeRight 分析反对角线方向 (↙)，返回 (i,j) 处的棋型。
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
//   4. 支持跳棋检测（X_XXX → SFOUR, XX_XX → SFOUR,  _X_XX → STHREE）
//
// 跳棋是五子棋的关键战术元素：
//   不走连续的连珠，而是中间留一个空格，
//   对手堵一端则从另一端补上形成四连/五连。
//
// ============================================================

func (e *boardEval) analysisLine(num, pos int) {
	// 填充哨兵值（0xf = 越界标记，与任何棋子值都不同）
	for i := num; i < 30; i++ {
		e.line[i] = 0xf
	}
	for i := 0; i < num; i++ {
		e.result[i] = todo
	}

	// 线段太短无法形成五连
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

	// 扩展到自由范围（遇对方棋子或边界止）
	leftRange := xl
	for leftRange > 0 && e.line[leftRange-1] != inverse {
		leftRange--
	}
	rightRange := xr
	for rightRange < num-1 && e.line[rightRange+1] != inverse {
		rightRange++
	}

	// 自由范围不足 5 格 → 无威胁
	if rightRange-leftRange < 4 {
		for k := leftRange; k <= rightRange; k++ {
			e.result[k] = analysed
		}
		return
	}

	// 标记连续段已分析
	for k := xl; k <= xr; k++ {
		e.result[k] = analysed
	}

	srange := xr - xl // 连续长度 - 1（即连续棋子数 - 1）

	// ── 五连 ──
	if srange >= 4 {
		e.result[pos] = five
		return
	}

	// ── 四连 ──
	if srange == 3 {
		leftFour := xl > 0 && e.line[xl-1] == 0
		if xr < num {
			if e.line[xr+1] == 0 {
				if leftFour {
					e.result[pos] = four // 两端全开 → 活四
				} else {
					e.result[pos] = sfour // 仅一端开 → 冲四
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

	// ── 三连 ──
	if srange == 2 {
		left3 := false
		if xl > 0 && e.line[xl-1] == 0 {
			// 跳四检测：X_XXX 模式
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
				// 跳四检测：XXX_X 模式
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

	// ── 二连 ──
	if srange == 1 {
		left2 := false
		if xl > 2 && e.line[xl-1] == 0 {
			if e.line[xl-2] == stone {
				if e.line[xl-3] == stone {
					// XX_X 模式 → 冲四
					e.result[xl-3] = analysed
					e.result[xl-2] = analysed
					e.result[xl] = sfour
				} else if e.line[xl-3] == 0 {
					// _X_X 模式 → 冲三
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
					// X_XX 模式 → 冲四
					e.result[xr+3] = analysed
					e.result[xr+2] = analysed
					e.result[xr] = sfour
				} else if e.line[xr+3] == 0 {
					// X_X_ 模式
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
					e.result[pos] = two // 两端开 → 活二
				} else {
					e.result[pos] = stwo // 一端开 → 冲二
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
