package game_engine

import (
	"fmt"
)

// 棋盘大小
const BoardSize = 15

// 棋子状态
type CellState int

// iota 是 Go 的"自动递增器"，只在 const 块内有效。每换一行，值自动 +1
const (
	Empty CellState = iota // iota = 0, Empty = 0
	Black                  // iota = 1, Black = 1
	White                  // iota = 2, White = 2
)

// 落子位置
type Move struct {
	Row int
	Col int
}

// 游戏引擎
type GameEngine struct {
	Board       [BoardSize][BoardSize]CellState
	CurrentTurn CellState
	MoveHistory []Move
	Winner      CellState
	GameOver    bool
}

// 创建新游戏：棋盘全空，黑棋先走，没历史，没胜者，没结束
func NewGame() *GameEngine {
	return &GameEngine{
		Board:       [BoardSize][BoardSize]CellState{},
		CurrentTurn: Black,
		MoveHistory: []Move{},
		Winner:      Empty,
		GameOver:    false,
	}
}

// 落子
func (g *GameEngine) MakeMove(row, col int) error {
	// Go 的 if 条件必须是布尔值，不能用整数 0/1 代替 true/false
	if g.GameOver {
		return fmt.Errorf("对局已结束")
	}
	// 前端已有判断，这里无需重复判断，但为了避免有神人绕过前端直接请求后端卡死程序，还是要留上这个保底机制
	if row < 0 || row >= BoardSize || col < 0 || col >= BoardSize {
		return fmt.Errorf("无效坐标: (%d, %d)", row, col)
	}
	// 如上
	if g.Board[row][col] != Empty {
		return fmt.Errorf("该位置已有棋子: (%d, %d)", row, col)
	}

	g.Board[row][col] = g.CurrentTurn
	// 这里把新的一步棋追加到历史记录末尾
	g.MoveHistory = append(g.MoveHistory, Move{Row: row, Col: col})

	// 检查胜负
	if g.CheckWin(row, col, g.CurrentTurn) {
		g.Winner = g.CurrentTurn
		g.GameOver = true
		return nil
	}

	// 检查平局，下了 225 步还没人赢，棋盘满了，平局
	if len(g.MoveHistory) == BoardSize*BoardSize {
		g.GameOver = true
		return nil
	}

	// 切换回合
	if g.CurrentTurn == Black {
		g.CurrentTurn = White
	} else {
		g.CurrentTurn = Black
	}

	return nil
}

// 检查胜负：我们不需要扫描整个棋盘，只需从最后一颗落子出发，沿四个方向（水平、垂直、两条对角线）分别向正反两侧数连续同色子的个数，任一个方向达到 5 颗就赢。
func (g *GameEngine) CheckWin(row, col int, player CellState) bool {
	// 如果有神人绕过前端传了 Empty（空位=0）进来检查，直接返回 false。没有棋子不可能构成五连。避免卡到后续程序崩溃
	if player == Empty {
		return false
	}
	directions := [][2]int{
		{0, 1},  // 水平
		{1, 0},  // 垂直
		{1, 1},  // 对角线
		{1, -1}, // 反对角线
	}
    
	// range 遍历切片，每次迭代返回两个值：索引, 元素。_ 是空白标识符——"我不要这个值"。这里用不到索引，所以用 _ 扔掉。
	for _, dir := range directions {
		count := 1
		// 正方向
		for i := 1; i < 5; i++ {
			r, c := row+dir[0]*i, col+dir[1]*i
			if r >= 0 && r < BoardSize && c >= 0 && c < BoardSize && g.Board[r][c] == player {
				count++
			} else {
				break
			}
		}
		// 反方向
		for i := 1; i < 5; i++ {
			r, c := row-dir[0]*i, col-dir[1]*i
			if r >= 0 && r < BoardSize && c >= 0 && c < BoardSize && g.Board[r][c] == player {
				count++
			} else {
				break
			}
		}
		if count >= 5 {
			return true
		}
	}
	return false
}

// GetCurrentPlayerInt 将 CurrentTurn 转为策略接口约定的 player 值 (1=黑, 2=白)
func (g *GameEngine) GetCurrentPlayerInt() int {
	if g.CurrentTurn == Black {
		return 1
	}
	return 2
}

// 获取棋盘状态（用于策略输入）
func (g *GameEngine) GetBoardState() [BoardSize][BoardSize]int {
	var state [BoardSize][BoardSize]int
	for i := 0; i < BoardSize; i++ {
		for j := 0; j < BoardSize; j++ {
			switch g.Board[i][j] {
			case Black:
				state[i][j] = 1
			case White:
				state[i][j] = 2
			default:
				state[i][j] = 0
			}
		}
	}
	return state
}

// 克隆游戏状态
func (g *GameEngine) Clone() *GameEngine {
	clone := &GameEngine{}
	clone.Board = g.Board
	clone.CurrentTurn = g.CurrentTurn
	clone.MoveHistory = make([]Move, len(g.MoveHistory))
	copy(clone.MoveHistory, g.MoveHistory)
	clone.Winner = g.Winner
	clone.GameOver = g.GameOver
	return clone
}
