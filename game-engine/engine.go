package game_engine

import (
	"fmt"
	"strings"
)

// 棋盘大小
const BoardSize = 15

// 棋子状态
type CellState int

const (
	Empty CellState = iota
	Black
	White
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

// 创建新游戏
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
	if g.GameOver {
		return fmt.Errorf("game is over")
	}
	if row < 0 || row >= BoardSize || col < 0 || col >= BoardSize {
		return fmt.Errorf("invalid position: (%d, %d)", row, col)
	}
	if g.Board[row][col] != Empty {
		return fmt.Errorf("position occupied: (%d, %d)", row, col)
	}

	g.Board[row][col] = g.CurrentTurn
	g.MoveHistory = append(g.MoveHistory, Move{Row: row, Col: col})

	// 检查胜负
	if g.CheckWin(row, col, g.CurrentTurn) {
		g.Winner = g.CurrentTurn
		g.GameOver = true
		return nil
	}

	// 检查平局
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

// 检查胜负
func (g *GameEngine) CheckWin(row, col int, player CellState) bool {
	directions := [][2]int{
		{0, 1},  // 水平
		{1, 0},  // 垂直
		{1, 1},  // 对角线
		{1, -1}, // 反对角线
	}

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

// 获取合法落子位置
func (g *GameEngine) GetValidMoves() []Move {
	var moves []Move
	for i := 0; i < BoardSize; i++ {
		for j := 0; j < BoardSize; j++ {
			if g.Board[i][j] == Empty {
				moves = append(moves, Move{Row: i, Col: j})
			}
		}
	}
	return moves
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
				state[i][j] = -1
			default:
				state[i][j] = 0
			}
		}
	}
	return state
}

// 克隆游戏状态
func (g *GameEngine) Clone() *GameEngine {
	clone := NewGame()
	clone.Board = g.Board
	clone.CurrentTurn = g.CurrentTurn
	clone.MoveHistory = make([]Move, len(g.MoveHistory))
	copy(clone.MoveHistory, g.MoveHistory)
	clone.Winner = g.Winner
	clone.GameOver = g.GameOver
	return clone
}

// 打印棋盘
func (g *GameEngine) PrintBoard() string {
	var sb strings.Builder
	sb.WriteString("   ")
	for j := 0; j < BoardSize; j++ {
		sb.WriteString(fmt.Sprintf("%2d ", j))
	}
	sb.WriteString("\n")

	for i := 0; i < BoardSize; i++ {
		sb.WriteString(fmt.Sprintf("%2d ", i))
		for j := 0; j < BoardSize; j++ {
			switch g.Board[i][j] {
			case Black:
				sb.WriteString(" X ")
			case White:
				sb.WriteString(" O ")
			default:
				sb.WriteString(" . ")
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}
