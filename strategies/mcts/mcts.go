package mcts

import (
	"math"
	"math/rand"
	"time"

	game_engine "gomokumind/game-engine"
)

// MCTS节点
type MCTSNode struct {
	move     game_engine.Move
	state    *game_engine.GameEngine
	parent   *MCTSNode
	children []*MCTSNode
	wins     float64
	visits   int
	player   int // 该节点对应的玩家
}

// MCTS策略
type MCTSStrategy struct {
	simulations int
	c           float64 // 探索常数
}

func NewMCTSStrategy(simulations int) *MCTSStrategy {
	return &MCTSStrategy{
		simulations: simulations,
		c:           1.41, // sqrt(2)
	}
}

func (m *MCTSStrategy) Name() string {
	return "蒙特卡洛树搜索(MCTS)"
}

func (m *MCTSStrategy) GetMove(board [game_engine.BoardSize][game_engine.BoardSize]int, player int) game_engine.Move {
	rand.Seed(time.Now().UnixNano())

	// 基于当前board创建根节点
	root := &MCTSNode{
		state:  m.boardToGame(board),
		parent: nil,
		player: player,
	}

	// MCTS主循环
	for i := 0; i < m.simulations; i++ {
		node := m.selectNode(root)
		winner := m.simulate(node.state)
		m.backpropagate(node, winner, player)
	}

	// 选择访问次数最多的子节点
	return m.bestChild(root).move
}

func (m *MCTSStrategy) Train(episodes int) error {
	// MCTS无需预训练
	return nil
}

// 选择节点（UCT算法）
func (m *MCTSStrategy) selectNode(node *MCTSNode) *MCTSNode {
	for len(node.children) > 0 && !node.state.GameOver {
		// 扩展未完全展开的节点
		if len(node.children) < len(node.state.GetValidMoves()) {
			return m.expand(node)
		}

		// 选择UCT值最大的子节点
		bestUCT := -math.MaxFloat64
		bestChild := node.children[0]

		for _, child := range node.children {
			uct := m.calculateUCT(child)
			if uct > bestUCT {
				bestUCT = uct
				bestChild = child
			}
		}
		node = bestChild
	}

	return node
}

// 计算UCT值
func (m *MCTSStrategy) calculateUCT(node *MCTSNode) float64 {
	if node.visits == 0 {
		return math.MaxFloat64
	}
	exploitation := node.wins / float64(node.visits)
	exploration := m.c * math.Sqrt(math.Log(float64(node.parent.visits))/float64(node.visits))
	return exploitation + exploration
}

// 扩展节点
func (m *MCTSStrategy) expand(node *MCTSNode) *MCTSNode {
	tried := make(map[string]bool)
	for _, child := range node.children {
		key := moveKey(child.move)
		tried[key] = true
	}

	moves := node.state.GetValidMoves()
	for _, move := range moves {
		if !tried[moveKey(move)] {
			newState := node.state.Clone()
			newState.MakeMove(move.Row, move.Col)

			child := &MCTSNode{
				move:   move,
				state:  newState,
				parent: node,
				player: node.player,
			}
			node.children = append(node.children, child)
			return child
		}
	}

	return node
}

// 模拟（随机对弈到结束）
func (m *MCTSStrategy) simulate(state *game_engine.GameEngine) int {
	simState := state.Clone()
	for !simState.GameOver {
		moves := simState.GetValidMoves()
		if len(moves) == 0 {
			break
		}
		move := moves[rand.Intn(len(moves))]
		simState.MakeMove(move.Row, move.Col)
	}

	if simState.Winner == game_engine.Empty {
		return 0 // 平局
	}
	if simState.Winner == game_engine.Black {
		return 1
	}
	return -1
}

// 反向传播
func (m *MCTSStrategy) backpropagate(node *MCTSNode, winner int, originalPlayer int) {
	for node != nil {
		node.visits++

		// 计算得分
		score := 0.0
		if winner == 0 {
			score = 0.5 // 平局
		} else if (originalPlayer == 1 && winner == 1) || (originalPlayer == -1 && winner == -1) {
			score = 1.0
		}
		node.wins += score

		node = node.parent
	}
}

// 选择最佳子节点
func (m *MCTSStrategy) bestChild(node *MCTSNode) *MCTSNode {
	if len(node.children) == 0 {
		// 如果没有子节点，返回随机合法移动
		moves := node.state.GetValidMoves()
		if len(moves) > 0 {
			return &MCTSNode{move: moves[rand.Intn(len(moves))]}
		}
		return node
	}

	bestVisits := -1
	bestChild := node.children[0]

	for _, child := range node.children {
		if child.visits > bestVisits {
			bestVisits = child.visits
			bestChild = child
		}
	}

	return bestChild
}

// 将board转换为GameEngine
func (m *MCTSStrategy) boardToGame(board [game_engine.BoardSize][game_engine.BoardSize]int) *game_engine.GameEngine {
	game := game_engine.NewGame()
	moveCount := 0

	for i := 0; i < game_engine.BoardSize; i++ {
		for j := 0; j < game_engine.BoardSize; j++ {
			if board[i][j] != 0 {
				player := game_engine.Black
				if board[i][j] == -1 {
					player = game_engine.White
				}
				game.Board[i][j] = player
				game.MoveHistory = append(game.MoveHistory, game_engine.Move{Row: i, Col: j})
				moveCount++
			}
		}
	}

	// 根据落子数判断当前玩家
	if moveCount%2 == 0 {
		game.CurrentTurn = game_engine.Black
	} else {
		game.CurrentTurn = game_engine.White
	}

	return game
}

func moveKey(move game_engine.Move) string {
	return string(rune(move.Row*game_engine.BoardSize + move.Col))
}
