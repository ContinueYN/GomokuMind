package main

import (
	"fmt"
	"math/rand"
	"time"

	game_engine "gomokumind/game-engine"
	"gomokumind/strategies/heuristic"
	"gomokumind/strategies/mcts"
)

// 对战结果
type MatchResult struct {
	Strategy1 string
	Strategy2 string
	Wins1     int
	Wins2     int
	Draws     int
	Total     int
}

// 运行两个策略之间的对战
func RunMatch(strategy1, strategy2 game_engine.Strategy, games int) MatchResult {
	result := MatchResult{
		Strategy1: strategy1.Name(),
		Strategy2: strategy2.Name(),
		Total:     games,
	}

	for i := 0; i < games; i++ {
		game := game_engine.NewGame()
		turn := 0 // 0=黑棋, 1=白棋

		for !game.GameOver {
			var move game_engine.Move
			board := game.GetBoardState()

			if turn == 0 {
				// 黑棋使用strategy1
				move = strategy1.GetMove(board, 1)
			} else {
				// 白棋使用strategy2
				move = strategy2.GetMove(board, -1)
			}

			game.MakeMove(move.Row, move.Col)
			turn = 1 - turn
		}

		if game.Winner == game_engine.Black {
			if turn == 1 {
				result.Wins1++
			} else {
				result.Wins2++
			}
		} else if game.Winner == game_engine.White {
			if turn == 0 {
				result.Wins2++
			} else {
				result.Wins1++
			}
		} else {
			result.Draws++
		}
	}

	return result
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("========================================")
	fmt.Println("     五子棋策略对比评估系统")
	fmt.Println("========================================")
	fmt.Println()

	// 初始化策略
	fmt.Println("初始化策略...")
	heuristic := heuristic.NewHeuristicStrategy()
	mcts := mcts.NewMCTSStrategy(100) // 100次模拟

	fmt.Printf("策略1: %s\n", heuristic.Name())
	fmt.Printf("策略2: %s\n", mcts.Name())
	fmt.Println()

	// 运行对战
	games := 10
	fmt.Printf("开始对战，每方%d局...\n", games)

	result := RunMatch(heuristic, mcts, games)

	// 输出结果
	fmt.Println()
	fmt.Println("========================================")
	fmt.Println("           对战结果")
	fmt.Println("========================================")
	fmt.Printf("%s 胜: %d\n", result.Strategy1, result.Wins1)
	fmt.Printf("%s 胜: %d\n", result.Strategy2, result.Wins2)
	fmt.Printf("平局: %d\n", result.Draws)
	fmt.Printf("总局数: %d\n", result.Total)
	fmt.Println()

	if result.Wins1 > result.Wins2 {
		fmt.Printf("🏆 %s 获胜！\n", result.Strategy1)
	} else if result.Wins2 > result.Wins1 {
		fmt.Printf("🏆 %s 获胜！\n", result.Strategy2)
	} else {
		fmt.Println("🤝 平局！")
	}
	fmt.Println("========================================")
}
