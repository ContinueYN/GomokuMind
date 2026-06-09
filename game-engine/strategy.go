package game_engine

// 策略接口 - 所有策略必须实现此接口
type Strategy interface {
	// 获取策略名称
	Name() string

	// 根据当前棋盘状态推荐下一步落子
	// board: 棋盘状态 (1=黑棋, -1=白棋, 0=空位)
	// player: 当前玩家 (1=黑棋, -1=白棋)
	// 返回: 推荐的落子位置
	GetMove(board [BoardSize][BoardSize]int, player int) Move

	// 训练策略（部分策略无需训练）
	Train(episodes int) error
}
