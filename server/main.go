package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"time"

	game_engine "gomokumind/game-engine"
	"gomokumind/strategies/alphabeta"
	"gomokumind/strategies/alphazero"
	"gomokumind/strategies/heuristic"
	"gomokumind/strategies/mcts"
)

// ============================================================
//  类型定义 — 用于与服务端外部（前端 / Python 策略）交换数据
// ============================================================

// GameStatus 游戏状态枚举（字符串形式，便于前端直接展示）
type GameStatus string

// 理论上可以直接用 string，但是可以"约束"编译器 — 让编译器帮你确保正确的人传正确的值。
const (
	Playing  GameStatus = "playing"   // 对局进行中
	BlackWin GameStatus = "black_win" // 黑棋胜
	WhiteWin GameStatus = "white_win" // 白棋胜
	Draw     GameStatus = "draw"      // 平局（棋盘满）
)

// AIType AI 策略标识，对应策略工厂的分发键
type AIType string

const (
	AIHeuristic AIType = "heuristic" // 启发式棋型评估（Go 原生）
	AIMCTS      AIType = "mcts"      // 蒙特卡洛树搜索（Go 原生）
	AIAlphaBeta AIType = "alphabeta" // Alpha-Beta 增强搜索（Go 原生）
	AIAlphaZero AIType = "alphazero" // AlphaZero 深度学习（Go + Python）
)

// Move 一步落子（行列坐标），JSON 序列化字段名与前端对齐
type Move struct {
	Row int `json:"row"`
	Col int `json:"col"`
}

// Game 对外展示的游戏视图，从 engine 转换而来
type Game struct {
	ID            string  `json:"id"`
	Board         [][]int `json:"board"`          // 15×15 二维数组：0=空 1=黑 2=白
	CurrentPlayer int     `json:"current_player"` // 1=黑棋回合 2=白棋回合
	Status        string  `json:"status"`         // playing / black_win / white_win / draw
	MoveHistory   []Move  `json:"move_history"`   // 落子历史
	BlackAI       string  `json:"black_ai"`       // 黑方 AI 类型（"" 表示人类）
	WhiteAI       string  `json:"white_ai"`       // 白方 AI 类型（"" 表示人类）
	CreatedAt     string  `json:"created_at"`     // 创建时间 RFC3339
	UpdatedAt     string  `json:"updated_at"`     // 最后修改时间 RFC3339
}

// ---- 请求 / 响应体 ----

// CreateGameRequest 前端创建游戏时传入的 JSON，omitempty - 如果字段为空，JSON序列化时忽略该字段
type CreateGameRequest struct {
	BlackAI string `json:"black_ai,omitempty"` // "" = 人类
	WhiteAI string `json:"white_ai,omitempty"` // "" = 人类
}

// MoveRequest 前端落子或 AI 代下时传入的 JSON
type MoveRequest struct {
	Row int    `json:"row"`
	Col int    `json:"col"`
	AI  string `json:"ai,omitempty"` // 仅 AI 代下时携带策略类型
}

// GameResponse 大多数接口的统一返回格式
type GameResponse struct {
	Game  *Game  `json:"game,omitempty"`
	Error string `json:"error,omitempty"`
}

// MoveResponse AI 落子接口的返回格式，比 GameResponse 多一个 move
type MoveResponse struct {
	Move Move  `json:"move"`
	Game *Game `json:"game"`
}

// ============================================================
//  游戏存储 — 内存 map，服务重启即丢失（适合演示/调试）
// ============================================================

// GameStore 线程安全的游戏存储：持有引擎实例 + 展示信息
type GameStore struct {
	mu    sync.RWMutex
	games map[string]*game_engine.GameEngine // 引擎实例，真实数据源
	infos map[string]*Game                   // 展示用快照（缓存 AI 类型等字段）
}

var store = &GameStore{
	games: make(map[string]*game_engine.GameEngine),
	infos: make(map[string]*Game),
}

// recordStore 持久化对局胜负记录（JSON 文件存储）
var recordStore *RecordStore

// 自增 ID 需要专用锁，避免与 store 锁竞争
var idCounter int
var idMu sync.Mutex

// nextID 生成唯一游戏 ID：game_<毫秒时间戳>_<自增序号>
func nextID() string {
	idMu.Lock()
	defer idMu.Unlock()
	idCounter++
	return fmt.Sprintf("game_%d_%d", time.Now().UnixMilli(), idCounter)
}

// ============================================================
//  策略工厂
// ============================================================

// azStrategy AlphaZero 策略单例，服务器启动时初始化一次
var azStrategy *alphazero.AlphaZeroStrategy

// getStrategy 根据 aiType 返回对应的策略实现
func getStrategy(aiType string) game_engine.Strategy {
	switch AIType(aiType) {
	case AIMCTS:
		return mcts.NewMCTSStrategy(400)
	case AIAlphaBeta:
		return alphabeta.NewAlphaBetaStrategy(4)
	case AIAlphaZero:
		if azStrategy != nil {
			return azStrategy
		}
		return heuristic.NewHeuristicStrategy()
	default:
		return heuristic.NewHeuristicStrategy()
	}
}

// ============================================================
//  工具函数 — engine 内部类型 ⇄ 对外 JSON 类型 的转换
// ============================================================

// cellStateToPiece 将引擎的棋盘（固定数组）转为前端用的二维切片
// 0 = 空  1 = 黑  2 = 白
func cellStateToPiece(b [game_engine.BoardSize][game_engine.BoardSize]game_engine.CellState) [][]int {
	result := make([][]int, game_engine.BoardSize)
	for i := 0; i < game_engine.BoardSize; i++ {
		result[i] = make([]int, game_engine.BoardSize)
		for j := 0; j < game_engine.BoardSize; j++ {
			result[i][j] = int(b[i][j])
		}
	}
	return result
}

// winnerToStatus 将引擎的 CellState 胜负值转为字符串状态
func winnerToStatus(w game_engine.CellState) GameStatus {
	switch w {
	case game_engine.Black:
		return BlackWin
	case game_engine.White:
		return WhiteWin
	default: // game_engine.Empty → 平局
		return Draw
	}
}

// engineToGame 将引擎的内部状态转换为对外展示的 Game 结构
// 注意：不设置 CreatedAt / UpdatedAt，由调用方决定
func engineToGame(id string, eng *game_engine.GameEngine, blackAI, whiteAI string) *Game {
	// 根据引擎的 GameOver / Winner 确定展示状态
	status := Playing
	if eng.GameOver {
		status = winnerToStatus(eng.Winner)
	}

	// 落子历史转换
	history := make([]Move, len(eng.MoveHistory))
	for i, m := range eng.MoveHistory {
		history[i] = Move{Row: m.Row, Col: m.Col}
	}

	// CurrentPlayer: engine 的 Black=1, White=2 → 前端期望 1/2
	player := 1
	if eng.CurrentTurn == game_engine.White {
		player = 2
	}

	return &Game{
		ID:            id,
		Board:         cellStateToPiece(eng.Board),
		CurrentPlayer: player,
		Status:        string(status),
		MoveHistory:   history,
		BlackAI:       blackAI,
		WhiteAI:       whiteAI,
	}
}

// writeJSON 统一写入 JSON 响应
func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

// writeError 快捷写入错误响应
func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, GameResponse{Error: msg})
}

// ============================================================
//  HTTP 处理器
// ============================================================

// healthHandler 健康检查，返回 {"status":"ok"}
func healthHandler(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// createGameHandler 新建游戏并返回初始状态
func createGameHandler(w http.ResponseWriter, r *http.Request) {
	var req CreateGameRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	id := nextID()
	now := time.Now().Format(time.RFC3339)

	// 创建引擎实例（Board=全空, CurrentTurn=黑棋先手）
	eng := game_engine.NewGame()

	// 构建展示对象
	info := &Game{
		ID:            id,
		Board:         cellStateToPiece(eng.Board),
		CurrentPlayer: 1, // 黑棋先手
		Status:        string(Playing),
		MoveHistory:   []Move{},
		BlackAI:       req.BlackAI,
		WhiteAI:       req.WhiteAI,
		CreatedAt:     now,
		UpdatedAt:     now,
	}

	// 写入存储
	store.mu.Lock()
	store.games[id] = eng
	store.infos[id] = info
	store.mu.Unlock()

	writeJSON(w, http.StatusOK, GameResponse{Game: info})
}

// getGameHandler 获取指定游戏的最新状态
func getGameHandler(w http.ResponseWriter, _ *http.Request, id string) {
	store.mu.RLock()
	eng, ok := store.games[id]
	info, infoOk := store.infos[id]
	store.mu.RUnlock()

	if !ok || !infoOk {
		writeError(w, http.StatusNotFound, "game not found")
		return
	}

	// 从引擎实时构建返回对象
	updated := engineToGame(id, eng, info.BlackAI, info.WhiteAI)
	updated.CreatedAt = info.CreatedAt
	updated.UpdatedAt = time.Now().Format(time.RFC3339)

	writeJSON(w, http.StatusOK, GameResponse{Game: updated})
}

// makeMoveHandler 处理人类玩家的落子请求
func makeMoveHandler(w http.ResponseWriter, r *http.Request, id string) {
	var req MoveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	store.mu.Lock()
	eng, ok := store.games[id]
	info, infoOk := store.infos[id]
	if !ok || !infoOk {
		store.mu.Unlock()
		writeError(w, http.StatusNotFound, "game not found")
		return
	}

	if eng.GameOver {
		store.mu.Unlock()
		writeError(w, http.StatusBadRequest, "game is over")
		return
	}

	// 调用引擎落子（内部完成胜负判定 + 回合切换）
	if err := eng.MakeMove(req.Row, req.Col); err != nil {
		store.mu.Unlock()
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	// 落子成功后刷新展示对象
	updated := engineToGame(id, eng, info.BlackAI, info.WhiteAI)
	updated.CreatedAt = info.CreatedAt
	updated.UpdatedAt = time.Now().Format(time.RFC3339)
	info.UpdatedAt = updated.UpdatedAt

	// 游戏结束时持久化记录
	gameEnded := eng.GameOver
	blackAI, whiteAI, winner, moveCount := info.BlackAI, info.WhiteAI, eng.Winner, len(eng.MoveHistory)
	store.mu.Unlock()

	if gameEnded && recordStore != nil {
		recordStore.Add(GameRecord{
			ID:         id,
			BlackAI:    aiLabel(blackAI),
			WhiteAI:    aiLabel(whiteAI),
			Status:     string(winnerToStatus(winner)),
			Winner:     winnerLabel(winner),
			MoveCount:  moveCount,
			CreatedAt:  info.CreatedAt,
			FinishedAt: time.Now().Format(time.RFC3339),
		})
	}

	writeJSON(w, http.StatusOK, GameResponse{Game: updated})
}

// aiMoveHandler 处理 AI 落子请求（自动计算出最优棋步并落子）
//
// 注意：当前实现中，策略计算（包括 Python 子进程调用）在写锁内执行，
// 意味着 AI 思考期间整个服务器的所有游戏操作都会被阻塞。
// 生产环境应将策略计算移到锁外，仅落子时加锁。
func aiMoveHandler(w http.ResponseWriter, r *http.Request, id string) {
	store.mu.Lock()
	eng, ok := store.games[id]
	info, infoOk := store.infos[id]
	if !ok || !infoOk {
		store.mu.Unlock()
		writeError(w, http.StatusNotFound, "game not found")
		return
	}

	if eng.GameOver {
		store.mu.Unlock()
		writeError(w, http.StatusBadRequest, "game is over")
		return
	}

	// 确定当前应使用的 AI 类型：
	//   1. 优先从游戏创建时的配置读取（info.BlackAI / info.WhiteAI）
	//   2. 若为空（如人类落子后请求 AI 代下），从请求体读取 ai 字段
	//   3. 仍然为空则默认 heuristic
	aiType := info.BlackAI
	if eng.CurrentTurn == game_engine.White {
		aiType = info.WhiteAI
	}
	if aiType == "" {
		var body MoveRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err == nil && body.AI != "" {
			aiType = body.AI
		} else {
			aiType = string(AIHeuristic)
		}
	}

	// 统一使用 Go 策略接口（heuristic / mcts / alphabeta / alphazero）
	strategy := getStrategy(aiType)
	boardState := eng.GetBoardState() // 1=黑 -1=白 0=空

	// engine 的 CurrentTurn 转为策略接口期望的 player 值
	p := 1 // 黑棋
	if eng.CurrentTurn == game_engine.White {
		p = -1 // 白棋
	}

	move := strategy.GetMove(boardState, p)

	if err := eng.MakeMove(move.Row, move.Col); err != nil {
		store.mu.Unlock()
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	updated := engineToGame(id, eng, info.BlackAI, info.WhiteAI)
	updated.CreatedAt = info.CreatedAt
	updated.UpdatedAt = time.Now().Format(time.RFC3339)
	info.UpdatedAt = updated.UpdatedAt

	// 游戏结束时持久化记录
	gameEnded := eng.GameOver
	blackAI, whiteAI, winner, moveCount := info.BlackAI, info.WhiteAI, eng.Winner, len(eng.MoveHistory)
	store.mu.Unlock()

	if gameEnded && recordStore != nil {
		recordStore.Add(GameRecord{
			ID:         id,
			BlackAI:    aiLabel(blackAI),
			WhiteAI:    aiLabel(whiteAI),
			Status:     string(winnerToStatus(winner)),
			Winner:     winnerLabel(winner),
			MoveCount:  moveCount,
			CreatedAt:  info.CreatedAt,
			FinishedAt: time.Now().Format(time.RFC3339),
		})
	}

	writeJSON(w, http.StatusOK, MoveResponse{
		Move: Move{Row: move.Row, Col: move.Col},
		Game: updated,
	})
}

// listGamesHandler 列出所有游戏（不含历史记录，仅摘要）
func listGamesHandler(w http.ResponseWriter, _ *http.Request) {
	store.mu.RLock()
	defer store.mu.RUnlock()

	games := make([]*Game, 0, len(store.games))
	for id, eng := range store.games {
		info := store.infos[id]
		g := engineToGame(id, eng, info.BlackAI, info.WhiteAI)
		g.CreatedAt = info.CreatedAt
		g.UpdatedAt = info.UpdatedAt
		games = append(games, g)
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{"games": games})
}

// deleteGameHandler 删除指定游戏（从内存中移除引擎和展示数据）
func deleteGameHandler(w http.ResponseWriter, _ *http.Request, id string) {
	store.mu.Lock()
	delete(store.games, id)
	delete(store.infos, id)
	store.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]string{"message": "deleted"})
}

// recordsHandler 返回对局历史记录，支持 ?limit=N（默认 50，最大 200）
func recordsHandler(w http.ResponseWriter, r *http.Request) {
	if recordStore == nil {
		writeJSON(w, http.StatusOK, map[string]interface{}{"records": []GameRecord{}})
		return
	}
	limit := 50
	if l := r.URL.Query().Get("limit"); l != "" {
		if n, err := fmt.Sscanf(l, "%d", &limit); err == nil && n == 1 {
			if limit < 1 {
				limit = 1
			}
			if limit > 200 {
				limit = 200
			}
		}
	}
	all := recordStore.Records()
	if limit > len(all) {
		limit = len(all)
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"records": all[:limit]})
}

// statsHandler 返回聚合统计 + 最近对局记录
func statsHandler(w http.ResponseWriter, _ *http.Request) {
	if recordStore == nil {
		writeJSON(w, http.StatusOK, StatsResponse{})
		return
	}
	writeJSON(w, http.StatusOK, recordStore.Stats())
}

// ============================================================
//  路由 — 基于路径和方法的简单分发
// ============================================================

// router 是唯一的 HTTP 入口，完成 CORS 设置 + 路径路由
func router(w http.ResponseWriter, r *http.Request) {
	// 允许跨域（开发阶段对所有来源开放）生产环境要指定具体域名
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// 浏览器预检请求直接返回
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// 去掉 /api 前缀后匹配路由
	path := strings.TrimPrefix(r.URL.Path, "/api")

	switch {
	case path == "/health":
		healthHandler(w, r)

	// GET /api/stats — 对局统计
	case path == "/stats" && r.Method == "GET":
		statsHandler(w, r)

	// GET /api/records — 对局历史
	case path == "/records" && r.Method == "GET":
		recordsHandler(w, r)

	// POST /api/gomoku — 创建游戏
	case path == "/gomoku" && r.Method == "POST":
		createGameHandler(w, r)

	// GET /api/gomoku — 列出所有游戏
	case path == "/gomoku" && r.Method == "GET":
		listGamesHandler(w, r)

	// POST /api/gomoku/:id/ai-move — AI 落子
	case strings.HasPrefix(path, "/gomoku/") && strings.HasSuffix(path, "/ai-move") && r.Method == "POST":
		id := strings.TrimSuffix(strings.TrimPrefix(path, "/gomoku/"), "/ai-move")
		aiMoveHandler(w, r, id)

	// PUT /api/gomoku/:id — 人类落子
	case strings.HasPrefix(path, "/gomoku/") && r.Method == "PUT":
		id := strings.TrimPrefix(path, "/gomoku/")
		makeMoveHandler(w, r, id)

	// DELETE /api/gomoku/:id — 删除游戏
	case strings.HasPrefix(path, "/gomoku/") && r.Method == "DELETE":
		id := strings.TrimPrefix(path, "/gomoku/")
		deleteGameHandler(w, r, id)

	// GET /api/gomoku/:id — 获取游戏状态
	case strings.HasPrefix(path, "/gomoku/") && r.Method == "GET":
		id := strings.TrimPrefix(path, "/gomoku/")
		getGameHandler(w, r, id)

	default:
		writeError(w, http.StatusNotFound, "not found")
	}
}

// main 启动 HTTP 服务器监听 8080 端口
func main() {
	// 初始化 AlphaZero 策略（持久 Python 推理进程）
	modelPath := filepath.Join("strategies", "alphazero", "checkpoint_11.pth.tar")
	var err error
	azStrategy, err = alphazero.NewAlphaZeroStrategy(modelPath)
	if err != nil {
		log.Printf("WARNING: AlphaZero strategy failed to initialize: %v", err)
		log.Println("AlphaZero will not be available; falling back to heuristic.")
	} else {
		defer azStrategy.Close()
		log.Println("AlphaZero strategy initialized successfully.")
	}

	// 初始化对局记录存储（JSON 文件持久化）
	recordStore, err = NewRecordStore(filepath.Join(".", "records.json"))
	if err != nil {
		log.Printf("WARNING: RecordStore failed to initialize: %v", err)
		log.Println("Game records will not be persisted.")
	} else {
		log.Printf("RecordStore ready, %d historical records loaded.", len(recordStore.Stats().RecentRecords))
	}

	addr := ":8080"
	fmt.Printf("GomokuMind server starting on %s\n", addr)
	log.Fatal(http.ListenAndServe(addr, http.HandlerFunc(router)))
}
