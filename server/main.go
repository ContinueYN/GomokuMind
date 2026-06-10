package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	game_engine "gomokumind/game-engine"
	"gomokumind/strategies/heuristic"
	"gomokumind/strategies/mcts"
)

// ---- 类型定义 ----

type Piece int

const (
	Empty Piece = iota
	Black
	White
)

type GameStatus string

const (
	Playing  GameStatus = "playing"
	BlackWin GameStatus = "black_win"
	WhiteWin GameStatus = "white_win"
	Draw     GameStatus = "draw"
)

type AIType string

const (
	AIHeuristic AIType = "heuristic"
	AIMCTS      AIType = "mcts"
	AIQLearning AIType = "q-learning"
	AIPPO       AIType = "ppo"
)

type Move struct {
	Row int `json:"row"`
	Col int `json:"col"`
}

type Game struct {
	ID            string  `json:"id"`
	Board         [][]int `json:"board"`
	CurrentPlayer int     `json:"current_player"`
	Status        string  `json:"status"`
	MoveHistory   []Move  `json:"move_history"`
	BlackAI       string  `json:"black_ai"`
	WhiteAI       string  `json:"white_ai"`
	CreatedAt     string  `json:"created_at"`
	UpdatedAt     string  `json:"updated_at"`
}

type CreateGameRequest struct {
	BlackAI string `json:"black_ai"`
	WhiteAI string `json:"white_ai"`
}

type MoveRequest struct {
	Row int    `json:"row"`
	Col int    `json:"col"`
	AI  string `json:"ai,omitempty"`
}

type GameResponse struct {
	Game  *Game  `json:"game,omitempty"`
	Error string `json:"error,omitempty"`
}

type MoveResponse struct {
	Move Move  `json:"move"`
	Game *Game `json:"game"`
}

// ---- 游戏存储 ----

type GameStore struct {
	mu    sync.RWMutex
	games map[string]*game_engine.GameEngine
	infos map[string]*Game
}

var store = &GameStore{
	games: make(map[string]*game_engine.GameEngine),
	infos: make(map[string]*Game),
}

var idCounter int
var idMu sync.Mutex

func nextID() string {
	idMu.Lock()
	defer idMu.Unlock()
	idCounter++
	return fmt.Sprintf("game_%d_%d", time.Now().UnixMilli(), idCounter)
}

// ---- 策略工厂 ----

func getStrategy(aiType string) game_engine.Strategy {
	switch AIType(aiType) {
	case AIMCTS:
		return mcts.NewMCTSStrategy(100)
	case AIHeuristic:
		fallthrough
	default:
		return heuristic.NewHeuristicStrategy()
	}
}

func isGoStrategy(aiType string) bool {
	t := AIType(aiType)
	return t == AIHeuristic || t == AIMCTS
}

// ---- 工具函数 ----

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

func winnerToStatus(w game_engine.CellState) GameStatus {
	switch w {
	case game_engine.Black:
		return BlackWin
	case game_engine.White:
		return WhiteWin
	default:
		if w == game_engine.Empty {
			return Draw
		}
		return Draw
	}
}

func engineToGame(id string, eng *game_engine.GameEngine, blackAI, whiteAI string) *Game {
	status := Playing
	if eng.GameOver {
		status = winnerToStatus(eng.Winner)
	}

	history := make([]Move, len(eng.MoveHistory))
	for i, m := range eng.MoveHistory {
		history[i] = Move{Row: m.Row, Col: m.Col}
	}

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

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, GameResponse{Error: msg})
}

// ---- HTTP 处理器 ----

func healthHandler(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func createGameHandler(w http.ResponseWriter, r *http.Request) {
	var req CreateGameRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	id := nextID()
	now := time.Now().Format(time.RFC3339)

	eng := game_engine.NewGame()

	info := &Game{
		ID:            id,
		Board:         cellStateToPiece(eng.Board),
		CurrentPlayer: 1,
		Status:        string(Playing),
		MoveHistory:   []Move{},
		BlackAI:       req.BlackAI,
		WhiteAI:       req.WhiteAI,
		CreatedAt:     now,
		UpdatedAt:     now,
	}

	store.mu.Lock()
	store.games[id] = eng
	store.infos[id] = info
	store.mu.Unlock()

	writeJSON(w, http.StatusOK, GameResponse{Game: info})
}

func getGameHandler(w http.ResponseWriter, r *http.Request, id string) {
	store.mu.RLock()
	eng, ok := store.games[id]
	info, infoOk := store.infos[id]
	store.mu.RUnlock()

	if !ok || !infoOk {
		writeError(w, http.StatusNotFound, "game not found")
		return
	}

	updated := engineToGame(id, eng, info.BlackAI, info.WhiteAI)
	updated.CreatedAt = info.CreatedAt
	updated.UpdatedAt = time.Now().Format(time.RFC3339)

	writeJSON(w, http.StatusOK, GameResponse{Game: updated})
}

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

	if err := eng.MakeMove(req.Row, req.Col); err != nil {
		store.mu.Unlock()
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	updated := engineToGame(id, eng, info.BlackAI, info.WhiteAI)
	updated.CreatedAt = info.CreatedAt
	updated.UpdatedAt = time.Now().Format(time.RFC3339)
	info.UpdatedAt = updated.UpdatedAt
	store.mu.Unlock()

	writeJSON(w, http.StatusOK, GameResponse{Game: updated})
}

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

	// 确定当前应使用的 AI
	aiType := info.BlackAI
	if eng.CurrentTurn == game_engine.White {
		aiType = info.WhiteAI
	}

	if !isGoStrategy(aiType) {
		store.mu.Unlock()
		writeError(w, http.StatusBadRequest, fmt.Sprintf("strategy '%s' requires Python runtime, not available in Go server", aiType))
		return
	}

	strategy := getStrategy(aiType)
	boardState := eng.GetBoardState()

	player := 1
	if eng.CurrentTurn == game_engine.White {
		player = -1
	}

	move := strategy.GetMove(boardState, player)

	if err := eng.MakeMove(move.Row, move.Col); err != nil {
		store.mu.Unlock()
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	updated := engineToGame(id, eng, info.BlackAI, info.WhiteAI)
	updated.CreatedAt = info.CreatedAt
	updated.UpdatedAt = time.Now().Format(time.RFC3339)
	info.UpdatedAt = updated.UpdatedAt
	store.mu.Unlock()

	writeJSON(w, http.StatusOK, MoveResponse{
		Move: Move{Row: move.Row, Col: move.Col},
		Game: updated,
	})
}

func listGamesHandler(w http.ResponseWriter, r *http.Request) {
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

func deleteGameHandler(w http.ResponseWriter, r *http.Request, id string) {
	store.mu.Lock()
	delete(store.games, id)
	delete(store.infos, id)
	store.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]string{"message": "deleted"})
}

// ---- 路由 ----

func router(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/api")

	switch {
	case path == "/health":
		healthHandler(w, r)

	case path == "/gomoku" && r.Method == "POST":
		createGameHandler(w, r)

	case path == "/gomoku" && r.Method == "GET":
		listGamesHandler(w, r)

	case strings.HasPrefix(path, "/gomoku/") && strings.HasSuffix(path, "/ai-move") && r.Method == "POST":
		id := strings.TrimSuffix(strings.TrimPrefix(path, "/gomoku/"), "/ai-move")
		aiMoveHandler(w, r, id)

	case strings.HasPrefix(path, "/gomoku/") && r.Method == "PUT":
		id := strings.TrimPrefix(path, "/gomoku/")
		makeMoveHandler(w, r, id)

	case strings.HasPrefix(path, "/gomoku/") && r.Method == "DELETE":
		id := strings.TrimPrefix(path, "/gomoku/")
		deleteGameHandler(w, r, id)

	case strings.HasPrefix(path, "/gomoku/") && r.Method == "GET":
		id := strings.TrimPrefix(path, "/gomoku/")
		getGameHandler(w, r, id)

	default:
		writeError(w, http.StatusNotFound, "not found")
	}
}

func main() {
	addr := ":8080"
	fmt.Printf("GomokuMind server starting on %s\n", addr)
	log.Fatal(http.ListenAndServe(addr, http.HandlerFunc(router)))
}
