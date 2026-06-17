package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	game_engine "gomokumind/game-engine"
)

// ============================================================
//  RecordStore — JSON 文件持久化的对局胜负记录存储
// ============================================================

// GameRecord 一局完成后的胜负记录，持久化到 records.json。
type GameRecord struct {
	ID         string `json:"id"`
	BlackAI    string `json:"black_ai"`
	WhiteAI    string `json:"white_ai"`
	Status     string `json:"status"` // "black_win" | "white_win" | "draw"
	Winner     string `json:"winner"` // "black" | "white" | "draw"
	MoveCount  int    `json:"move_count"`
	CreatedAt  string `json:"created_at"`
	FinishedAt string `json:"finished_at"`
}

// AIStat 单个 AI 类型的聚合统计。
type AIStat struct {
	AIType  string  `json:"ai_type"`
	Total   int     `json:"total"`
	Wins    int     `json:"wins"`
	Losses  int     `json:"losses"`
	WinRate float64 `json:"win_rate"`
}

// StatsResponse 对局统计响应体。
type StatsResponse struct {
	TotalGames    int          `json:"total_games"`
	BlackWins     int          `json:"black_wins"`
	WhiteWins     int          `json:"white_wins"`
	Draws         int          `json:"draws"`
	AvgMoves      float64      `json:"avg_moves"`
	ByAI          []AIStat     `json:"by_ai"`
	RecentRecords []GameRecord `json:"recent_records"`
}

// RecordStore 线程安全的对局记录存储，内存索引 + JSON 文件持久化。
type RecordStore struct {
	mu       sync.RWMutex
	records  []GameRecord
	filePath string
}

// NewRecordStore 加载或初始化记录存储。
func NewRecordStore(filePath string) (*RecordStore, error) {
	rs := &RecordStore{
		filePath: filePath,
		records:  make([]GameRecord, 0),
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("[RecordStore] No existing records file at %s, starting fresh.", filePath)
			return rs, nil
		}
		return nil, fmt.Errorf("read records file: %w", err)
	}

	if len(data) > 0 {
		if err := json.Unmarshal(data, &rs.records); err != nil {
			return nil, fmt.Errorf("parse records file: %w", err)
		}
	}

	log.Printf("[RecordStore] Loaded %d records from %s", len(rs.records), filePath)
	return rs, nil
}

// Add 追加一条记录并即时写入磁盘。
func (rs *RecordStore) Add(record GameRecord) error {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	rs.records = append(rs.records, record)
	if err := rs.save(); err != nil {
		return fmt.Errorf("save records: %w", err)
	}

	log.Printf("[RecordStore] Saved record %s: %s vs %s → %s (%d moves)",
		record.ID, record.BlackAI, record.WhiteAI, record.Winner, record.MoveCount)
	return nil
}

// Records 返回全部记录副本（按 finished_at 降序）。
func (rs *RecordStore) Records() []GameRecord {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	result := make([]GameRecord, len(rs.records))
	copy(result, rs.records)
	// 倒序：最新在前
	for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
		result[i], result[j] = result[j], result[i]
	}
	return result
}

// Stats 计算聚合统计。
func (rs *RecordStore) Stats() StatsResponse {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	stats := StatsResponse{
		TotalGames: len(rs.records),
	}

	var totalMoves int
	aiStats := make(map[string]*AIStat) // keyed by AI type

	for _, r := range rs.records {
		// 胜负统计
		switch r.Winner {
		case "black":
			stats.BlackWins++
		case "white":
			stats.WhiteWins++
		case "draw":
			stats.Draws++
		}
		totalMoves += r.MoveCount

		// 按 AI 类型聚合（区分黑白两侧）
		checkAISide := func(ai string, isBlack bool) {
			if ai == "" || ai == "human" {
				return
			}
			if _, ok := aiStats[ai]; !ok {
				aiStats[ai] = &AIStat{AIType: ai}
			}
			aiStats[ai].Total++
			if (isBlack && r.Winner == "black") || (!isBlack && r.Winner == "white") {
				aiStats[ai].Wins++
			} else if (isBlack && r.Winner == "white") || (!isBlack && r.Winner == "black") {
				aiStats[ai].Losses++
			}
			// draw: neither win nor loss
		}
		checkAISide(r.BlackAI, true)
		checkAISide(r.WhiteAI, false)
	}

	if stats.TotalGames > 0 {
		stats.AvgMoves = float64(totalMoves) / float64(stats.TotalGames)
	}

	// 构建 AI 统计切片
	stats.ByAI = make([]AIStat, 0, len(aiStats))
	for _, s := range aiStats {
		if s.Total > 0 {
			s.WinRate = float64(s.Wins) / float64(s.Total) * 100
		}
		stats.ByAI = append(stats.ByAI, *s)
	}

	// 最近 20 条记录
	n := len(rs.records)
	limit := 20
	if n < limit {
		limit = n
	}
	stats.RecentRecords = make([]GameRecord, limit)
	for i := 0; i < limit; i++ {
		stats.RecentRecords[i] = rs.records[n-1-i] // 倒序，最新在前
	}

	return stats
}

// save 将内存中的记录写入 JSON 文件（原子写入：写临时文件 → rename）。
// 调用方必须持有 mu 写锁。
func (rs *RecordStore) save() error {
	data, err := json.MarshalIndent(rs.records, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal records: %w", err)
	}

	tmpPath := rs.filePath + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("write temp file: %w", err)
	}

	if err := os.Rename(tmpPath, rs.filePath); err != nil {
		return fmt.Errorf("rename temp file: %w", err)
	}

	return nil
}

// winnerLabel 将引擎的 CellState 转为可读标签。
func winnerLabel(w game_engine.CellState) string {
	switch w {
	case game_engine.Black:
		return "black"
	case game_engine.White:
		return "white"
	default:
		return "draw"
	}
}

// aiLabel 将空字符串的 AI 类型转为 "human"。
func aiLabel(s string) string {
	if s == "" {
		return "human"
	}
	return s
}

// formatTime 格式化时间为简洁的本地时间字符串。
func formatTime(t time.Time) string {
	return t.Format("2006-01-02 15:04")
}
