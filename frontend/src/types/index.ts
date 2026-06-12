export interface Move {
  row: number;
  col: number;
}

export type Piece = 0 | 1 | 2;

export type GameStatus = 'playing' | 'black_win' | 'white_win' | 'draw';

export type AIType = 'heuristic' | 'mcts' | 'q-learning' | 'ppo';

// 游戏模式
export type GameMode = 'pvp' | 'pve_black' | 'pve_white' | 'ai_vs_ai';

// 玩家类型
export type PlayerKind = 'human' | AIType;

export interface GameConfig {
  mode: GameMode;
  blackKind: PlayerKind;
  whiteKind: PlayerKind;
}

export interface Board {
  board: Piece[][];
}

export interface Game {
  id: string;
  board: Piece[][];
  current_player: Piece;
  status: GameStatus;
  move_history: Move[];
  black_ai: string;
  white_ai: string;
  created_at: string;
  updated_at: string;
}

export interface CreateGameRequest {
  black_ai?: AIType;
  white_ai?: AIType;
}

export interface MoveRequest {
  row: number;
  col: number;
  ai?: AIType;
}

export interface GameResponse {
  game: Game;
  error?: string;
}

export interface MoveResponse {
  move: Move;
  game: Game;
}
