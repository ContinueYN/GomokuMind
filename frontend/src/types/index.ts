export interface Move {
  row: number;
  col: number;
}

export type Piece = 0 | 1 | 2;

export type GameStatus = 'playing' | 'black_win' | 'white_win' | 'draw';

export type AIType = 'heuristic' | 'mcts' | 'q-learning' | 'ppo';

export interface Board {
  board: Piece[][];
}

export interface Game {
  id: string;
  board: Piece[][];
  current_player: Piece;
  status: GameStatus;
  move_history: Move[];
  black_ai: AIType;
  white_ai: AIType;
  created_at: string;
  updated_at: string;
}

export interface CreateGameRequest {
  black_ai: AIType;
  white_ai: AIType;
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
