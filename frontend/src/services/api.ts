import { Game, CreateGameRequest, MoveRequest, GameResponse, MoveResponse } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Request failed' }));
    throw new Error(error.error || `HTTP ${response.status}`);
  }
  
  return response.json();
}

export const gameApi = {
  createGame: (req: CreateGameRequest) =>
    request<GameResponse>('/gomoku', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  getGame: (id: string) =>
    request<GameResponse>(`/gomoku/${id}`),

  makeMove: (id: string, req: MoveRequest) =>
    request<GameResponse>(`/gomoku/${id}`, {
      method: 'PUT',
      body: JSON.stringify(req),
    }),

  getAIMove: (id: string) =>
    request<MoveResponse>(`/gomoku/${id}/ai-move`, {
      method: 'POST',
    }),

  listGames: () =>
    request<{ games: Game[] }>('/gomoku'),

  deleteGame: (id: string) =>
    request<{ message: string }>(`/gomoku/${id}`, {
      method: 'DELETE',
    }),

  healthCheck: () =>
    request<{ status: string }>('/health'),
};
