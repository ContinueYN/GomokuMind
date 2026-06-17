import { Game, CreateGameRequest, MoveRequest, GameResponse, MoveResponse, AIType, StatsResponse, GameRecord } from '../types';

// API 基础路径：优先使用环境变量，否则走 Vite 代理前缀 /api
const API_BASE = import.meta.env.VITE_API_URL || '/api';

/**
 * 通用 HTTP 请求封装
 * - 自动拼接 API_BASE 前缀
 * - 默认带上 JSON Content-Type，同时支持调用方追加自定义 headers
 * - 非 2xx 响应统一抛出 Error
 * - 泛型 T 约束返回值的 JSON 类型
 */
async function request<T>(url: string, options?: RequestInit): Promise<T> {
  // 提取调用方传入的自定义 headers，避免浅层展开覆盖掉 Content-Type
  const { headers: customHeaders, ...restOptions } = options || {};

  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
      ...customHeaders, // 延后展开，确保调用方 headers 不会覆盖 Content-Type
    },
    ...restOptions,
  });

  if (!response.ok) {
    // 尝试从响应体中读取服务器返回的 error 字段；解析失败则兜底
    const errorBody = await response.json().catch(() => ({ error: 'Request failed' }));
    throw new Error(errorBody.error || `HTTP ${response.status}`);
  }

  return response.json();
}

// ---- 导出：面向业务的方法集合 ----
export const gameApi = {
  // 创建新游戏：传入黑白双方的 AI 类型（人类玩家不传）
  createGame: (req: CreateGameRequest) =>
    request<GameResponse>('/gomoku', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  // 获取指定游戏的最新状态（棋盘、当前玩家、胜负等）
  getGame: (id: string) =>
    request<GameResponse>(`/gomoku/${id}`),

  // 人类落子：向指定游戏提交行列坐标
  makeMove: (id: string, req: MoveRequest) =>
    request<GameResponse>(`/gomoku/${id}`, {
      method: 'PUT',
      body: JSON.stringify(req),
    }),

  // AI 落子：可传入 ai 参数临时指定策略（不传则使用游戏创建时配置的 AI）
  getAIMove: (id: string, ai?: AIType) =>
    request<MoveResponse>(`/gomoku/${id}/ai-move`, {
      method: 'POST',
      body: ai ? JSON.stringify({ ai }) : undefined,
    }),

  // 列出所有进行中的游戏
  listGames: () =>
    request<{ games: Game[] }>('/gomoku'),

  // 删除指定游戏
  deleteGame: (id: string) =>
    request<{ message: string }>(`/gomoku/${id}`, {
      method: 'DELETE',
    }),

  // 健康检查：确认服务器是否在线
  healthCheck: () =>
    request<{ status: string }>('/health'),

  // 对局统计：总局数、胜率、AI 排行
  getStats: () =>
    request<StatsResponse>('/stats'),

  // 对局历史记录，支持 ?limit=N（默认 50）
  getRecords: (limit?: number) =>
    request<{ records: GameRecord[] }>(`/records${limit ? `?limit=${limit}` : ''}`),
};
