import React, { useState } from 'react';
import { AIType, GameConfig, GameMode } from '../types';
import './GameControl.css';

/** 父组件传入的属性 */
interface GameControlProps {
  onCreateGame: (config: GameConfig) => void;  // 创建游戏回调，传递模式和 AI 配置
  loading: boolean;                             // 是否正在创建中（禁用按钮防重复点击）
}

/** 可选 AI 策略列表，用于下拉选择 */
const AI_OPTIONS: { value: AIType; label: string; desc: string }[] = [
  { value: 'mcts', label: 'MCTS', desc: '蒙特卡洛树搜索' },
  { value: 'alphabeta', label: 'Alpha-Beta', desc: '博弈树搜索，最优平衡' },
  { value: 'heuristic', label: '启发式', desc: '棋型评估，快速响应' },
  { value: 'alphazero', label: 'AlphaZero', desc: '深度学习策略' },
];

/**
 * 游戏创建面板
 *
 * 提供四种游戏模式的选择：
 * - 执黑先手（人 vs AI）
 * - 执白后手（AI vs 人）
 * - 双人对战（人 vs 人）
 * - AI 对战（AI vs AI，可分别选策略）
 *
 * 根据所选模式动态显示不同的配置项（对手 AI 选择、双方 AI 选择等）。
 */
const GameControl: React.FC<GameControlProps> = ({ onCreateGame, loading }) => {
  // 当前选中的游戏模式
  const [mode, setMode] = useState<GameMode>('pve_black');
  // PvE 模式下对手 AI 的策略类型
  const [aiType, setAiType] = useState<AIType>('mcts');
  // AI vs AI 模式下黑棋的策略类型
  const [blackAI, setBlackAI] = useState<AIType>('mcts');
  // AI vs AI 模式下白棋的策略类型
  const [whiteAI, setWhiteAI] = useState<AIType>('mcts');

  /**
   * 根据当前模式组装 GameConfig 并回调父组件
   * pve_black: 人(黑) vs AI(白)   — 人类先手
   * pve_white: AI(黑) vs 人(白)   — 人类后手
   * pvp:       人(黑) vs 人(白)   — 双人同屏
   * ai_vs_ai:  AI(黑) vs AI(白)   — AI 自动对弈
   */
  const handleCreate = () => {
    switch (mode) {
      case 'pvp':
        onCreateGame({ mode: 'pvp', blackKind: 'human', whiteKind: 'human' });
        break;
      case 'pve_black':
        onCreateGame({ mode: 'pve_black', blackKind: 'human', whiteKind: aiType });
        break;
      case 'pve_white':
        onCreateGame({ mode: 'pve_white', blackKind: aiType, whiteKind: 'human' });
        break;
      case 'ai_vs_ai':
        onCreateGame({ mode: 'ai_vs_ai', blackKind: blackAI, whiteKind: whiteAI });
        break;
    }
  };

  return (
    <div className="game-control">
      <h2>创建新游戏</h2>

      <div className="mode-selector">
        <button
          className={`mode-btn ${mode === 'pve_black' ? 'active' : ''}`}
          onClick={() => setMode('pve_black')}
        >
          执黑先手
        </button>
        <button
          className={`mode-btn ${mode === 'pve_white' ? 'active' : ''}`}
          onClick={() => setMode('pve_white')}
        >
          执白后手
        </button>
        <button
          className={`mode-btn ${mode === 'pvp' ? 'active' : ''}`}
          onClick={() => setMode('pvp')}
        >
          双人对战
        </button>
        <button
          className={`mode-btn ${mode === 'ai_vs_ai' ? 'active' : ''}`}
          onClick={() => setMode('ai_vs_ai')}
        >
          AI 对战
        </button>
      </div>

      {(mode === 'pve_black' || mode === 'pve_white') && (
        <div className="pve-config">
          <p className="config-hint">
            {mode === 'pve_black' ? '你执黑先手' : '你执白后手'}，对手策略：
          </p>
          <div className="ai-select">
            <select value={aiType} onChange={e => setAiType(e.target.value as AIType)}>
              {AI_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label} - {opt.desc}
                </option>
              ))}
            </select>
          </div>
          <p className="config-note">对局中可开启"托管"让 AI 代你落子</p>
        </div>
      )}

      {mode === 'ai_vs_ai' && (
        <div className="ai-select-group">
          <div className="ai-select">
            <label>黑棋 AI (先手)</label>
            <select value={blackAI} onChange={e => setBlackAI(e.target.value as AIType)}>
              {AI_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <div className="ai-select">
            <label>白棋 AI (后手)</label>
            <select value={whiteAI} onChange={e => setWhiteAI(e.target.value as AIType)}>
              {AI_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <p className="config-note">对局有 1.5 秒落子间隔，可暂停观察</p>
        </div>
      )}

      {mode === 'pvp' && (
        <p className="config-hint">两位玩家在同一设备上轮流落子</p>
      )}

      <button
        className="btn-primary"
        onClick={handleCreate}
        disabled={loading}
      >
        {loading ? '创建中...' : '开始游戏'}
      </button>
    </div>
  );
};

export default GameControl;
