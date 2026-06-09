import React, { useState } from 'react';
import { AIType } from '../types';
import './GameControl.css';

interface GameControlProps {
  onCreateGame: (blackAI: AIType, whiteAI: AIType) => void;
  loading: boolean;
}

const AI_OPTIONS: { value: AIType; label: string; desc: string }[] = [
  { value: 'hybrid', label: '混合策略', desc: '威胁检测 + Minimax搜索 (推荐)' },
  { value: 'minimax', label: 'Minimax', desc: 'Alpha-Beta剪枝 + 迭代加深' },
  { value: 'threat', label: '威胁空间', desc: 'VCF/VCT 连续冲四求解' },
  { value: 'mcts', label: 'MCTS', desc: '蒙特卡洛树搜索' },
  { value: 'heuristic', label: '启发式', desc: '纯棋型评估，快速响应' },
];

const GameControl: React.FC<GameControlProps> = ({ onCreateGame, loading }) => {
  const [blackAI, setBlackAI] = useState<AIType>('hybrid');
  const [whiteAI, setWhiteAI] = useState<AIType>('hybrid');
  const [mode, setMode] = useState<'ai' | 'pvp'>('ai');

  const handleCreate = () => {
    if (mode === 'pvp') {
      onCreateGame('heuristic', 'heuristic');
    } else {
      onCreateGame(blackAI, whiteAI);
    }
  };

  return (
    <div className="game-control">
      <h2>创建新游戏</h2>

      <div className="mode-selector">
        <button
          className={`mode-btn ${mode === 'ai' ? 'active' : ''}`}
          onClick={() => setMode('ai')}
        >
          人机对战
        </button>
        <button
          className={`mode-btn ${mode === 'pvp' ? 'active' : ''}`}
          onClick={() => setMode('pvp')}
        >
          双人对战
        </button>
      </div>

      {mode === 'ai' && (
        <div className="ai-select-group">
          <div className="ai-select">
            <label>黑棋AI (先手)</label>
            <select value={blackAI} onChange={e => setBlackAI(e.target.value as AIType)}>
              {AI_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label} - {opt.desc}
                </option>
              ))}
            </select>
          </div>
          <div className="ai-select">
            <label>白棋AI (后手)</label>
            <select value={whiteAI} onChange={e => setWhiteAI(e.target.value as AIType)}>
              {AI_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label} - {opt.desc}
                </option>
              ))}
            </select>
          </div>
        </div>
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
