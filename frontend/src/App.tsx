import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import Board from './components/Board';
import GameControl from './components/GameControl';
import ThemeSwitcher from './components/ThemeSwitcher';
import { gameApi } from './services/api';
import { Game, GameConfig, GameMode, PlayerKind, AIType } from './types';
import { ThemeKey, DEFAULT_THEME, THEMES, BOARD_THEMES } from './themes';
import './App.css';

const AI_MOVE_DELAY = 1500; // AI vs AI 模式下的落子间隔 (ms)
const AI_RESPONSE_DELAY = 400;  // PvE 模式下 AI 响应间隔 (ms)

/* 各主题对应的装饰粒子 */
const PARTICLE_MAP: Record<ThemeKey, string> = {
  spring: '🌸',
  autumn: '🍂',
  winter: '❄️',
  starry: '✨',
};

const App: React.FC = () => {
  const [game, setGame] = useState<Game | null>(null);
  const [config, setConfig] = useState<GameConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aiThinking, setAiThinking] = useState(false);
  const [autoPlay, setAutoPlay] = useState(false); // 托管/自动对弈开关
  const [theme, setTheme] = useState<ThemeKey>(DEFAULT_THEME);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 创建游戏
  const createGame = useCallback(async (cfg: GameConfig) => {
    setLoading(true);
    setError(null);
    setAutoPlay(false);
    setConfig(cfg);

    // 映射到 server 的 black_ai / white_ai
    const toAI = (k: PlayerKind): AIType => (k === 'human' ? 'heuristic' : k);
    const blackAI = toAI(cfg.blackKind);
    const whiteAI = toAI(cfg.whiteKind);

    try {
      const res = await gameApi.createGame({ black_ai: blackAI, white_ai: whiteAI });
      setGame(res.game);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // 判断当前是否该 AI 落子
  const isCurrentPlayerAI = useCallback((): boolean => {
    if (!game || !config) return false;
    const isBlack = game.current_player === 1;
    const kind = isBlack ? config.blackKind : config.whiteKind;

    if (config.mode === 'pvp') return false;
    if (config.mode === 'ai_vs_ai') return true;
    if (config.mode === 'pve_black' || config.mode === 'pve_white') {
      return autoPlay || kind !== 'human';
    }
    return false;
  }, [game, config, autoPlay]);

  // 是否允许人类点击落子
  const canHumanMove = useCallback((): boolean => {
    if (!game || !config || game.status !== 'playing' || aiThinking || loading) return false;
    if (config.mode === 'ai_vs_ai') return false;
    if (config.mode === 'pvp') return true;
    if (autoPlay) return false;
    const isBlack = game.current_player === 1;
    const kind = isBlack ? config.blackKind : config.whiteKind;
    return kind === 'human';
  }, [game, config, aiThinking, loading, autoPlay]);

  // 人类落子
  const handleMove = useCallback(async (row: number, col: number) => {
    if (!canHumanMove() || !game) return;

    setLoading(true);
    setError(null);
    try {
      const res = await gameApi.makeMove(game.id, { row, col });
      setGame(res.game);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [game, canHumanMove]);

  // AI 落子（由 useEffect 触发）
  const triggerAIMove = useCallback(async () => {
    if (!game || game.status !== 'playing') return;

    setAiThinking(true);
    try {
      const res = await gameApi.getAIMove(game.id);
      setGame(res.game);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setAiThinking(false);
    }
  }, [game]);

  // 创建游戏后，如果是执白后手模式，AI 自动下第一步
  useEffect(() => {
    if (game && config && game.status === 'playing' && game.move_history.length === 0) {
      if (config.mode === 'pve_white' || config.mode === 'ai_vs_ai') {
        if (config.mode === 'ai_vs_ai' && !autoPlay) return;
        const timer = setTimeout(() => {
          triggerAIMove();
        }, AI_RESPONSE_DELAY);
        return () => clearTimeout(timer);
      }
    }
  }, [game?.id]);

  // AI 自动落子循环
  useEffect(() => {
    if (!game || game.status !== 'playing' || !config) return;
    if (!isCurrentPlayerAI()) return;

    // AI vs AI 模式下需要 autoPlay 为 true
    if (config.mode === 'ai_vs_ai' && !autoPlay) return;

    const delay = config.mode === 'ai_vs_ai' ? AI_MOVE_DELAY : AI_RESPONSE_DELAY;

    timerRef.current = setTimeout(() => {
      triggerAIMove();
    }, delay);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [game, game?.current_player, game?.status, config, autoPlay, isCurrentPlayerAI, triggerAIMove]);

  // 同步主题到 html 标签
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // 清理 timer
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // 当前主题信息
  const themeInfo = useMemo(() => THEMES.find(t => t.key === theme)!, [theme]);

  const boardColors = useMemo(() => BOARD_THEMES[theme], [theme]);

  // 一键 AI 代下（单步）
  const handleSingleAI = useCallback(async () => {
    if (!game || game.status !== 'playing' || aiThinking) return;
    triggerAIMove();
  }, [game, aiThinking, triggerAIMove]);

  // 托管开关切换
  const toggleAutoPlay = useCallback(() => {
    setAutoPlay(v => !v);
  }, []);

  // 状态文字
  const getStatusText = (): string => {
    if (!game) return '';
    switch (game.status) {
      case 'black_win': return '黑棋获胜！';
      case 'white_win': return '白棋获胜！';
      case 'draw': return '平局！';
      default:
        const player = game.current_player === 1 ? '黑棋' : '白棋';
        if (config?.mode === 'pvp') return `${player} 回合`;
        if (aiThinking) return `${player} 思考中...`;
        if (canHumanMove()) return `你的回合 (${player})`;
        return `${player} 回合`;
    }
  };

  // 玩家类型标签
  const playerLabel = (kind: PlayerKind): string => {
    if (kind === 'human') return '玩家';
    return `AI (${kind})`;
  };

  return (
    <div className="app">
      {/* 装饰粒子层 */}
      <div className={`particles-layer ${themeInfo.particles}`}>
        {Array.from({ length: 10 }).map((_, i) => (
          <span key={i} className="particle">{PARTICLE_MAP[theme]}</span>
        ))}
      </div>

      <header className="app-header">
        <h1><span className="header-icon">{themeInfo.icon}</span> GomokuMind</h1>
        <p className="subtitle">15×15 五子棋AI对战平台</p>
        <ThemeSwitcher current={theme} onChange={setTheme} />
      </header>

      <main className="app-main">
        {!game ? (
          <GameControl onCreateGame={createGame} loading={loading} />
        ) : (
          <div className="game-container">
            <div className="game-info">
              <div className="status-bar">
                <span className={`status ${game.status !== 'playing' ? 'winner' : ''}`}>
                  {getStatusText()}
                </span>
                {autoPlay && (
                  <span className="autoplay-badge">托管中</span>
                )}
              </div>
              <div className="game-details">
                <span>黑棋: {config ? playerLabel(config.blackKind) : ''}</span>
                <span>白棋: {config ? playerLabel(config.whiteKind) : ''}</span>
                <span>步数: {game.move_history.length}</span>
              </div>

              <div className="game-controls">
                <button
                  className="btn-secondary"
                  onClick={() => { setGame(null); setConfig(null); setAutoPlay(false); }}
                  disabled={loading}
                >
                  新游戏
                </button>

                {/* 托管按钮：仅 PvE 和 AI vs AI 模式显示 */}
                {config && config.mode !== 'pvp' && game.status === 'playing' && (
                  <button
                    className={`btn-autoplay ${autoPlay ? 'active' : ''}`}
                    onClick={toggleAutoPlay}
                  >
                    {autoPlay ? '取消托管' : (config.mode === 'ai_vs_ai' ? '开始对弈' : '开启托管')}
                  </button>
                )}

                {/* AI 代下单步：仅 PvE 且非托管时显示 */}
                {config && (config.mode === 'pve_black' || config.mode === 'pve_white')
                  && game.status === 'playing' && !autoPlay && !aiThinking && (
                  <button
                    className="btn-hint"
                    onClick={handleSingleAI}
                    disabled={loading}
                  >
                    AI 代下
                  </button>
                )}
              </div>
            </div>

            <Board
              board={game.board}
              currentPlayer={game.current_player}
              onMove={handleMove}
              disabled={!canHumanMove()}
              lastMove={game.move_history.length > 0 ? game.move_history[game.move_history.length - 1] : null}
              boardColors={boardColors}
            />
          </div>
        )}

        {error && (
          <div className="error-toast">
            <span>{error}</span>
            <button onClick={() => setError(null)}>×</button>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
