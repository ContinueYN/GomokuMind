import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import Board from './components/Board';
import GameControl from './components/GameControl';
import ThemeSwitcher from './components/ThemeSwitcher';
import PoemDisplay from './components/PoemDisplay';
import { gameApi } from './services/api';
import { Game, GameConfig, CreateGameRequest, PlayerKind, AIType } from './types';
import { ThemeKey, DEFAULT_THEME, THEMES, BOARD_THEMES } from './themes';
import './App.css';

const AI_MOVE_DELAY = 1500; // AI vs AI 模式下的落子间隔 (ms)
const AI_RESPONSE_DELAY = 500;  // PvE 模式下 AI 响应间隔 (ms)

const App: React.FC = () => {
  // 存储完整的游戏状态：棋盘、当前玩家、历史记录、胜负等
  const [game, setGame] = useState<Game | null>(null);

  // 存储游戏配置：模式(pvp/pve/ai_vs_ai)、黑白棋类型
  const [config, setConfig] = useState<GameConfig | null>(null);

  // 加载状态，防止重复点击和显示加载动画
  const [loading, setLoading] = useState(false);

  // 错误信息，如网络错误、服务器错误等
  const [error, setError] = useState<string | null>(null);

  // AI思考状态，用于显示"思考中..."提示
  const [aiThinking, setAiThinking] = useState(false);

  // 托管开关：true时AI自动下棋，false时手动
  const [autoPlay, setAutoPlay] = useState(false); 
  
  // AI代下策略类型：MCTS / Alpha-Beta / 启发式 / Q-Learning
  const [hintAI, setHintAI] = useState<AIType>('mcts');

  // 当前主题
  const [theme, setTheme] = useState<ThemeKey>(DEFAULT_THEME);

  // 存储定时器ID，用于清理
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 对局结束弹窗：是否已被用户关闭
  const [gameOverDismissed, setGameOverDismissed] = useState(false);

  // 当游戏结束时（对局刚结束），重置弹窗状态
  useEffect(() => {
    if (game && game.status !== 'playing') {
      setGameOverDismissed(false);
    }
  }, [game?.status]);

  // 创建游戏：仅 AI 传递策略类型，真人玩家不传
  const createGame = useCallback(async (cfg: GameConfig) => {
    setLoading(true);
    setError(null);
    setAutoPlay(false);
    setConfig(cfg);

    const req: CreateGameRequest = {};
    if (cfg.blackKind !== 'human') req.black_ai = cfg.blackKind;
    if (cfg.whiteKind !== 'human') req.white_ai = cfg.whiteKind;

    try {
      const res = await gameApi.createGame(req);
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
  }, [game?.current_player, game?.status, config?.mode, config?.blackKind, config?.whiteKind, autoPlay]);

  // 是否允许人类点击落子
  const canHumanMove = useCallback((): boolean => {
    // 必须满足：游戏存在 && 配置存在 && 游戏中 && AI没在思考 && 没在加载
    if (!game || !config || game.status !== 'playing' || aiThinking || loading) return false;
    if (config.mode === 'ai_vs_ai') return false;
    if (config.mode === 'pvp') return true;
    if (autoPlay) return false;
    const isBlack = game.current_player === 1;
    const kind = isBlack ? config.blackKind : config.whiteKind;
    // 只有当前玩家是人类的回合才能下
    return kind === 'human';
  }, [game?.current_player, game?.status, config?.mode, config?.blackKind, config?.whiteKind, autoPlay, aiThinking, loading]);

  // 人类落子
  const handleMove = useCallback(async (row: number, col: number) => {
    if (!canHumanMove() || !game) return;

    setLoading(true);
    setError(null);
    try {
      const res = await gameApi.makeMove(game.id, { row, col });
      // 更新游戏状态
      setGame(res.game);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [game?.id, canHumanMove]);

  // AI 落子（aiOverride 为可选代下策略，可以临时指定AI策略
  const triggerAIMove = useCallback(async (aiOverride?: AIType) => {
    if (!game || game.status !== 'playing') return;

    setAiThinking(true);
    try {
      const res = await gameApi.getAIMove(game.id, aiOverride);
      // 如果传了aiOverride（如'mcts'），就用这个策略下棋
      // 如果不传，用游戏创建时配置的AI策略
      setGame(res.game);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setAiThinking(false);
    }
  }, [game?.id, game?.status]);

  // 创建游戏后，AI vs AI 模式需手动启动，不在此自动开始
  useEffect(() => {
    if (game && config && game.status === 'playing' && game.move_history.length === 0) {
      if (config.mode === 'ai_vs_ai') {
        if (!autoPlay) return;
        const timer = setTimeout(() => {
          triggerAIMove();
        }, AI_RESPONSE_DELAY);
        return () => clearTimeout(timer);
      }
    }
  }, [game?.id, config, autoPlay, triggerAIMove]);

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
  }, [game?.id, config?.mode, autoPlay, isCurrentPlayerAI, triggerAIMove]);

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

  // 当前主题信息（！是非空断言，意味着这个值一定存在）
  const themeInfo = useMemo(() => THEMES.find(t => t.key === theme)!, [theme]);

  const boardColors = useMemo(() => BOARD_THEMES[theme], [theme]);

  // AI提示（单步代下）
  const handleSingleAI = useCallback(async () => {
    if (!game || game.status !== 'playing' || aiThinking) return;
    triggerAIMove(hintAI);
  }, [game?.status, aiThinking, hintAI, triggerAIMove]);

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
  const playerLabel = useCallback((kind: PlayerKind): string => {
    if (kind === 'human') return '玩家';
    return `AI (${kind})`;
  }, []);

  // 对局结束弹窗信息
  const gameOverInfo = useMemo(() => {
    if (!game || !config || game.status === 'playing') return null;
    switch (game.status) {
      case 'black_win': return {
        result: '黑棋获胜！',
        detail: `黑方: ${playerLabel(config.blackKind)}  ·  ${game.move_history.length} 手`,
      };
      case 'white_win': return {
        result: '白棋获胜！',
        detail: `白方: ${playerLabel(config.whiteKind)}  ·  ${game.move_history.length} 手`,
      };
      case 'draw': return {
        result: '平局！',
        detail: `双方势均力敌  ·  ${game.move_history.length} 手`,
      };
      default: return null;
    }
  }, [game?.status, config, game?.move_history.length, playerLabel]);

  return (
    <div className="app">
      {/* 诗句背景层 */}
      <PoemDisplay />

      <header className="app-header">
        <h1><img className="header-icon" src={themeInfo.icon} alt={themeInfo.name} /> GomokuMind</h1>
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
                  onClick={() => { setGame(null); setConfig(null); setAutoPlay(false); setGameOverDismissed(true); }}
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

                {/* AI 代下：仅 PvE 且非托管时显示，可自选策略 */}
                {config && (config.mode === 'pve_black' || config.mode === 'pve_white')
                  && game.status === 'playing' && !autoPlay && !aiThinking && (
                  <div className="hint-group">
                    <select
                      className="hint-select"
                      value={hintAI}
                      onChange={e => setHintAI(e.target.value as AIType)}
                    >
                      <option value="mcts">MCTS</option>
                      <option value="alphabeta">Alpha-Beta</option>
                      <option value="heuristic">启发式</option>
                      <option value="alphazero">AlphaZero</option>
                    </select>
                    <button
                      className="btn-hint"
                      onClick={handleSingleAI}
                      disabled={loading}
                    >
                      AI 代下
                    </button>
                  </div>
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

        {/* 对局结束弹窗 */}
        {gameOverInfo && !gameOverDismissed && (
          <div className="gameover-overlay" onClick={() => setGameOverDismissed(true)}>
            <div className="gameover-modal" onClick={e => e.stopPropagation()}>
              <div className="gameover-result">{gameOverInfo.result}</div>
              <div className="gameover-detail">{gameOverInfo.detail}</div>
              <div className="gameover-buttons">
                <button
                  className="gameover-btn primary"
                  onClick={() => { setGame(null); setConfig(null); setAutoPlay(false); setGameOverDismissed(true); }}
                >
                  新游戏
                </button>
                <button
                  className="gameover-btn secondary"
                  onClick={() => setGameOverDismissed(true)}
                >
                  关闭
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
