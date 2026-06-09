import React, { useState, useCallback } from 'react';
import Board from './components/Board';
import GameControl from './components/GameControl';
import { gameApi } from './services/api';
import { Game, AIType, Piece } from './types';
import './App.css';

const App: React.FC = () => {
  const [game, setGame] = useState<Game | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aiThinking, setAiThinking] = useState(false);

  const createGame = useCallback(async (blackAI: AIType, whiteAI: AIType) => {
    setLoading(true);
    setError(null);
    try {
      const res = await gameApi.createGame({ black_ai: blackAI, white_ai: whiteAI });
      setGame(res.game);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleMove = useCallback(async (row: number, col: number) => {
    if (!game || game.status !== 'playing' || aiThinking) return;

    setLoading(true);
    setError(null);
    try {
      const res = await gameApi.makeMove(game.id, { row, col });
      setGame(res.game);

      // If game still playing, trigger AI move
      if (res.game.status === 'playing') {
        setAiThinking(true);
        const aiRes = await gameApi.getAIMove(game.id);
        setGame(aiRes.game);
        setAiThinking(false);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [game, aiThinking]);

  const handleAIMove = useCallback(async () => {
    if (!game || game.status !== 'playing' || aiThinking) return;

    setAiThinking(true);
    setError(null);
    try {
      const res = await gameApi.getAIMove(game.id);
      setGame(res.game);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setAiThinking(false);
    }
  }, [game, aiThinking]);

  const getStatusText = (): string => {
    if (!game) return '';
    switch (game.status) {
      case 'black_win': return '黑棋获胜!';
      case 'white_win': return '白棋获胜!';
      case 'draw': return '平局!';
      default:
        const player = game.current_player === 1 ? '黑棋' : '白棋';
        return aiThinking ? `${player}思考中...` : `${player}回合`;
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>GomokuMind</h1>
        <p className="subtitle">15×15 五子棋AI对战平台</p>
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
                {aiThinking && <span className="thinking-indicator">AI思考中...</span>}
              </div>
              <div className="game-details">
                <span>黑棋AI: {game.black_ai}</span>
                <span>白棋AI: {game.white_ai}</span>
                <span>步数: {game.move_history.length}</span>
              </div>
              <button
                className="btn-secondary"
                onClick={() => setGame(null)}
                disabled={loading}
              >
                新游戏
              </button>
            </div>

            <Board
              board={game.board}
              currentPlayer={game.current_player}
              onMove={handleMove}
              disabled={game.status !== 'playing' || aiThinking || loading}
              lastMove={game.move_history.length > 0 ? game.move_history[game.move_history.length - 1] : null}
            />

            <div className="game-actions">
              {game.status === 'playing' && !aiThinking && (
                <button
                  className="btn-primary"
                  onClick={handleAIMove}
                  disabled={loading}
                >
                  AI代下
                </button>
              )}
            </div>
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
