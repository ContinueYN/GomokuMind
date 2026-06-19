import { useState, useEffect } from 'react';
import { gameApi } from '../services/api';
import { StatsResponse } from '../types';
import './RecordsView.css';

interface RecordsViewProps {
  onBack: () => void;
}

/** AI 类型对应的中文显示名 */
const AI_LABELS: Record<string, string> = {
  human: '人类',
  heuristic: 'heuristic',
  mcts: 'MCTS',
  alphabeta: 'Alpha-Beta',
  alphazero: 'AlphaZero',
  minimax: 'Minimax',
};

function aiLabel(type: string): string {
  return AI_LABELS[type] || type;
}

/** 结果对应的徽章样式和文字 */
function resultBadge(status: string, winner: string) {
  switch (winner) {
    case 'black':
      return { className: 'badge-win badge-black', text: '黑胜' };
    case 'white':
      return { className: 'badge-win badge-white', text: '白胜' };
    case 'draw':
      return { className: 'badge-draw', text: '平局' };
    default:
      return { className: 'badge-draw', text: status };
  }
}

/** 格式化 ISO 时间为本地短格式 */
function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
  } catch {
    return iso;
  }
}

export default function RecordsView({ onBack }: RecordsViewProps) {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await gameApi.getStats();
        if (!cancelled) {
          setStats(data);
          setError(null);
        }
      } catch (e: any) {
        if (!cancelled) {
          setError(e.message || '加载对局记录失败');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    // 用 rAF 延迟触发渐入，避免 React StrictMode 双挂载导致闪烁
    const raf = requestAnimationFrame(() => {
      if (!cancelled) setVisible(true);
    });
    return () => { cancelled = true; cancelAnimationFrame(raf); };
  }, []);

  if (loading) {
    return (
      <div className={`records-view${visible ? ' visible' : ''}`}>
        <div className="records-loading">加载对局记录中…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`records-view${visible ? ' visible' : ''}`}>
        <div className="records-error">
          <p>⚠️ {error}</p>
          <button className="records-back-btn" onClick={onBack}>返回</button>
        </div>
      </div>
    );
  }

  if (!stats) return null;

  const maxWinRate = Math.max(...stats.by_ai.map(s => s.win_rate), 0);

  return (
    <div className={`records-view${visible ? ' visible' : ''}`}>
      <div className="records-header">
        <h2>对局记录</h2>
        <button className="records-back-btn" onClick={onBack}>← 返回</button>
      </div>

      {/* ---- 统计卡片 ---- */}
      <div className="stats-cards">
        <div className="stat-card">
          <div className="stat-number">{stats.total_games}</div>
          <div className="stat-label">总局数</div>
        </div>
        <div className="stat-card stat-black">
          <div className="stat-number">{stats.black_wins}</div>
          <div className="stat-label">黑棋胜</div>
          <div className="stat-pct">
            {stats.total_games > 0
              ? `${Math.round((stats.black_wins / stats.total_games) * 100)}%`
              : '—'}
          </div>
        </div>
        <div className="stat-card stat-white">
          <div className="stat-number">{stats.white_wins}</div>
          <div className="stat-label">白棋胜</div>
          <div className="stat-pct">
            {stats.total_games > 0
              ? `${Math.round((stats.white_wins / stats.total_games) * 100)}%`
              : '—'}
          </div>
        </div>
        <div className="stat-card stat-draw">
          <div className="stat-number">{stats.draws}</div>
          <div className="stat-label">平局</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{stats.avg_moves.toFixed(1)}</div>
          <div className="stat-label">平均手数</div>
        </div>
      </div>

      {/* ---- AI 胜率柱状图 ---- */}
      {stats.by_ai.length > 0 && (
        <div className="records-section">
          <h3>AI 胜率对比</h3>
          <div className="bar-chart">
            {stats.by_ai.map(s => (
              <div className="bar-item" key={s.ai_type}>
                <div className="bar-label">{aiLabel(s.ai_type)}</div>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{ width: `${s.win_rate}%` }}
                  />
                </div>
                <div className="bar-value">{s.win_rate.toFixed(0)}%</div>
                <div className="bar-detail">{s.wins}胜 / {s.losses}负 / {s.total}局</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ---- 最近对局 ---- */}
      <div className="records-section">
        <h3>最近对局</h3>
        {stats.recent_records.length === 0 ? (
          <p className="records-empty">暂无对局记录，完成一局后自动记录</p>
        ) : (
          <div className="records-table-wrap">
            <table className="records-table">
              <thead>
                <tr>
                  <th>时间</th>
                  <th>黑方</th>
                  <th>白方</th>
                  <th>结果</th>
                  <th>手数</th>
                </tr>
              </thead>
              <tbody>
                {stats.recent_records.map(r => {
                  const badge = resultBadge(r.status, r.winner);
                  return (
                    <tr key={r.id}>
                      <td className="col-time">{formatTime(r.finished_at)}</td>
                      <td>{aiLabel(r.black_ai)}</td>
                      <td>{aiLabel(r.white_ai)}</td>
                      <td><span className={badge.className}>{badge.text}</span></td>
                      <td className="col-moves">{r.move_count}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
