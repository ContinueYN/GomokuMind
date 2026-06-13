import React, { useRef, useEffect, useState } from 'react';
import { Piece, Move } from '../types';
import { BoardThemeColors } from '../themes';
import './Board.css';

/** 组件接收的属性 */
interface BoardProps {
  board: Piece[][];          // 15x15 棋盘数据：0=空 1=黑 2=白
  currentPlayer: Piece;      // 当前回合方，用于悬停预览颜色
  onMove: (row: number, col: number) => void;  // 落子回调，由父组件处理 API 请求
  disabled: boolean;         // 禁止落子（非人类回合 / AI思考中 / 游戏结束）
  lastMove: Move | null;     // 上一步落子位置，绘制红星标记
  boardColors: BoardThemeColors;  // 当前主题对应的棋盘配色
}

// 棋盘绘制常量
const BOARD_SIZE = 15;       // 15×15 标准五子棋盘
const CELL_SIZE = 36;        // 单元格像素大小
const PADDING = 30;          // 棋盘四周留白
const CANVAS_SIZE = PADDING * 2 + CELL_SIZE * (BOARD_SIZE - 1);  // Canvas 总像素

/**
 * 棋盘组件
 *
 * 使用 Canvas 自绘棋盘，不依赖 DOM 元素。
 * 每次 board/lastMove/hoverPos/theme 变化时触发重绘。
 * 鼠标悬停时在空位显示半透明预览棋子，点击时回调 onMove。
 */
const Board: React.FC<BoardProps> = ({ board, currentPlayer, onMove, disabled, lastMove, boardColors }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // 鼠标悬停的格子坐标 [row, col]，null 表示不在棋盘上方
  const [hoverPos, setHoverPos] = useState<[number, number] | null>(null);

  /* ======== 主绘制函数 ======== */
  const drawBoard = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // -- 底色渐变 --
    const baseGrad = ctx.createLinearGradient(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    baseGrad.addColorStop(0, boardColors.bgStart);
    baseGrad.addColorStop(1, boardColors.bgEnd);
    ctx.fillStyle = baseGrad;
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // -- 网格线 --
    ctx.strokeStyle = boardColors.line;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.7;
    for (let i = 0; i < BOARD_SIZE; i++) {
      const pos = PADDING + i * CELL_SIZE;
      ctx.beginPath();
      ctx.moveTo(pos, PADDING);
      ctx.lineTo(pos, CANVAS_SIZE - PADDING);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(PADDING, pos);
      ctx.lineTo(CANVAS_SIZE - PADDING, pos);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // -- 星位点 --
    const starPoints = [
      [3, 3], [3, 11], [7, 7], [11, 3], [11, 11],
      [3, 7], [7, 3], [7, 11], [11, 7],
    ];
    ctx.fillStyle = boardColors.starPoint;
    for (const [r, c] of starPoints) {
      ctx.beginPath();
      ctx.arc(PADDING + c * CELL_SIZE, PADDING + r * CELL_SIZE, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // -- 棋子 --
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (board[r][c] !== 0) {
          const x = PADDING + c * CELL_SIZE;
          const y = PADDING + r * CELL_SIZE;
          const radius = CELL_SIZE * 0.42;

          // 阴影
          ctx.beginPath();
          ctx.arc(x + 2, y + 2, radius, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(0, 0, 0, 0.18)';
          ctx.fill();

          // 棋子主体
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);

          if (board[r][c] === 1) {
            const grad = ctx.createRadialGradient(x - 4, y - 4, 0, x, y, radius);
            grad.addColorStop(0, boardColors.blackStart);
            grad.addColorStop(1, boardColors.blackEnd);
            ctx.fillStyle = grad;
          } else {
            const grad = ctx.createRadialGradient(x - 4, y - 4, 0, x, y, radius);
            grad.addColorStop(0, boardColors.whiteStart);
            grad.addColorStop(1, boardColors.whiteEnd);
            ctx.fillStyle = grad;
          }
          ctx.fill();

          ctx.strokeStyle = board[r][c] === 1 ? boardColors.blackBorder : boardColors.whiteBorder;
          ctx.lineWidth = 0.5;
          ctx.stroke();

          // 光泽高光
          ctx.beginPath();
          ctx.arc(x - radius * 0.25, y - radius * 0.25, radius * 0.25, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
          ctx.fill();
        }
      }
    }

    // -- 最后落子标记 --
    if (lastMove) {
      const x = PADDING + lastMove.col * CELL_SIZE;
      const y = PADDING + lastMove.row * CELL_SIZE;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = boardColors.lastMove;
      ctx.fill();
      // 外光环
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.strokeStyle = boardColors.lastMove;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // -- 悬停预览 --
    if (hoverPos && !disabled) {
      const [hr, hc] = hoverPos;
      if (board[hr][hc] === 0) {
        const x = PADDING + hc * CELL_SIZE;
        const y = PADDING + hr * CELL_SIZE;
        ctx.beginPath();
        ctx.arc(x, y, CELL_SIZE * 0.42, 0, Math.PI * 2);
        ctx.fillStyle = currentPlayer === 1
          ? 'rgba(0, 0, 0, 0.25)'
          : 'rgba(255, 255, 255, 0.45)';
        ctx.fill();
      }
    }
  };

  // 依赖变化时重新绘制棋盘
  useEffect(() => {
    drawBoard();
  }, [board, lastMove, hoverPos, currentPlayer, disabled, boardColors]);

  /* ======== 鼠标坐标转换 ======== */
  /**
   * 将鼠标像素坐标转换为棋盘行列索引
   * 使用 Math.round 实现最近格点吸附（靠近哪个格子就映射到哪个格子）
   * 返回 [row, col]，超出棋盘范围返回 null
   */
  const getGridPos = (e: React.MouseEvent<HTMLCanvasElement>): [number, number] | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.round((x - PADDING) / CELL_SIZE);
    const row = Math.round((y - PADDING) / CELL_SIZE);
    if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
      return [row, col];
    }
    return null;
  };

  /** 点击落子：仅当未禁用且点击空位时触发 */
  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled) return;
    const pos = getGridPos(e);
    if (pos && board[pos[0]][pos[1]] === 0) {
      onMove(pos[0], pos[1]);
    }
  };

  /** 鼠标移动：更新悬停预览位置 */
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled) {
      setHoverPos(null);
      return;
    }
    const pos = getGridPos(e);
    setHoverPos(pos);
  };

  /** 鼠标离开棋盘：清除悬停预览 */
  const handleMouseLeave = () => {
    setHoverPos(null);
  };

  return (
    <div className="board-wrapper">
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="board-canvas"
      />
    </div>
  );
};

export default Board;
