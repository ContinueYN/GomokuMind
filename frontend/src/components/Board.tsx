import React, { useRef, useEffect, useState } from 'react';
import { Piece, Move } from '../types';
import { BoardThemeColors } from '../themes';
import './Board.css';

interface BoardProps {
  board: Piece[][];
  currentPlayer: Piece;
  onMove: (row: number, col: number) => void;
  disabled: boolean;
  lastMove: Move | null;
  boardColors: BoardThemeColors;
}

const BOARD_SIZE = 15;
const CELL_SIZE = 36;
const PADDING = 30;
const CANVAS_SIZE = PADDING * 2 + CELL_SIZE * (BOARD_SIZE - 1);

const Board: React.FC<BoardProps> = ({ board, currentPlayer, onMove, disabled, lastMove, boardColors }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
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

  useEffect(() => {
    drawBoard();
  }, [board, lastMove, hoverPos, currentPlayer, disabled, boardColors]);

  /* ======== 鼠标坐标转换 ======== */
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

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled) return;
    const pos = getGridPos(e);
    if (pos && board[pos[0]][pos[1]] === 0) {
      onMove(pos[0], pos[1]);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled) {
      setHoverPos(null);
      return;
    }
    const pos = getGridPos(e);
    setHoverPos(pos);
  };

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
