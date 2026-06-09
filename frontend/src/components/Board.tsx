import React, { useRef, useEffect, useState } from 'react';
import { Piece, Move } from '../types';
import './Board.css';

interface BoardProps {
  board: Piece[][];
  currentPlayer: Piece;
  onMove: (row: number, col: number) => void;
  disabled: boolean;
  lastMove: Move | null;
}

const BOARD_SIZE = 15;
const CELL_SIZE = 36;
const PADDING = 30;
const CANVAS_SIZE = PADDING * 2 + CELL_SIZE * (BOARD_SIZE - 1);

const Board: React.FC<BoardProps> = ({ board, currentPlayer, onMove, disabled, lastMove }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoverPos, setHoverPos] = useState<[number, number] | null>(null);

  const drawBoard = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Background - wood texture
    const gradient = ctx.createLinearGradient(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    gradient.addColorStop(0, '#DEB887');
    gradient.addColorStop(1, '#D2A86E');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Grid lines
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
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

    // Star points (天元和星位)
    const starPoints = [[3, 3], [3, 11], [7, 7], [11, 3], [11, 11], [3, 7], [7, 3], [7, 11], [11, 7]];
    ctx.fillStyle = '#333';
    for (const [r, c] of starPoints) {
      ctx.beginPath();
      ctx.arc(PADDING + c * CELL_SIZE, PADDING + r * CELL_SIZE, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Pieces
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (board[r][c] !== 0) {
          const x = PADDING + c * CELL_SIZE;
          const y = PADDING + r * CELL_SIZE;
          
          // Shadow
          ctx.beginPath();
          ctx.arc(x + 2, y + 2, CELL_SIZE * 0.42, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
          ctx.fill();

          // Piece
          ctx.beginPath();
          ctx.arc(x, y, CELL_SIZE * 0.42, 0, Math.PI * 2);
          
          if (board[r][c] === 1) {
            // Black piece
            const grad = ctx.createRadialGradient(x - 5, y - 5, 0, x, y, CELL_SIZE * 0.42);
            grad.addColorStop(0, '#555');
            grad.addColorStop(1, '#111');
            ctx.fillStyle = grad;
          } else {
            // White piece
            const grad = ctx.createRadialGradient(x - 5, y - 5, 0, x, y, CELL_SIZE * 0.42);
            grad.addColorStop(0, '#fff');
            grad.addColorStop(1, '#ddd');
            ctx.fillStyle = grad;
          }
          ctx.fill();
          ctx.strokeStyle = board[r][c] === 1 ? '#000' : '#999';
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }

    // Last move indicator
    if (lastMove) {
      const x = PADDING + lastMove.col * CELL_SIZE;
      const y = PADDING + lastMove.row * CELL_SIZE;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#e2b04a';
      ctx.fill();
    }

    // Hover preview
    if (hoverPos && !disabled) {
      const [hr, hc] = hoverPos;
      if (board[hr][hc] === 0) {
        const x = PADDING + hc * CELL_SIZE;
        const y = PADDING + hr * CELL_SIZE;
        ctx.beginPath();
        ctx.arc(x, y, CELL_SIZE * 0.42, 0, Math.PI * 2);
        ctx.fillStyle = currentPlayer === 1 ? 'rgba(0, 0, 0, 0.3)' : 'rgba(255, 255, 255, 0.5)';
        ctx.fill();
      }
    }
  };

  useEffect(() => {
    drawBoard();
  }, [board, lastMove, hoverPos, currentPlayer, disabled]);

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
