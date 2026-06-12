import React from 'react';
import './PoemDisplay.css';

const LEFT_POEMS = [
  { src: '/poem-3.svg', alt: '五子连珠日' },
  { src: '/poem-4.svg', alt: '胜负一瞬间' },
];

const RIGHT_POEMS = [
  { src: '/poem-1.svg', alt: '棋盘方寸间' },
  { src: '/poem-2.svg', alt: '黑白舞翩跹' },
];

/** 复制一遍内容实现无缝循环 */
const dup = <T,>(arr: T[]): T[] => [...arr, ...arr];

const PoemDisplay: React.FC = () => {
  return (
    <div className="poem-layer">
      {/* 左半区：向下滚动 */}
      <div className="poem-zone poem-zone-left">
        <div className="poem-track poem-scroll-down">
          {dup(LEFT_POEMS).map((p, i) => (
            <img key={i} src={p.src} alt={p.alt} className="poem-img" />
          ))}
        </div>
      </div>

      {/* 右半区：向上滚动 */}
      <div className="poem-zone poem-zone-right">
        <div className="poem-track poem-scroll-up">
          {dup(RIGHT_POEMS).map((p, i) => (
            <img key={i} src={p.src} alt={p.alt} className="poem-img" />
          ))}
        </div>
      </div>
    </div>
  );
};

export default PoemDisplay;
