import React, { useEffect, useRef } from 'react';
import gsap from 'gsap';
import './PoemDisplay.css';

const LEFT_POEMS = [
  { src: '/poem-3.svg', alt: '五子连珠日' },
  { src: '/poem-4.svg', alt: '胜负一瞬间' },
];

const RIGHT_POEMS = [
  { src: '/poem-1.svg', alt: '棋盘方寸间' },
  { src: '/poem-2.svg', alt: '黑白舞翩跹' },
];

const dup = <T,>(arr: T[]): T[] => [...arr, ...arr];

const PoemDisplay: React.FC = () => {
  const trackLeftRef = useRef<HTMLDivElement>(null);
  const trackRightRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const left = trackLeftRef.current;
    const right = trackRightRef.current;

    if (!left || !right) return;

    // 左右对称：起始都距可见区 100%（一份内容的高度）
    // 左：向下滚，-100 → 100，跳回时 -100 与 100 视觉重合
    gsap.set(left, { yPercent: -100 });
    // 右：向上滚，100 → -100，跳回时 100 与 -100 视觉重合
    gsap.set(right, { yPercent: 100 });

    const ctx = gsap.context(() => {
      gsap.to(left, {
        yPercent: 100,
        ease: 'none',
        duration: 36,
        repeat: -1,
      });
      gsap.to(right, {
        yPercent: -100,
        ease: 'none',
        duration: 36,
        repeat: -1,
      });
    });

    return () => ctx.revert();
  }, []);

  return (
    <div className="poem-layer">
      <div className="poem-zone poem-zone-left">
        <div className="poem-track" ref={trackLeftRef}>
          {dup(LEFT_POEMS).map((p, i) => (
            <img key={i} src={p.src} alt={p.alt} className="poem-img" />
          ))}
        </div>
      </div>
      <div className="poem-zone poem-zone-right">
        <div className="poem-track" ref={trackRightRef}>
          {dup(RIGHT_POEMS).map((p, i) => (
            <img key={i} src={p.src} alt={p.alt} className="poem-img" />
          ))}
        </div>
      </div>
    </div>
  );
};

export default PoemDisplay;
