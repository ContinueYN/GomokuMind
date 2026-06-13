import React from 'react';
import { ThemeKey, THEMES } from '../themes';
import './ThemeSwitcher.css';

/** 父组件传入的属性 */
interface ThemeSwitcherProps {
  current: ThemeKey;                         // 当前激活的主题 key
  onChange: (key: ThemeKey) => void;          // 切换主题回调
}

/**
 * 主题切换器
 *
 * 遍历 THEMES 配置渲染按钮组。
 * 当前激活的主题按钮高亮显示。
 * 点击按钮时回调 onChange 通知父组件更新主题状态。
 */
const ThemeSwitcher: React.FC<ThemeSwitcherProps> = ({ current, onChange }) => {
  return (
    <div className="theme-switcher">
      <span className="theme-label">主题切换</span>
      <div className="theme-options">
        {THEMES.map(t => (
          <button
            key={t.key}
            className={`theme-btn theme-${t.key} ${current === t.key ? 'active' : ''}`}
            onClick={() => onChange(t.key)}
            title={t.name}
          >
            <span className="theme-icon">{t.icon}</span>
            <span className="theme-name">{t.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default ThemeSwitcher;
