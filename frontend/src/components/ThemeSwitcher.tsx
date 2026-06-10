import React from 'react';
import { ThemeKey, THEMES } from '../themes';
import './ThemeSwitcher.css';

interface ThemeSwitcherProps {
  current: ThemeKey;
  onChange: (key: ThemeKey) => void;
}

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
