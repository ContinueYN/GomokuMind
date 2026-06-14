/* === 主题系统 ===
 * 4个主题，通过 CSS 变量驱动
 * 通过 <html data-theme="xxx"> 切换
 */

export type ThemeKey = 'spring' | 'autumn' | 'winter' | 'starry';

export interface ThemeInfo {
  key: ThemeKey;
  name: string;
  icon: string;       // 装饰 emoji
}

export const THEMES: ThemeInfo[] = [
  {
    key: 'spring',
    name: '春之呼吸',
    icon: '/icons/spring.ico',
  },
  {
    key: 'autumn',
    name: '秋日郊野',
    icon: '/icons/autumn.ico',
  },
  {
    key: 'winter',
    name: '寒冬低语',
    icon: '/icons/winter.ico',
  },
  {
    key: 'starry',
    name: '繁星守望',
    icon: '/icons/starry.ico',
  },
];

export const DEFAULT_THEME: ThemeKey = 'spring';

/* 棋盘纹理色映射（供 Board 组件 Canvas 使用） */
export interface BoardThemeColors {
  bgStart: string;
  bgEnd: string;
  line: string;
  starPoint: string;
  blackStart: string;
  blackEnd: string;
  whiteStart: string;
  whiteEnd: string;
  blackBorder: string;
  whiteBorder: string;
  lastMove: string;
}

export const BOARD_THEMES: Record<ThemeKey, BoardThemeColors> = {
  spring: {
    bgStart: '#D4E8C2',
    bgEnd: '#B8D4A0',
    line: '#5A7D4A',
    starPoint: '#4A6D3A',
    blackStart: '#5AAA50',
    blackEnd: '#2A5A28',
    whiteStart: '#F5FBF0',
    whiteEnd: '#E8F0D8',
    blackBorder: '#2A5A28',
    whiteBorder: '#8CA878',
    lastMove: '#E8A0BF',
  },
  autumn: {
    bgStart: '#E8D5B0',
    bgEnd: '#D4BA8A',
    line: '#7A5C3A',
    starPoint: '#6A4C2A',
    blackStart: '#E08040',
    blackEnd: '#6B2810',
    whiteStart: '#FFF8E8',
    whiteEnd: '#F5E8C8',
    blackBorder: '#6B2810',
    whiteBorder: '#B89868',
    lastMove: '#E07040',
  },
  winter: {
    bgStart: '#D8E8F0',
    bgEnd: '#C0D8E8',
    line: '#4A6A8A',
    starPoint: '#3A5A7A',
    blackStart: '#5098C8',
    blackEnd: '#1A4868',
    whiteStart: '#F0F8FF',
    whiteEnd: '#E0F0F8',
    blackBorder: '#1A4868',
    whiteBorder: '#88A8C8',
    lastMove: '#F0B0D0',
  },
  starry: {
    bgStart: '#C8C0E0',
    bgEnd: '#B0A0D0',
    line: '#5A4A7A',
    starPoint: '#4A3A6A',
    blackStart: '#8A6AC0',
    blackEnd: '#2A1A50',
    whiteStart: '#F8F0FF',
    whiteEnd: '#E8E0F8',
    blackBorder: '#2A1A50',
    whiteBorder: '#A898C8',
    lastMove: '#F0C0D8',
  },
};
