import { readFileSync } from 'node:fs'
import { expect, it } from 'vitest'

const readThemeFile = (path: string) => readFileSync(`docs/.vitepress/theme/${path}`, 'utf8')

const withoutComments = (source: string) => source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '')

const block = (css: string, selector: string) => {
  const match = css.match(new RegExp(`${selector}\\s*\\{([^}]*)\\}`, 'm'))
  if (!match) throw new Error(`Missing ${selector} token block`)
  return match[1]
}

const declaration = (css: string, name: string) => {
  const match = withoutComments(css).match(new RegExp(`${name.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')}\\s*:\\s*([^;]+);`))
  if (!match) throw new Error(`Missing ${name} declaration`)
  return match[1].trim()
}

const activeThemeBinding = (source: string) => withoutComments(source).match(
  /^\s*import\s+([A-Za-z_$][\w$]*)\s+from\s+['"]vitepress\/theme-without-fonts['"]\s*;?\s*$/m,
)?.[1]

it('does not treat a commented token as a declaration', () => {
  expect(() => declaration('/* --nl-paper: #f4f0e7; */', '--nl-paper')).toThrow('Missing --nl-paper declaration')
})

it('does not treat a comment or string as the active font-free theme import', () => {
  const inactiveThemeReference = [
    "// import CommentedTheme from 'vitepress/theme-without-fonts'",
    "const documentation = 'vitepress/theme-without-fonts'",
    'export default { extends: DefaultTheme }',
  ].join('\n')

  expect(activeThemeBinding(inactiveThemeReference)).toBeUndefined()
})

it('defines the approved research handbook tokens for both colour modes', () => {
  const css = readThemeFile('styles/tokens.css')
  const light = block(css, ':root')
  const dark = block(css, '\\.dark')

  expect(Object.fromEntries([
    ['--nl-paper', declaration(light, '--nl-paper')],
    ['--nl-surface', declaration(light, '--nl-surface')],
    ['--nl-ink', declaration(light, '--nl-ink')],
    ['--nl-muted', declaration(light, '--nl-muted')],
    ['--nl-line', declaration(light, '--nl-line')],
    ['--nl-blue', declaration(light, '--nl-blue')],
    ['--nl-green', declaration(light, '--nl-green')],
    ['--nl-orange', declaration(light, '--nl-orange')],
  ])).toEqual({
    '--nl-paper': '#f4f0e7',
    '--nl-surface': '#fbf9f4',
    '--nl-ink': '#192333',
    '--nl-muted': '#68717d',
    '--nl-line': '#cbc5b9',
    '--nl-blue': '#174ca6',
    '--nl-green': '#247156',
    '--nl-orange': '#bb5e2b',
  })

  expect(Object.fromEntries([
    ['--nl-paper', declaration(dark, '--nl-paper')],
    ['--nl-surface', declaration(dark, '--nl-surface')],
    ['--nl-ink', declaration(dark, '--nl-ink')],
    ['--nl-muted', declaration(dark, '--nl-muted')],
    ['--nl-line', declaration(dark, '--nl-line')],
    ['--nl-blue', declaration(dark, '--nl-blue')],
    ['--nl-green', declaration(dark, '--nl-green')],
    ['--nl-orange', declaration(dark, '--nl-orange')],
  ])).toEqual({
    '--nl-paper': '#171a1e',
    '--nl-surface': '#1d2126',
    '--nl-ink': '#edf0f2',
    '--nl-muted': '#9da6b2',
    '--nl-line': '#3a414a',
    '--nl-blue': '#80aefb',
    '--nl-green': '#70c7a8',
    '--nl-orange': '#f0a36d',
  })
})

it('maps VitePress callouts to handbook semantics in both colour modes', () => {
  const css = readThemeFile('styles/tokens.css')

  for (const tokens of [block(css, ':root'), block(css, '\\.dark')]) {
    expect(declaration(tokens, '--vp-c-tip-1')).toBe('var(--nl-green)')
    expect(declaration(tokens, '--vp-c-tip-soft')).toContain('var(--nl-green)')
    expect(declaration(tokens, '--vp-c-warning-1')).toBe('var(--nl-orange)')
    expect(declaration(tokens, '--vp-c-warning-soft')).toContain('var(--nl-orange)')
    expect(declaration(tokens, '--vp-c-important-1')).toBe('var(--nl-blue)')
    expect(declaration(tokens, '--vp-c-important-soft')).toContain('var(--nl-blue)')
  }
})

it('loads the focused style layers, font-free theme, and accessible focus treatment', () => {
  const imports = [...readThemeFile('custom.css').matchAll(/@import\s+['"]([^'"]+)['"]\s*;/g)]
    .map(([, path]) => path)

  expect(imports).toEqual([
    './styles/tokens.css',
    './styles/base.css',
    './styles/home.css',
    './styles/article.css',
    './styles/components.css',
  ])
  const themeEntry = readThemeFile('index.ts')
  const themeBinding = activeThemeBinding(themeEntry)

  expect(themeBinding).toBe('DefaultTheme')
  expect(withoutComments(themeEntry)).toMatch(new RegExp(`\\bextends\\s*:\\s*${themeBinding}\\b`))
  expect(readThemeFile('styles/base.css')).toMatch(/outline:\s*3px solid var\(--nl-blue\)/)
})

it('does not restore removed purple, emoji feature-card, or image-zoom styling', () => {
  const themeCss = [
    readThemeFile('custom.css'),
    readThemeFile('styles/tokens.css'),
    readThemeFile('styles/base.css'),
    readThemeFile('styles/home.css'),
    readThemeFile('styles/article.css'),
    readThemeFile('styles/components.css'),
  ].join('\n')

  expect(themeCss).not.toMatch(/#(?:bd34fe|41d1ff|6366f1)\b/i)
  expect(themeCss).not.toMatch(/\.VPFeature(?:\s|\.|\{|:)/)
  expect(themeCss).not.toMatch(/\.vp-doc\s+img:hover/)
  expect(themeCss).not.toMatch(/scale\(1\.02\)/)
})

it('keeps Mermaid diagrams in an isolated drag-safe scroll container', () => {
  const articleCss = readThemeFile('styles/article.css')
  const mermaidBlock = block(articleCss, "\\.vp-doc \\.mermaid")
  const svgBlock = block(articleCss, "\\.vp-doc \\.mermaid svg")

  expect(declaration(mermaidBlock, 'overscroll-behavior-inline')).toBe('contain')
  expect(declaration(mermaidBlock, 'contain')).toBe('content')
  expect(declaration(mermaidBlock, 'touch-action')).toBe('pan-x pan-y')
  expect(declaration(mermaidBlock, 'cursor')).toBe('grab')
  expect(declaration(mermaidBlock, 'user-select')).toBe('none')
  expect(declaration(svgBlock, 'width')).toBe('max-content')
  expect(declaration(svgBlock, 'height')).toBe('auto')
  expect(declaration(svgBlock, 'will-change')).toBe('transform')
})

it('gives the homepage and article intro a refined handbook surface treatment', () => {
  const homeCss = readThemeFile('styles/home.css')
  const articleCss = readThemeFile('styles/article.css')
  const heroBlock = block(homeCss, '\\.nl-home-hero')
  const moduleCardBlock = block(homeCss, '\\.nl-module-card')
  const introBlock = block(articleCss, '\\.nl-article-intro')

  expect(declaration(heroBlock, 'position')).toBe('relative')
  expect(declaration(heroBlock, 'border-radius')).toBe('1.1rem')
  expect(declaration(heroBlock, 'background')).toContain('radial-gradient')
  expect(declaration(moduleCardBlock, 'border-radius')).toBe('0.85rem')
  expect(declaration(moduleCardBlock, 'box-shadow')).toContain('color-mix')
  expect(declaration(introBlock, 'border-radius')).toBe('0.9rem')
  expect(declaration(introBlock, 'background')).toContain('linear-gradient')
})
