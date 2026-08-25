# Notes on LLMs Knowledge Site Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 VitePress 博客升级为具有知识地图、学习路径、内容状态和可靠 GitHub Pages 部署的 LLM 研究手册，并完成首期入口内容更新。

**Architecture:** 保留 VitePress 1.x、Vue 3、Markdown 和 GitHub Pages，使用 Frontmatter 内容契约与 VitePress 构建时数据加载器生成内容索引。主题通过默认布局插槽扩展，首页、文章框架、SEO 和导航消费同一份内容数据，不引入运行时后端。

**Tech Stack:** VitePress 1.6.4、Vue 3.5、TypeScript、Vitest、gray-matter、Mermaid、GitHub Actions、GitHub Pages

**Spec:** `docs/superpowers/specs/2026-08-24-llm-knowledge-site-redesign-design.md`

## Global Constraints

- 保留 `base: '/notes-on-llms/'`；所有站内链接必须在该子路径下工作。
- 保留静态站架构；不增加数据库、CMS、登录、评论、服务端搜索或运行时 API。
- 视觉固定为“研究手册 + 知识地图”：暖纸色、深墨色、钴蓝强调、绿色核验状态、橙色实践提示。
- 首期只重写首页、学习路径、可见入口页和六个模块概览；深层文章只补元数据和状态，不改写正文。
- 技术事实只使用官方文档、标准、原始论文或官方项目仓库核验；无法确认时使用 `needs-review`。
- AI 改写必须保留作者经验、示例和代码；每篇 Markdown 单独审查 diff。
- `draft` 不得进入生产构建、导航、搜索、最近更新或 sitemap。
- 最终生产构建、内容检查、站内链接检查和浏览器验收全部通过；Lighthouse Performance、Accessibility、SEO 均达到 90 分以上。

## Planned File Structure

```text
docs/.vitepress/
├── config.ts
├── config/{site,modules,nav,sidebar,head,mermaid}.ts
├── content/{model,content.data,selectors}.ts
└── theme/
    ├── index.ts
    ├── Layout.vue
    ├── components/{article,home}/
    └── styles/{tokens,base,home,article,components}.css
scripts/{check-content,migrate-frontmatter,check-built-site}.ts
tests/{content,config}/
```

---

### Task 1: Establish the content contract and test harness

**Files:**
- Modify: `package.json`
- Modify: `pnpm-lock.yaml`
- Create: `docs/.vitepress/content/model.ts`
- Create: `tests/content/model.test.ts`

**Interfaces:**
- Produces: `ContentFrontmatter`, `ContentIndexItem`, `assertValidFrontmatter(frontmatter, relativePath)`, `isDraft(frontmatter)`.
- Consumes: no project code.

- [ ] **Step 1: Add test dependencies and scripts**

Run:

```bash
pnpm add -D gray-matter@4.0.3 vitest@3.2.4 tsx@4.20.5
```

Add to `package.json`:

```json
{
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest",
    "content:check": "tsx scripts/check-content.ts"
  }
}
```

- [ ] **Step 2: Write the failing model tests**

Create `tests/content/model.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { assertValidFrontmatter, isDraft } from '../../docs/.vitepress/content/model'

const article = {
  title: 'RAG 检索策略', description: '混合检索与重排序', pageType: 'article', module: 'rag',
  level: 'intermediate', prerequisites: ['/llms/rag/embedding'], updated: '2026-08-25',
  reviewed: '2026-08-25', contentStatus: 'verified', techVersion: '2025–2026', tags: ['rag']
}

describe('assertValidFrontmatter', () => {
  it('accepts a complete article', () => {
    expect(() => assertValidFrontmatter(article, 'llms/rag/retrieval.md')).not.toThrow()
  })
  it('rejects an article without reviewed', () => {
    expect(() => assertValidFrontmatter({ ...article, reviewed: undefined }, 'llms/rag/retrieval.md'))
      .toThrow('llms/rag/retrieval.md: reviewed is required for a published article')
  })
  it('allows landing pages without article-only fields', () => {
    expect(() => assertValidFrontmatter({
      title: '关于本站', description: '内容原则', pageType: 'landing', module: 'site',
      updated: '2026-08-25', contentStatus: 'verified', tags: ['about']
    }, 'about/index.md')).not.toThrow()
  })
})

it('identifies drafts', () => expect(isDraft({ contentStatus: 'draft' })).toBe(true))
```

- [ ] **Step 3: Verify the test fails**

Run: `pnpm test -- tests/content/model.test.ts`

Expected: FAIL because `model.ts` does not exist.

- [ ] **Step 4: Implement the model and conditional validation**

```ts
export const PAGE_TYPES = ['article', 'path', 'landing'] as const
export const MODULES = ['rag', 'agent', 'mcp', 'prompt', 'training', 'multimodal', 'site'] as const
export const LEVELS = ['beginner', 'intermediate', 'advanced'] as const
export const CONTENT_STATUSES = ['verified', 'needs-review', 'opinion', 'historical', 'draft'] as const

export type PageType = typeof PAGE_TYPES[number]
export type ModuleKey = typeof MODULES[number]
export type ContentStatus = typeof CONTENT_STATUSES[number]

export interface ContentFrontmatter {
  title: string; description: string; pageType: PageType; module: ModuleKey
  level?: typeof LEVELS[number]; prerequisites?: string[]; updated: string; reviewed?: string
  contentStatus: ContentStatus; techVersion?: string; tags: string[]; author?: string; order?: number
}

export interface ContentIndexItem extends ContentFrontmatter {
  url: string; readingTime: number; sourceCount: number
}

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/
export function isDraft(value: Pick<ContentFrontmatter, 'contentStatus'> | Record<string, unknown>): boolean {
  return value.contentStatus === 'draft'
}

export function assertValidFrontmatter(value: Record<string, unknown>, path: string): asserts value is ContentFrontmatter {
  const fail = (message: string): never => { throw new Error(`${path}: ${message}`) }
  if (!value.title || typeof value.title !== 'string') fail('title is required')
  if (!value.description || typeof value.description !== 'string') fail('description is required')
  if (!PAGE_TYPES.includes(value.pageType as PageType)) fail('pageType is invalid')
  if (!MODULES.includes(value.module as ModuleKey)) fail('module is invalid')
  if (!value.updated || !ISO_DATE.test(String(value.updated))) fail('updated must use YYYY-MM-DD')
  if (!CONTENT_STATUSES.includes(value.contentStatus as ContentStatus)) fail('contentStatus is invalid')
  if (!Array.isArray(value.tags)) fail('tags must be an array')
  if (value.pageType === 'article' && value.contentStatus !== 'draft') {
    if (!LEVELS.includes(value.level as typeof LEVELS[number])) fail('level is required for an article')
    if (!Array.isArray(value.prerequisites)) fail('prerequisites is required for an article')
    if (!value.reviewed || !ISO_DATE.test(String(value.reviewed))) fail('reviewed is required for a published article')
    if (!value.techVersion || typeof value.techVersion !== 'string') fail('techVersion is required for an article')
  }
}
```

- [ ] **Step 5: Run tests and commit**

Run: `pnpm test -- tests/content/model.test.ts`

Expected: PASS.

```bash
git add package.json pnpm-lock.yaml docs/.vitepress/content/model.ts tests/content/model.test.ts
git commit -m 'test: define content metadata contract'
```

---

### Task 2: Add content checks, migrate metadata, and remove base-unsafe links

**Files:**
- Create: `scripts/check-content.ts`
- Create: `scripts/migrate-frontmatter.ts`
- Create: `tests/content/check-content.test.ts`
- Modify: `docs/.vitepress/config.ts`
- Modify: all public Markdown under `docs/llms/`, `docs/guide/`, `docs/interviews/`, `docs/reference/`, `docs/resources/`, `docs/about/`, plus `docs/index.md`
- Modify base-unsafe links in: `docs/llms/index.md`, all module `index.md` except multimodal, `docs/guide/roadmap.md`, `docs/guide/prerequisites.md`

**Interfaces:**
- Consumes: `assertValidFrontmatter()`.
- Produces: `collectPublicMarkdown(root)`, `findBaseUnsafeLinks(source)`, `checkContent(root)` and `pnpm content:check`.

- [ ] **Step 1: Write failing checker tests**

```ts
import { expect, it } from 'vitest'
import { findBaseUnsafeLinks } from '../../scripts/check-content'

it('finds raw HTML links that bypass base', () => {
  expect(findBaseUnsafeLinks('<a href="/llms/rag/">RAG</a>')).toEqual(['/llms/rag/'])
  expect(findBaseUnsafeLinks('[RAG](/llms/rag/)')).toEqual([])
})
```

Add a temporary-directory test proving `checkContent()` rejects `contentStatus: draft` outside `_drafts/`.

- [ ] **Step 2: Verify the test fails**

Run: `pnpm test -- tests/content/check-content.test.ts`

Expected: FAIL because `check-content.ts` does not exist.

- [ ] **Step 3: Implement the checker**

```ts
import { readdir, readFile } from 'node:fs/promises'
import { relative, resolve } from 'node:path'
import matter from 'gray-matter'
import { assertValidFrontmatter } from '../docs/.vitepress/content/model'

export const EXCLUDED_PATHS = ['.vitepress/', 'public/', 'superpowers/', '_drafts/', '开发计划.md']

export function findBaseUnsafeLinks(source: string): string[] {
  return [...source.matchAll(/href=["'](\/(?!\/)[^"']*)["']/g)].map(match => match[1])
}

export async function collectPublicMarkdown(root: string): Promise<string[]> {
  const files: string[] = []
  async function walk(directory: string): Promise<void> {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const absolute = resolve(directory, entry.name)
      const path = relative(root, absolute).replaceAll('\\', '/')
      if (EXCLUDED_PATHS.some(excluded => `${path}${entry.isDirectory() ? '/' : ''}`.startsWith(excluded))) continue
      if (entry.isDirectory()) await walk(absolute)
      else if (entry.isFile() && entry.name.endsWith('.md')) files.push(absolute)
    }
  }
  await walk(root)
  return files.sort()
}

export async function checkContent(root: string): Promise<void> {
  const errors: string[] = []
  for (const file of await collectPublicMarkdown(root)) {
    const path = relative(root, file).replaceAll('\\', '/')
    const source = await readFile(file, 'utf8')
    const { data } = matter(source)
    try { assertValidFrontmatter(data, path) } catch (error) { errors.push((error as Error).message) }
    for (const href of findBaseUnsafeLinks(source)) errors.push(`${path}: raw HTML href bypasses base: ${href}`)
    if (data.contentStatus === 'draft' && !path.startsWith('_drafts/') && !path.endsWith('.draft.md')) {
      errors.push(`${path}: draft pages must live under _drafts or use .draft.md`)
    }
  }
  if (errors.length) throw new Error(errors.join('\n'))
}
```

The CLI calls `checkContent(resolve('docs'))`, prints the checked count, and sets `process.exitCode = 1` on failure. Guard CLI execution so tests can import the functions.

- [ ] **Step 4: Implement and run the idempotent migration**

`migrate-frontmatter.ts` preserves Markdown bodies and applies these path rules:

```ts
const MODULE_DEFAULTS = {
  prompt: { level: 'beginner', prerequisites: [], techVersion: '待复核（2026-08）' },
  rag: { level: 'intermediate', prerequisites: ['/llms/prompt/'], techVersion: '待复核（2026-08）' },
  agent: { level: 'advanced', prerequisites: ['/llms/prompt/', '/llms/rag/'], techVersion: '待复核（2026-08）' },
  mcp: { level: 'intermediate', prerequisites: ['/llms/agent/tool-calling'], techVersion: '待复核（2026-08）' },
  training: { level: 'advanced', prerequisites: ['/guide/prerequisites'], techVersion: '待复核（2026-08）' },
  multimodal: { level: 'advanced', prerequisites: ['/guide/prerequisites'], techVersion: '待复核（2026-08）' }
} as const
```

Rules: module Markdown becomes `article`; `guide/roadmap.md` becomes `path`; home, about, resources and `llms/index.md` become `landing`; other guide/interview/reference pages become `article` with `module: site`. Existing deep pages start as `needs-review`. `updated` comes from `git log -1 --format=%cs -- <file>` and `reviewed` is `2026-08-25` for articles. Existing title and description win.

Run twice:

```bash
pnpm tsx scripts/migrate-frontmatter.ts
git diff -- docs
pnpm tsx scripts/migrate-frontmatter.ts
git diff --check
```

Expected: second run produces no new diff. Review every metadata block before continuing.

- [ ] **Step 5: Exclude drafts and internal documents**

Add environment-aware exclusions to `defineConfig`:

```ts
srcExclude: [
  'superpowers/**',
  '开发计划.md',
  ...(process.env.NODE_ENV === 'production' ? ['_drafts/**', '**/*.draft.md'] : [])
],
```

This keeps drafts available during local development while guaranteeing that production does not emit them.

- [ ] **Step 6: Replace raw root HTML links**

Module indexes use `./article-slug`, `docs/llms/index.md` uses `./rag/` and sibling paths, and guide pages use `../llms/<module>/`. Markdown links remain unchanged.

Run: `rg -n 'href=["'"']/[^/]' docs -g '*.md' -g '!superpowers/**'`

Expected: no matches.

- [ ] **Step 7: Validate and commit**

```bash
pnpm test -- tests/content
pnpm content:check
pnpm docs:build
git add package.json scripts docs tests/content
git commit -m 'feat: validate and migrate content metadata'
```

---

### Task 3: Build the derived content index and selectors

**Files:**
- Create: `docs/.vitepress/content/content.data.ts`
- Create: `docs/.vitepress/content/selectors.ts`
- Create: `tests/content/selectors.test.ts`

**Interfaces:**
- Produces: loader `data: ContentIndexItem[]`, `selectRecentUpdates(items, limit)`, `findAdjacentArticle(items, url)`, `normalizeUrl(url)`.
- Consumes: Task 1 content model.

- [ ] **Step 1: Write failing selector tests**

```ts
expect(selectRecentUpdates(pages, 2).map(page => page.title)).toEqual(['检索', 'RAG'])
expect(findAdjacentArticle(pages, '/llms/rag/').next?.url).toBe('/llms/rag/retrieval')
expect(findAdjacentArticle(pages, '/llms/rag')).toEqual(findAdjacentArticle(pages, '/llms/rag/'))
```

Use fixtures with dates, module order and one draft.

- [ ] **Step 2: Verify failure and implement selectors**

Run: `pnpm test -- tests/content/selectors.test.ts`

Expected: FAIL because selectors do not exist.

```ts
export const normalizeUrl = (url: string) => url === '/' ? '/' : `/${url.replace(/^\//, '').replace(/\/$/, '')}`

export function selectRecentUpdates(items: ContentIndexItem[], limit = 4) {
  return items.filter(item => item.contentStatus !== 'draft')
    .sort((a, b) => b.updated.localeCompare(a.updated) || a.title.localeCompare(b.title, 'zh-CN')).slice(0, limit)
}
```

`findAdjacentArticle()` normalizes URLs, filters to the current module, excludes drafts and sorts defined `order` values.

- [ ] **Step 3: Implement the build-time loader**

```ts
import { createContentLoader } from 'vitepress'
import { assertValidFrontmatter, isDraft, type ContentIndexItem } from './model'

declare const data: ContentIndexItem[]
export { data }

export default createContentLoader('**/*.md', {
  includeSrc: true,
  transform(raw): ContentIndexItem[] {
    return raw.filter(page => !page.url.startsWith('/superpowers/') && page.url !== '/开发计划')
      .map(page => {
        assertValidFrontmatter(page.frontmatter, page.url)
        const source = page.src ?? ''
        const words = source.replace(/^---[\s\S]*?---/, '').split(/\s+|(?=[\u4e00-\u9fff])/).filter(Boolean).length
        const refs = source.match(/^##\s+参考资料[\s\S]*$/m)?.[0] ?? ''
        return { ...page.frontmatter, url: page.url.replace(/\.html$/, '').replace(/\/index$/, '/'),
          readingTime: Math.max(1, Math.ceil(words / 350)), sourceCount: (refs.match(/^\s*[-*]\s+/gm) ?? []).length } as ContentIndexItem
      }).filter(page => !isDraft(page))
  }
})
```

- [ ] **Step 4: Validate and commit**

```bash
pnpm test -- tests/content/selectors.test.ts
pnpm docs:build
git add docs/.vitepress/content tests/content/selectors.test.ts
git commit -m 'feat: generate the build-time content index'
```

---

### Task 4: Split configuration and add page-level SEO

**Files:**
- Modify: `docs/.vitepress/config.ts`
- Create: `docs/.vitepress/config/site.ts`
- Create: `docs/.vitepress/config/modules.ts`
- Create: `docs/.vitepress/config/nav.ts`
- Create: `docs/.vitepress/config/sidebar.ts`
- Create: `docs/.vitepress/config/head.ts`
- Create: `docs/.vitepress/config/mermaid.ts`
- Modify: `docs/.vitepress/content/content.data.ts`
- Create: `tests/config/head.test.ts`

**Interfaces:**
- Produces: `SITE_ORIGIN`, `SITE_BASE`, `SITE_URL`, `MODULE_DEFINITIONS`, `buildPageHead(context)`.
- Consumes: Task 1 Frontmatter.

- [ ] **Step 1: Write the failing SEO test**

```ts
const head = buildPageHead({ page: 'llms/rag/index.md', title: 'RAG 技术全景', description: 'RAG 学习入口',
  frontmatter: { pageType: 'article', author: 'likebeans', updated: '2026-08-25', reviewed: '2026-08-25' } })
expect(head).toContainEqual(['link', { rel: 'canonical', href: 'https://likebeans.github.io/notes-on-llms/llms/rag/' }])
expect(JSON.stringify(head)).toContain('application/ld+json')
expect(JSON.stringify(head)).toContain('og:title')
```

Run: `pnpm test -- tests/config/head.test.ts`

Expected: FAIL because `head.ts` does not exist.

- [ ] **Step 2: Define site and module constants**

```ts
export const SITE_ORIGIN = 'https://likebeans.github.io'
export const SITE_BASE = '/notes-on-llms/'
export const SITE_URL = `${SITE_ORIGIN}${SITE_BASE}`
export const SITE_AUTHOR = 'likebeans'
export const SITE_TITLE = 'Notes on LLMs'
export const SITE_DESCRIPTION = '从模型原理到可运行智能系统的大模型研究手册'
```

`MODULE_DEFINITIONS` contains six keys, each with `title`, `shortDescription`, `path`, and ordered `items: Array<{ text: string; link: string }>` copied from the existing sidebar. It becomes the single source for sidebar, progress and next-step order. Export:

```ts
import type { ModuleKey } from '../content/model'
import { normalizeUrl } from '../content/selectors'

export function findModuleOrder(module: ModuleKey, url: string): number | undefined {
  const definition = MODULE_DEFINITIONS[module as keyof typeof MODULE_DEFINITIONS]
  const index = definition?.items.findIndex(item => normalizeUrl(item.link) === normalizeUrl(url)) ?? -1
  return index < 0 ? undefined : index
}
```

Update `content.data.ts` so every item sets `order: frontmatter.order ?? findModuleOrder(frontmatter.module, url)`. This enables progress and next-step links without duplicating order in Markdown.

- [ ] **Step 3: Split nav, sidebar and Mermaid configuration**

`nav.ts` exports 学习路径 `/guide/`, 知识体系 `/llms/`, 实践 `/practice/`, 关于 `/about/`. `sidebar.ts` derives module groups from `MODULE_DEFINITIONS` and retains interview/reference/resource/about groups. Move existing Mermaid options unchanged to `mermaid.ts`. Reduce `config.ts` to composition.

- [ ] **Step 4: Implement base-aware head entries**

`buildPageHead()` always returns canonical, `og:type`, `og:title`, `og:description`, `og:url`, `og:image`, `twitter:card`, `twitter:title`, and `twitter:description`. Use `property` for Open Graph. Articles also receive JSON-LD with `@type: Article`, headline, author, dateModified, lastReviewed, URL and mainEntityOfPage.

- [ ] **Step 5: Wire sitemap and dynamic head**

```ts
sitemap: { hostname: SITE_URL, transformItems: items => items.filter(item => !item.url.includes('/_drafts/') && !item.url.includes('/superpowers/')) },
transformHead({ page, title, description, pageData }) {
  return page === '404.md' ? [] : buildPageHead({ page, title, description, frontmatter: pageData.frontmatter })
},
head: [
  ['link', { rel: 'icon', type: 'image/svg+xml', href: `${SITE_BASE}logo.svg` }],
  ['meta', { name: 'theme-color', content: '#174ca6' }]
]
```

- [ ] **Step 6: Validate and commit**

```bash
pnpm test -- tests/config/head.test.ts
pnpm content:check
pnpm docs:build
git add docs/.vitepress/config.ts docs/.vitepress/config tests/config
git commit -m 'feat: split site config and add page SEO'
```

---

### Task 5: Implement the research-handbook visual foundation

**Files:**
- Modify: `docs/.vitepress/theme/index.ts`
- Replace: `docs/.vitepress/theme/custom.css`
- Create: `docs/.vitepress/theme/styles/tokens.css`
- Create: `docs/.vitepress/theme/styles/base.css`
- Create: `docs/.vitepress/theme/styles/home.css`
- Create: `docs/.vitepress/theme/styles/article.css`
- Create: `docs/.vitepress/theme/styles/components.css`
- Create: `tests/config/theme-contract.test.ts`

**Interfaces:**
- Produces semantic CSS tokens and responsive layout classes used by Tasks 6 and 7.

- [ ] **Step 1: Write a failing token contract test**

```ts
import { readFileSync } from 'node:fs'
import { expect, it } from 'vitest'

it('defines the approved research handbook tokens', () => {
  const css = readFileSync('docs/.vitepress/theme/styles/tokens.css', 'utf8')
  for (const token of ['--nl-paper', '--nl-ink', '--nl-blue', '--nl-green', '--nl-orange', '--nl-line']) {
    expect(css).toContain(token)
  }
})
```

Run: `pnpm test -- tests/config/theme-contract.test.ts`

Expected: FAIL because the style files do not exist.

- [ ] **Step 2: Create semantic light and dark tokens**

```css
:root {
  --nl-paper: #f4f0e7; --nl-surface: #fbf9f4; --nl-ink: #192333; --nl-muted: #68717d;
  --nl-line: #cbc5b9; --nl-blue: #174ca6; --nl-green: #247156; --nl-orange: #bb5e2b;
  --vp-font-family-base: Inter, ui-sans-serif, system-ui, sans-serif;
  --vp-font-family-mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  --vp-c-brand-1: var(--nl-blue); --vp-c-bg: var(--nl-surface);
}
.dark {
  --nl-paper: #171a1e; --nl-surface: #1d2126; --nl-ink: #edf0f2; --nl-muted: #9da6b2;
  --nl-line: #3a414a; --nl-blue: #80aefb; --nl-green: #70c7a8; --nl-orange: #f0a36d;
}
```

- [ ] **Step 3: Replace the legacy stylesheet with focused files**

`custom.css` becomes:

```css
@import './styles/tokens.css';
@import './styles/base.css';
@import './styles/home.css';
@import './styles/article.css';
@import './styles/components.css';
```

Port only styles still used by Markdown: tables, code overflow, Mermaid and custom cards. Remove purple hero gradients, emoji feature-card styling and article-image hover zoom. Add 620px and 900px breakpoints and `prefers-reduced-motion` handling.

- [ ] **Step 4: Use the font-free default theme**

```ts
import DefaultTheme from 'vitepress/theme-without-fonts'
import type { Theme } from 'vitepress'
import './custom.css'

export default { extends: DefaultTheme } satisfies Theme
```

- [ ] **Step 5: Validate and commit**

```bash
pnpm test -- tests/config/theme-contract.test.ts
pnpm docs:build
git add docs/.vitepress/theme tests/config/theme-contract.test.ts
git commit -m 'style: add research handbook design system'
```

---

### Task 6: Implement the article learning shell

**Files:**
- Modify: `docs/.vitepress/theme/index.ts`
- Create: `docs/.vitepress/theme/Layout.vue`
- Create: `docs/.vitepress/theme/components/article/ArticleIntro.vue`
- Create: `docs/.vitepress/theme/components/article/ArticleMeta.vue`
- Create: `docs/.vitepress/theme/components/article/LearningObjectives.vue`
- Create: `docs/.vitepress/theme/components/article/ContentStatus.vue`
- Create: `docs/.vitepress/theme/components/article/DraftNotice.vue`
- Create: `docs/.vitepress/theme/components/article/SourceList.vue`
- Create: `docs/.vitepress/theme/components/article/NextStep.vue`
- Create: `docs/.vitepress/theme/components/article/ModuleProgress.vue`
- Create: `docs/.vitepress/theme/components/article/ReadingProgress.vue`
- Create: `tests/content/article-navigation.test.ts`

**Interfaces:**
- Consumes: content loader data, selectors, `MODULE_DEFINITIONS`, `useData()` and `useRoute()`.
- Produces: default-theme wrapper plus global `LearningObjectives` and `SourceList` authoring components.

- [ ] **Step 1: Add failing route-normalization tests**

```ts
expect(findAdjacentArticle(pages, '/llms/rag')).toEqual(findAdjacentArticle(pages, '/llms/rag/'))
expect(findAdjacentArticle([{ ...pages[0], order: undefined }], '/llms/rag/').next).toBeUndefined()
```

Run: `pnpm test -- tests/content/article-navigation.test.ts`

Expected: FAIL until `normalizeUrl()` is used for current and candidate URLs.

- [ ] **Step 2: Pass the selector tests**

```ts
export const normalizeUrl = (url: string) => url === '/' ? '/' : `/${url.replace(/^\//, '').replace(/\/$/, '')}`
```

Use this helper in `findAdjacentArticle()`.

- [ ] **Step 3: Implement exact component contracts**

- `ArticleIntro({ page: ContentIndexItem })`: module label, status, one H1, description and `ArticleMeta`.
- `ArticleMeta({ level, readingTime, prerequisites, techVersion })`.
- `ContentStatus({ status, reviewed, sourceCount })`; always renders text as well as color.
- `DraftNotice()` renders a visible `草稿：不会进入生产发布` warning from page frontmatter.
- `LearningObjectives({ items: string[] })`.
- `SourceList({ items: Array<{ title: string; href: string; note?: string }> })`; external links use `rel='noreferrer'`.
- `NextStep({ previous?: ContentIndexItem, next?: ContentIndexItem })`.
- `ModuleProgress({ module: ModuleKey, currentUrl: string })`.
- `ReadingProgress()` updates one CSS custom property from scroll and removes listeners on unmount.

- [ ] **Step 4: Compose official default-theme slots**

```vue
<script setup lang='ts'>
import DefaultTheme from 'vitepress/theme-without-fonts'
import { computed } from 'vue'
import { useData, useRoute } from 'vitepress'
import { data as content } from '../content/content.data'
import { findAdjacentArticle, normalizeUrl } from '../content/selectors'
import ArticleIntro from './components/article/ArticleIntro.vue'
import ContentStatus from './components/article/ContentStatus.vue'
import DraftNotice from './components/article/DraftNotice.vue'
import ModuleProgress from './components/article/ModuleProgress.vue'
import NextStep from './components/article/NextStep.vue'
import ReadingProgress from './components/article/ReadingProgress.vue'
const { Layout } = DefaultTheme
const { frontmatter } = useData()
const route = useRoute()
const page = computed(() => content.find(item => normalizeUrl(item.url) === normalizeUrl(route.path)))
const adjacent = computed(() => findAdjacentArticle(content, route.path))
</script>

<template>
  <div :class="{ 'nl-article-page': frontmatter.pageType === 'article' }">
    <Layout>
      <template #doc-before>
        <DraftNotice v-if='frontmatter.contentStatus === "draft"' />
        <ArticleIntro v-if='page && frontmatter.pageType === "article"' :page='page' />
      </template>
      <template #sidebar-nav-before><ModuleProgress v-if='page?.module && page.module !== "site"' :module='page.module' :current-url='route.path' /></template>
      <template #aside-outline-before><ReadingProgress v-if='frontmatter.pageType === "article"' /></template>
      <template #doc-after>
        <ContentStatus v-if='page && frontmatter.pageType === "article"' :status='page.contentStatus' :reviewed='page.reviewed' :source-count='page.sourceCount' />
        <NextStep v-if='frontmatter.pageType === "article"' v-bind='adjacent' />
      </template>
    </Layout>
  </div>
</template>
```

Hide only the original first Markdown H1 on `.nl-article-page`; `ArticleIntro` is the single visible H1. Do not hide H2/H3 outline headings.

- [ ] **Step 5: Register layout and authoring components**

```ts
export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('LearningObjectives', LearningObjectives)
    app.component('SourceList', SourceList)
  }
} satisfies Theme
```

- [ ] **Step 6: Validate representative desktop and mobile articles**

```bash
pnpm test -- tests/content/article-navigation.test.ts
pnpm docs:build
pnpm docs:dev --host 127.0.0.1
```

Inspect `/notes-on-llms/llms/rag/retrieval` at 1440 and 360 pixels. Expected: one H1, no page overflow, keyboard-accessible links and textual content status.

- [ ] **Step 7: Commit**

```bash
git add docs/.vitepress/theme docs/.vitepress/content/selectors.ts tests/content/article-navigation.test.ts
git commit -m 'feat: add the article learning shell'
```

---

### Task 7: Build the knowledge-map homepage

**Files:**
- Replace: `docs/index.md`
- Create: `docs/.vitepress/theme/components/home/HomePage.vue`
- Create: `docs/.vitepress/theme/components/home/HomeKnowledgeMap.vue`
- Create: `docs/.vitepress/theme/components/home/LearningPaths.vue`
- Create: `docs/.vitepress/theme/components/home/RecentUpdates.vue`
- Create: `docs/.vitepress/theme/components/home/SiteIntroduction.vue`
- Create: `tests/content/home-selectors.test.ts`

**Interfaces:**
- Consumes: content data, `selectRecentUpdates()`, `MODULE_DEFINITIONS`, `withBase()`.
- Produces: custom homepage matching approved direction A.

- [ ] **Step 1: Add and pass deterministic recent-update tests**

The failing test uses equal dates and expects title sorting:

```ts
expect(selectRecentUpdates(tiedPages, 4).map(page => page.title)).toEqual(['Agent', 'RAG'])
```

After date comparison, add `a.title.localeCompare(b.title, 'zh-CN')`. Run `pnpm test -- tests/content/home-selectors.test.ts` and expect PASS.

- [ ] **Step 2: Implement homepage components**

- `HomeKnowledgeMap` renders six modules from `MODULE_DEFINITIONS`; every URL passes through `withBase()`.
- `LearningPaths` renders 应用开发者、算法工程师、面试冲刺 and anchors in `/guide/`.
- `RecentUpdates` displays four indexed pages with date, module, title and status text.
- `SiteIntroduction` explains author perspective, review policy and GitHub source.
- `HomePage` composes hero, knowledge map, paths, recent updates and introduction.

- [ ] **Step 3: Replace the default VitePress home**

```md
---
layout: page
title: Notes on LLMs
description: 从模型原理到可运行智能系统的大模型研究手册
pageType: landing
module: site
updated: 2026-08-25
contentStatus: verified
tags: [llm, knowledge-map, learning-path]
sidebar: false
aside: false
---

<HomePage />
```

Register `HomePage` globally.

- [ ] **Step 4: Validate responsive output and commit**

```bash
pnpm content:check
pnpm docs:build
git add docs/index.md docs/.vitepress/theme tests/content/home-selectors.test.ts
git commit -m 'feat: build the knowledge map homepage'
```

At 1440, 768 and 360 pixels all six modules and three paths must remain readable. The handwritten 2024 update table must be absent.

---

### Task 8: Rewrite learning paths and every public landing page

**Files:**
- Create: `docs/guide/index.md`
- Modify: `docs/guide/roadmap.md`
- Modify: `docs/guide/prerequisites.md`
- Modify: `docs/llms/index.md`
- Create: `docs/practice/index.md`
- Modify: `docs/interviews/index.md`
- Create: `docs/resources/index.md`
- Modify: `docs/resources/{papers,videos,blogs,repos}.md`
- Create: `docs/reference/index.md`
- Modify: `docs/reference/{glossary,checklists,metrics,templates}.md`
- Modify: `docs/about/{index,changelog}.md`
- Modify: `docs/.vitepress/config/{nav,sidebar}.ts`

**Interfaces:**
- Consumes metadata contract and global authoring components.
- Produces complete destinations for every visible navigation entry.

- [ ] **Step 1: Record the existing placeholder matches**

Run:

```bash
rg -n '^> 待补充' docs/about docs/resources docs/reference docs/interviews
```

Expected: current placeholder pages match.

- [ ] **Step 2: Create the three learning paths**

`guide/index.md` contains anchored sections for:

1. 应用开发者：Prompt → RAG → Agent → MCP → 生产实践。
2. 算法工程师：前置知识 → 数据 → SFT/LoRA → DPO/RLHF → 评估 → 多模态。
3. 面试冲刺：术语速查 → RAG/Agent/训练问答 → 系统设计 → 代码题。
4. 如何使用手册：状态标签、前置知识和阅读顺序。

Keep `roadmap.md` as detailed timelines and project ideas, remove raw styled cards, and link to the three anchors.

- [ ] **Step 3: Complete knowledge, practice and interview landings**

- `llms/index.md`: six modules, dependencies, start points and status legend.
- `practice/index.md`: beginner, intermediate and production project cards pointing to existing articles.
- `interviews/index.md`: preparation order, five existing categories and the interview path.

- [ ] **Step 4: Complete resource, reference and about pages**

- Resource pages list primary papers, official blogs/videos and official repositories with title, year, URL and relevance.
- Reference index explains the four collections; each detail page contains at least five useful entries and deep links.
- About states audience, author perspective, status meanings, source policy, update process and GitHub route.
- Changelog records the redesign without claiming deployment before it occurs.

- [ ] **Step 5: Update navigation and verify no placeholder remains**

Top nav remains `/guide/`, `/llms/`, `/practice/`, `/about/`. Secondary sidebars begin with `/resources/`, `/reference/`, and `/interviews/` indexes.

```bash
rg -n '^> 待补充' docs -g '*.md' -g '!superpowers/**' -g '!_drafts/**'
pnpm content:check
pnpm docs:build
```

Expected: `rg` has no matches; checks PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/guide docs/llms/index.md docs/practice docs/interviews/index.md docs/resources docs/reference docs/about docs/.vitepress/config
git commit -m 'docs: rewrite learning paths and site landings'
```

---

### Task 9: Refresh the RAG and Agent module overviews

**Files:**
- Rewrite: `docs/llms/rag/index.md`
- Rewrite: `docs/llms/agent/index.md`

**Interfaces:**
- Consumes: `LearningObjectives`, `SourceList`, metadata contract and module navigation.
- Produces: two verified overview pages; child articles retain their existing review status.

- [ ] **Step 1: Audit time-sensitive claims against primary sources**

For RAG, verify the original RAG paper, dense retrieval, late interaction, current GraphRAG documentation and production evaluation guidance. For Agent, verify ReAct, tool use, current agent-building guidance, tool-calling interfaces and evaluation guidance. Record exact primary-source URLs; search-result summaries are not evidence.

- [ ] **Step 2: Rewrite each overview as one learning unit**

Each page contains, in order:

1. `<LearningObjectives :items='[...]' />` with three measurable outcomes.
2. `## 这个模块解决什么问题`.
3. `## 核心工作流` with one valid Mermaid diagram.
4. `## 关键概念` with a compact comparison table.
5. `## 推荐学习顺序` linking every current child article.
6. `## 实践检查点`.
7. `## 版本与边界`.
8. `## 参考资料` or `<SourceList>` with primary sources.

Target 2,500–4,500 Chinese characters per page. Remove duplicated deep exposition rather than creating new files; Git history preserves the prior prose.

- [ ] **Step 3: Update metadata only after verification**

Set `updated` and `reviewed` to the actual work date, `contentStatus: verified`, and explicit `techVersion`. Do not upgrade child-page statuses.

- [ ] **Step 4: Validate and commit**

```bash
pnpm content:check
pnpm docs:build
git add docs/llms/rag/index.md docs/llms/agent/index.md
git commit -m 'docs: refresh RAG and Agent overviews'
```

Open both pages and verify Mermaid, child links, source links and next-step navigation.

---

### Task 10: Refresh the MCP and Prompt module overviews

**Files:**
- Rewrite: `docs/llms/mcp/index.md`
- Rewrite: `docs/llms/prompt/index.md`

**Interfaces:**
- Same authoring contracts as Task 9; independently reviewable.

- [ ] **Step 1: Audit primary sources**

For MCP, use the current Model Context Protocol specification, official SDK repositories and official security guidance. For Prompt, use current model-provider prompting documentation and original papers. Explicitly distinguish prompt engineering from broader context engineering.

- [ ] **Step 2: Rewrite with the eight-section template**

MCP includes roles, capability negotiation, tools/resources/prompts, transports, authorization boundary and learning order. Prompt includes instruction hierarchy, examples, structured outputs, context construction, evaluation and security boundaries.

- [ ] **Step 3: Set verified metadata, validate and commit**

```bash
pnpm content:check
pnpm docs:build
git add docs/llms/mcp/index.md docs/llms/prompt/index.md
git commit -m 'docs: refresh MCP and Prompt overviews'
```

No fixed version may be described as current without an explicit date or version scope.

---

### Task 11: Refresh the Training and Multimodal module overviews

**Files:**
- Rewrite: `docs/llms/training/index.md`
- Rewrite: `docs/llms/multimodal/index.md`

**Interfaces:**
- Same authoring contracts as Task 9; completes all six first-phase overviews.

- [ ] **Step 1: Audit primary sources**

For Training, verify SFT, PEFT/LoRA, DPO, RLHF, evaluation and serving through original papers and official project documentation. For Multimodal, verify visual encoders, connectors, unified models, generation and evaluation through original papers and official model/project documentation. Avoid leaderboard claims unless dated and directly sourced.

- [ ] **Step 2: Rewrite with the eight-section template**

Training distinguishes pretraining, post-training, preference optimization and serving. Multimodal distinguishes representation alignment, instruction tuning, understanding, generation and deployment. Preserve every child-article link.

- [ ] **Step 3: Set verified metadata, validate and commit**

```bash
pnpm content:check
pnpm docs:build
git add docs/llms/training/index.md docs/llms/multimodal/index.md
git commit -m 'docs: refresh Training and Multimodal overviews'
```

Both Mermaid diagrams must render and every factual update must have a primary source.

---

### Task 12: Enforce built-site quality and complete production acceptance

**Files:**
- Modify: `package.json`
- Create: `scripts/check-built-site.ts`
- Create: `tests/content/check-built-site.test.ts`
- Modify: `.github/workflows/deploy.yml`
- Create: `docs/public/og-default.png`
- Modify: `docs/.vitepress/config/head.ts`
- Modify: `docs/about/changelog.md`

**Interfaces:**
- Consumes: built VitePress output and page-head generator.
- Produces: `checkBuiltSite(dist)`, `pnpm quality`, strict deployment and final acceptance evidence.

- [ ] **Step 1: Write failing built-site tests**

Create fixtures proving `checkBuiltSite(dist)` rejects a missing canonical, canonical without `/notes-on-llms/`, `meta name='og:title'`, a missing favicon target, and output paths containing `_drafts`, `superpowers` or `开发计划`.

```ts
await expect(checkBuiltSite(fixtureDir)).rejects.toThrow('canonical')
```

Run: `pnpm test -- tests/content/check-built-site.test.ts`

Expected: FAIL because `check-built-site.ts` does not exist.

- [ ] **Step 2: Implement the built-site checker**

Export `checkBuiltSite(dist: string): Promise<void>`. Recursively inspect `.html`, referenced favicon/share assets and `sitemap.xml`; aggregate every error and print inspected page count. The CLI checks `docs/.vitepress/dist`.

- [ ] **Step 3: Add the default social share image**

Create `docs/public/og-default.png` at 1200×630 using paper, ink and cobalt. Include only the site name and short descriptor, keep it below 400 KB, and change `head.ts` to `${SITE_ORIGIN}${SITE_BASE}og-default.png`.

- [ ] **Step 4: Define and run the complete quality command**

```json
{
  "scripts": {
    "site:check": "tsx scripts/check-built-site.ts",
    "quality": "pnpm test && pnpm content:check && pnpm docs:build && pnpm site:check"
  }
}
```

Run: `pnpm quality`

Expected: all tests, content checks, build and built-site checks PASS.

- [ ] **Step 5: Make GitHub Actions run the same gate**

Change install to `pnpm install --frozen-lockfile`; replace build with `pnpm quality`; keep upload path `docs/.vitepress/dist` and current Pages permissions.

- [ ] **Step 6: Run responsive browser acceptance**

Start `pnpm docs:preview --host 127.0.0.1`. Using the Playwright skill, inspect 1440×1000, 768×900 and 360×800 for homepage, `/guide/`, one page from each module, `/practice/`, `/resources/`, `/reference/`, `/about/`, local search, light/dark mode, code, tables and Mermaid. Check console errors and horizontal overflow. Save only final screenshots under `output/playwright/` and remove transient session artifacts.

- [ ] **Step 7: Run Lighthouse twice**

Test homepage and one representative article with mobile emulation. Required lower score across two runs: Performance ≥ 90, Accessibility ≥ 90, SEO ≥ 90. Fix measured causes and repeat; do not lower thresholds.

- [ ] **Step 8: Run final hygiene checks**

```bash
pnpm quality
rg -n 'href=["'"']/[^/]' docs -g '*.md' -g '!superpowers/**'
git diff --check
git status --short
```

Expected: quality passes, `rg` has no matches, diff check is clean, status contains only Task 12 files.

- [ ] **Step 9: Record release notes and commit**

Update `docs/about/changelog.md` with actual completion date and verified changes.

```bash
git add package.json .github/workflows/deploy.yml scripts tests docs/public/og-default.png docs/.vitepress/config/head.ts docs/about/changelog.md output/playwright
git commit -m 'ci: enforce production site quality gates'
```

---

## Final Acceptance Checklist

- [ ] `pnpm quality` passes from a clean checkout with frozen lockfile.
- [ ] GitHub Pages base works for home, landing and deep direct URLs.
- [ ] Every visible navigation destination contains useful content.
- [ ] Six module overviews are sourced and verified; deep articles keep honest review states.
- [ ] Homepage updates are data-driven.
- [ ] Article pages expose difficulty, reading time, prerequisites, version, status, progress and next step.
- [ ] Canonical, Open Graph, Twitter Card, Article JSON-LD, favicon and sitemap are present.
- [ ] Drafts and internal planning documents are absent from build, search and sitemap.
- [ ] 360, 768 and 1440 pixel layouts have no console errors or page overflow.
- [ ] Lighthouse Performance, Accessibility and SEO are each at least 90.

## Deferred Follow-up Plans

After this first-phase plan ships, create one independent deep-content plan in this order: RAG, Agent, MCP, Prompt, Training, Multimodal. Each batch reuses this content contract and theme, and completes its own source audit, diff review, build and browser acceptance.
