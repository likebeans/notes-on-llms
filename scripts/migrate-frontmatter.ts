import { execFileSync } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'
import { relative, resolve } from 'node:path'
import matter from 'gray-matter'
import { collectPublicMarkdown } from './check-content'

const MODULE_DEFAULTS = {
  prompt: { level: 'beginner', prerequisites: [], techVersion: '待复核（2026-08）' },
  rag: { level: 'intermediate', prerequisites: ['/llms/prompt/'], techVersion: '待复核（2026-08）' },
  agent: { level: 'advanced', prerequisites: ['/llms/prompt/', '/llms/rag/'], techVersion: '待复核（2026-08）' },
  mcp: { level: 'intermediate', prerequisites: ['/llms/agent/tool-calling'], techVersion: '待复核（2026-08）' },
  training: { level: 'advanced', prerequisites: ['/guide/prerequisites'], techVersion: '待复核（2026-08）' },
  multimodal: { level: 'advanced', prerequisites: ['/guide/prerequisites'], techVersion: '待复核（2026-08）' },
} as const

type Module = keyof typeof MODULE_DEFAULTS

const KNOWN_METADATA = new Set([
  'title', 'description', 'pageType', 'module', 'level', 'prerequisites', 'updated',
  'reviewed', 'contentStatus', 'techVersion', 'tags', 'author', 'order',
])

const FALLBACK_METADATA: Record<string, { title: string, description: string, tags: string[] }> = {
  'index.md': {
    title: 'Notes on LLMs',
    description: '大模型技术探索与实践',
    tags: ['llm', 'knowledge-map', 'learning-path'],
  },
  'llms/index.md': {
    title: 'LLMs 技术专区',
    description: '大模型核心技术学习专区，涵盖 RAG、Agent、训练微调和多模态等方向。',
    tags: ['llm', 'modules'],
  },
  'llms/rag/csdn_articles.md': {
    title: 'CSDN 专栏文章列表',
    description: 'CSDN 专栏文章索引',
    tags: ['rag'],
  },
}

function moduleFor(path: string): Module | undefined {
  const match = path.match(/^llms\/(prompt|rag|agent|mcp|training|multimodal)\//)
  return match?.[1] as Module | undefined
}

function pageTypeFor(path: string): 'article' | 'path' | 'landing' {
  if (path === 'index.md' || path === 'llms/index.md' || path.startsWith('about/') || path.startsWith('resources/')) {
    return 'landing'
  }
  if (path === 'guide/roadmap.md') return 'path'
  return 'article'
}

function siteTags(path: string): string[] {
  return [path.split('/')[0].replace(/\.md$/, '')]
}

function updatedFor(file: string): string {
  const date = execFileSync('git', ['log', '-1', '--format=%cs', '--', file], {
    cwd: process.cwd(),
    encoding: 'utf8',
  }).trim()
  if (!date) throw new Error(`No git modification date found for ${file}`)
  return date
}

export function stringifyFrontmatter(source: string, data: Record<string, unknown>): string {
  const output = matter.stringify(matter(source).content, data)
  return source.endsWith('\n') ? output : output.replace(/\n$/, '')
}

export async function migrateFrontmatter(
  root = resolve('docs'),
  migratedAt = updatedFor,
): Promise<number> {
  let migrated = 0

  for (const file of await collectPublicMarkdown(root)) {
    const path = relative(root, file).replaceAll('\\', '/')
    const source = await readFile(file, 'utf8')
    const parsed = matter(source)
    const module = moduleFor(path)
    const pageType = pageTypeFor(path)
    const fallback = FALLBACK_METADATA[path]
    const title = parsed.data.title ?? fallback?.title
    const description = parsed.data.description ?? fallback?.description

    if (!title || !description) throw new Error(`Missing title or description for ${path}`)

    const metadata: Record<string, unknown> = {
      title,
      description,
      pageType,
      module: module ?? 'site',
      updated: parsed.data.updated ?? migratedAt(file),
      contentStatus: parsed.data.contentStatus ?? 'needs-review',
      tags: parsed.data.tags ?? fallback?.tags ?? (module ? [module] : siteTags(path)),
    }

    if (pageType === 'article') {
      const defaults = module ? MODULE_DEFAULTS[module] : {
        level: 'intermediate', prerequisites: [], techVersion: '待复核（2026-08）',
      }
      metadata.level = parsed.data.level ?? defaults.level
      metadata.prerequisites = parsed.data.prerequisites ?? defaults.prerequisites
      metadata.reviewed = parsed.data.reviewed ?? '2026-08-25'
      metadata.techVersion = parsed.data.techVersion ?? defaults.techVersion
    }

    const extras = Object.fromEntries(Object.entries(parsed.data).filter(([key]) => !KNOWN_METADATA.has(key)))
    const nextSource = stringifyFrontmatter(source, { ...metadata, ...extras })
    if (nextSource !== source) {
      await writeFile(file, nextSource, 'utf8')
      migrated++
    }
  }

  return migrated
}

async function main(): Promise<void> {
  const migrated = await migrateFrontmatter()
  console.log(`Migrated ${migrated} public Markdown files.`)
}

if (process.argv[1] && new URL(import.meta.url).pathname === resolve(process.argv[1])) {
  main().catch(error => {
    console.error((error as Error).message)
    process.exitCode = 1
  })
}
