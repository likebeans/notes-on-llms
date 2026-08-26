import { readdir, readFile } from 'node:fs/promises'
import { relative, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import matter from 'gray-matter'
import { assertValidFrontmatter } from '../docs/.vitepress/content/model'

export const EXCLUDED_PATHS = ['.vitepress/', 'public/', 'superpowers/', '_drafts/', '开发计划.md']

export function findBaseUnsafeLinks(source: string): string[] {
  return [...source.matchAll(/href=["'](\/(?!\/)[^"']*)["']/g)].map(match => match[1])
}

export function findPlaceholderMarkers(source: string): string[] {
  return [...new Set([...source.matchAll(/加密内容|待补充/g)].map(match => match[0]))]
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

    try {
      assertValidFrontmatter(data, path)
    } catch (error) {
      errors.push((error as Error).message)
    }

    for (const href of findBaseUnsafeLinks(source)) {
      errors.push(`${path}: raw HTML href bypasses base: ${href}`)
    }

    for (const marker of findPlaceholderMarkers(source)) {
      errors.push(`${path}: public placeholder marker: ${marker}`)
    }

    if (data.contentStatus === 'draft' && !path.startsWith('_drafts/') && !path.endsWith('.draft.md')) {
      errors.push(`${path}: draft pages must live under _drafts or use .draft.md`)
    }
  }

  if (errors.length) throw new Error(errors.join('\n'))
}

async function main(): Promise<void> {
  const root = resolve('docs')
  const files = await collectPublicMarkdown(root)
  await checkContent(root)
  console.log(`Checked ${files.length} public Markdown files.`)
}

const invokedPath = process.argv[1] && pathToFileURL(resolve(process.argv[1])).href
if (invokedPath === import.meta.url) {
  main().catch(error => {
    console.error((error as Error).message)
    process.exitCode = 1
  })
}
