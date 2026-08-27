import { readFile } from 'node:fs/promises'
import { join, relative } from 'node:path'
import { describe, expect, it } from 'vitest'
import { collectPublicMarkdown } from '../../scripts/check-content'

describe('CSDN mirror content', () => {
  it('uses Markdown fences instead of raw HTML code blocks', async () => {
    const root = join(process.cwd(), 'docs')
    const files = (await collectPublicMarkdown(root))
      .filter(file => /\/csdn\/[^/]+\.md$/.test(file))

    const rawCodeBlocks: string[] = []
    for (const file of files) {
      const source = await readFile(file, 'utf8')
      if (/<pre><code\b/i.test(source)) rawCodeBlocks.push(relative(root, file))
    }

    expect(rawCodeBlocks).toEqual([])
  })
})
