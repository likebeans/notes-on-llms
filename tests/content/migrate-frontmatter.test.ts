import matter from 'gray-matter'
import { expect, it } from 'vitest'
import { stringifyFrontmatter } from '../../scripts/migrate-frontmatter'

it('preserves a Markdown body without a final newline', () => {
  const source = '---\ntitle: 视频\ndescription: 精选视频资源索引\n---\n\n# 视频\n\n> 待补充'
  const output = stringifyFrontmatter(source, {
    title: '视频',
    description: '精选视频资源索引',
    pageType: 'landing',
    module: 'site',
    updated: '2026-01-10',
    contentStatus: 'needs-review',
    tags: ['resources'],
  })

  expect(matter(output).content).toBe(matter(source).content)
})
