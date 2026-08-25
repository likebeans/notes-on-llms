import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import matter from 'gray-matter'
import { afterEach, expect, it } from 'vitest'
import { migrateFrontmatter, stringifyFrontmatter } from '../../scripts/migrate-frontmatter'

const temporaryRoots: string[] = []

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map(root => rm(root, { recursive: true, force: true })))
})

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

it('keeps an existing updated date on a post-migration rerun', async () => {
  const root = await mkdtemp(join(tmpdir(), 'frontmatter-rerun-'))
  temporaryRoots.push(root)
  const file = join(root, 'article.md')
  await writeFile(file, `---
title: Existing article
description: Already migrated content
pageType: article
module: site
updated: '2024-01-02'
contentStatus: needs-review
tags:
  - guide
level: intermediate
prerequisites: []
reviewed: '2026-08-25'
techVersion: 待复核（2026-08）
---

# Existing article
`, 'utf8')

  await expect(migrateFrontmatter(root, () => '2026-08-25')).resolves.toBe(0)
  expect(matter(await readFile(file, 'utf8')).data.updated).toBe('2024-01-02')
})

it('uses the Git-derived date for initially unmigrated content', async () => {
  const root = await mkdtemp(join(tmpdir(), 'frontmatter-initial-'))
  temporaryRoots.push(root)
  const file = join(root, 'article.md')
  await writeFile(file, '---\ntitle: New article\ndescription: Legacy content without migration metadata\n---\n\n# New article\n', 'utf8')

  await migrateFrontmatter(root, () => '2025-03-04')
  expect(matter(await readFile(file, 'utf8')).data.updated).toBe('2025-03-04')
})
