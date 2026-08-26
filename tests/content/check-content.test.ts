import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, it } from 'vitest'
import { checkContent, findBaseUnsafeLinks } from '../../scripts/check-content'

const temporaryRoots: string[] = []

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map(root => rm(root, { recursive: true, force: true })))
})

it('finds raw HTML links that bypass base', () => {
  expect(findBaseUnsafeLinks('<a href="/llms/rag/">RAG</a>')).toEqual(['/llms/rag/'])
  expect(findBaseUnsafeLinks('[RAG](/llms/rag/)')).toEqual([])
})

it('rejects draft content outside the drafts directory', async () => {
  const root = await mkdtemp(join(tmpdir(), 'content-check-'))
  temporaryRoots.push(root)
  await writeFile(join(root, 'article.md'), `---
title: Draft article
description: A draft outside the draft directory
pageType: article
module: rag
level: intermediate
prerequisites: []
updated: 2026-08-25
contentStatus: draft
tags: [rag]
---
`, 'utf8')

  await expect(checkContent(root)).rejects.toThrow(
    'article.md: draft pages must live under _drafts or use .draft.md',
  )
})

it('rejects public placeholder markers', async () => {
  const root = await mkdtemp(join(tmpdir(), 'content-check-'))
  temporaryRoots.push(root)
  await writeFile(join(root, 'placeholder.md'), `---
title: Placeholder
description: A public placeholder page
pageType: article
module: site
level: intermediate
prerequisites: []
updated: '2026-08-25'
reviewed: '2026-08-25'
contentStatus: needs-review
tags: [interviews]
---

# Placeholder

这部分内容待补充。
`, 'utf8')

  await expect(checkContent(root)).rejects.toThrow('placeholder marker')
})
