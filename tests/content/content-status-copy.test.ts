import { describe, expect, it } from 'vitest'
import { CONTENT_STATUSES, type ContentStatus } from '../../docs/.vitepress/content/model'
import { getContentStatusCopy } from '../../docs/.vitepress/theme/components/article/contentStatusCopy'

describe('content status copy', () => {
  it.each(CONTENT_STATUSES)('provides reader guidance for %s pages', (status: ContentStatus) => {
    const copy = getContentStatusCopy(status)

    expect(copy.label.length).toBeGreaterThan(0)
    expect(copy.guidance.length).toBeGreaterThan(8)
  })

  it('warns readers about needs-review, opinion and historical content boundaries', () => {
    expect(getContentStatusCopy('needs-review').guidance).toContain('时效')
    expect(getContentStatusCopy('opinion').guidance).toContain('观点')
    expect(getContentStatusCopy('historical').guidance).toContain('演进')
  })
})
