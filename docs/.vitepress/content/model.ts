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
  reviewScope?: string; exampleStatus?: 'not-run' | 'partial' | 'executed'
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
  if (value.reviewScope !== undefined && typeof value.reviewScope !== 'string') fail('reviewScope must be text')
  if (value.exampleStatus !== undefined && !['not-run','partial','executed'].includes(String(value.exampleStatus))) fail('exampleStatus is invalid')
  if (value.pageType === 'article' && value.contentStatus !== 'draft') {
    if (!LEVELS.includes(value.level as typeof LEVELS[number])) fail('level is required for an article')
    if (!Array.isArray(value.prerequisites)) fail('prerequisites is required for an article')
    if (!value.reviewed || !ISO_DATE.test(String(value.reviewed))) fail('reviewed is required for a published article')
    if (!value.techVersion || typeof value.techVersion !== 'string') fail('techVersion is required for an article')
  }
}
