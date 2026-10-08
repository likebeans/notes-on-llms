import { describe, expect, it } from 'vitest'
import { libraryStateFromSearch, libraryUrlWithState } from '../../docs/.vitepress/theme/components/learning/libraryState'

describe('article library URL state', () => {
  it('restores a Chinese query, combined filters and an expanded result list', () => {
    expect(libraryStateFromSearch('?q=工具+调用&module=agent&level=advanced&kind=mainline&saved=bookmarked&limit=40')).toEqual({
      query: '工具 调用', module: 'agent', level: 'advanced', kind: 'mainline', saved: 'bookmarked', limit: 40,
    })
  })

  it('falls back to usable defaults for unsupported filters and invalid list limits', () => {
    expect(libraryStateFromSearch('?module=unknown&level=expert&kind=other&saved=private&limit=100000')).toEqual({
      query: '', module: 'all', level: 'all', kind: 'all', saved: 'all', limit: 20,
    })
    for (const limit of ['0', '-20', '20.5', '40junk', 'Infinity']) {
      expect(libraryStateFromSearch(`?limit=${limit}`).limit).toBe(20)
    }
  })

  it('preserves the base path, unrelated parameters and fragment when filters change', () => {
    const url = new URL(libraryUrlWithState('http://localhost:4173/notes-on-llms/guide/library.html?revision=1#article-library-title', {
      query: 'RAG + 引用', module: 'rag', level: 'all', kind: 'mainline', saved: 'all', limit: 60,
    }))
    expect(url.pathname).toBe('/notes-on-llms/guide/library.html')
    expect(url.hash).toBe('#article-library-title')
    expect(url.searchParams.get('revision')).toBe('1')
    expect(url.searchParams.get('q')).toBe('RAG + 引用')
    expect(url.searchParams.get('limit')).toBe('60')
    expect(url.searchParams.has('level')).toBe(false)
    expect(libraryStateFromSearch(url.search)).toMatchObject({ query: 'RAG + 引用', module: 'rag', limit: 60 })
  })

  it('clears stale filter parameters when reset while retaining unrelated URL state', () => {
    const result = libraryUrlWithState('https://example.com/notes-on-llms/guide/library?q=old&module=rag&level=advanced&kind=mirror&saved=completed&limit=40&revision=1#results', libraryStateFromSearch(''))
    expect(result).toBe('https://example.com/notes-on-llms/guide/library?revision=1#results')
  })

  it('bounds imported queries without losing ordinary punctuation or Chinese text', () => {
    expect(libraryStateFromSearch(`?q=${'词'.repeat(300)}`).query).toHaveLength(200)
    expect(libraryStateFromSearch('?q=%E4%B8%AD%E6%96%87%26%3D%2B').query).toBe('中文&=+')
  })
})
