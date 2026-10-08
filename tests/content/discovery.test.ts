import { describe, expect, it } from 'vitest'
import { discoverArticles, resolvePrerequisites } from '../../docs/.vitepress/content/discovery'
import type { ContentIndexItem } from '../../docs/.vitepress/content/model'

const page = (overrides: Partial<ContentIndexItem>): ContentIndexItem => ({
  title:'检索基础', description:'混合召回与评估', pageType:'article', module:'rag',
  level:'beginner', prerequisites:[], updated:'2026-10-08', reviewed:'2026-10-08',
  contentStatus:'needs-review', tags:['retrieval'], url:'/llms/rag/retrieval',
  techVersion:'教学示例', readingTime:8, sourceCount:2, ...overrides,
})
const pages = [page({}), page({title:'Agent 记忆',module:'agent',url:'/llms/agent/memory',level:'advanced'}),
  page({title:'检索实践',url:'/llms/rag/csdn/example',tags:['csdn-mirror']}),
  page({title:'草稿',url:'/draft',contentStatus:'draft'}),page({title:'入口',pageType:'landing',url:'/llms/'})]

describe('article discovery', () => {
  it('combines text, module, level and mainline filters', () => {
    expect(discoverArticles(pages,{query:'混合 检索',module:'rag',level:'beginner',kind:'mainline'}).map(p=>p.url)).toEqual(['/llms/rag/retrieval'])
    expect(discoverArticles(pages,{query:'AGENT',module:'all',level:'all',kind:'all'}).map(p=>p.url)).toEqual(['/llms/agent/memory'])
  })
  it('excludes drafts and supports mirror and saved-article filtering', () => {
    expect(discoverArticles(pages,{query:'',module:'all',level:'all',kind:'mirror'}).map(p=>p.url)).toEqual(['/llms/rag/csdn/example'])
    expect(discoverArticles(pages,{query:'',module:'all',level:'all',kind:'all',savedUrls:['/llms/agent/memory.html']}).map(p=>p.url)).toEqual(['/llms/agent/memory'])
  })
  it('resolves prerequisite titles including anchors and preserves the destination', () => {
    expect(resolvePrerequisites(['/llms/rag/retrieval.html#hybrid','/unknown'],pages)).toEqual([
      {href:'/llms/rag/retrieval.html#hybrid',title:'检索基础'}, {href:'/unknown',title:'延伸阅读'},
    ])
  })
})
