import { describe, expect, it } from 'vitest'
import { extractSourceLinks } from '../../docs/.vitepress/content/sourceLinks'

describe('article source links', () => {
  it('counts inline, reference, autolink and HTML sources once per document', async () => {
    const source = '[论文](https://example.com/paper#a) [重复](https://example.com/paper#b)\n[规范][spec]\n\n[spec]: https://example.com/spec\n\n<https://example.com/auto> <a href="https://example.com/html">说明</a>'
    expect(await extractSourceLinks(source)).toEqual([
      'https://example.com/paper', 'https://example.com/spec',
      'https://example.com/auto', 'https://example.com/html',
    ])
  })
  it('excludes images, local navigation, code examples and unused reference definitions', async () => {
    const source = '![图](https://example.com/img.png) [本地](/guide/)\n`[伪引用](https://example.com/inline)`\n\n```md\n[伪引用](https://example.com/code)\n```\n\n[unused]: https://example.com/unused'
    expect(await extractSourceLinks(source)).toEqual([])
  })
  it('supports SourceList without counting its duplicates or links in fenced examples', async () => {
    const source = '<SourceList :items="[{ title: \'文档\', href: \'https://example.com/docs\' }]" />\n\n[文档](https://example.com/docs)\n\n```vue\n<SourceList :items="[{ href: \'https://example.com/fake\' }]" />\n```'
    expect(await extractSourceLinks(source)).toEqual(['https://example.com/docs'])
  })
})
