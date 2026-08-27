import { describe, expect, it } from 'vitest'
import { sidebar } from '../../docs/.vitepress/config/sidebar'

describe('sidebar', () => {
  it('exposes the CSDN mirror from the resources section', () => {
    const resourceItems = sidebar['/resources/'][0].items

    expect(resourceItems).toContainEqual({ text: 'CSDN 镜像', link: '/resources/csdn' })
  })
})
