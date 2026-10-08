import { afterEach, describe, expect, it, vi } from 'vitest'
import { createRenderer, createSSRApp, h, nextTick } from 'vue'
import { renderToString } from 'vue/server-renderer'
import { useLibraryFilters } from '../../docs/.vitepress/theme/components/learning/useLibraryFilters'

// A no-DOM host runs the composable's real Vue lifecycle and watcher scheduling.
const renderer = createRenderer<object, object>({
  createElement: () => ({}), createText: () => ({}), createComment: () => ({}),
  insert() {}, remove() {}, setText() {}, setElementText() {}, patchProp() {},
  parentNode: () => null, nextSibling: () => null,
})
const unmounts: (() => void)[] = []
afterEach(() => { unmounts.splice(0).forEach(unmount => unmount()); vi.unstubAllGlobals() })

function mountLibrary(href: string) {
  const savedHistory = { scrollPosition: { top: 420, left: 0 }, routerMarker: 'keep-me' }
  const browser = Object.assign(new EventTarget(), {
    location: new URL(href),
    history: { state: savedHistory, replaceState: vi.fn() },
  })
  browser.history.replaceState.mockImplementation((state, _title, url) => {
    browser.history.state = state
    browser.location = new URL(url, browser.location)
  })
  vi.stubGlobal('window', browser)
  let filters!: ReturnType<typeof useLibraryFilters>
  const app = renderer.createApp({ setup() { filters = useLibraryFilters(); return () => null } })
  app.mount({})
  unmounts.push(() => app.unmount())
  return { browser, filters, savedHistory }
}

describe('library filter browser lifecycle', () => {
  it('renders default state during SSR without accessing window', async () => {
    vi.stubGlobal('window', undefined)
    const app = createSSRApp({ setup() {
      const state = useLibraryFilters()
      return () => h('p', `${state.module.value}:${state.limit.value}`)
    } })
    expect(await renderToString(app)).toBe('<p>all:20</p>')
  })

  it('restores refresh state without resetting limit and preserves history state when filtering', async () => {
    const { browser, filters, savedHistory } = mountLibrary('https://example.com/notes/guide/library?q=工具&module=agent&limit=60#results')
    await nextTick()
    expect(filters.query.value).toBe('工具')
    expect(filters.module.value).toBe('agent')
    expect(filters.limit.value).toBe(60)
    filters.query.value = '新查询'
    await nextTick()
    expect(filters.limit.value).toBe(20)
    expect(browser.location.searchParams.get('q')).toBe('新查询')
    expect(browser.location.searchParams.has('limit')).toBe(false)
    expect(browser.location.hash).toBe('#results')
    expect(browser.history.state).toEqual(savedHistory)
  })

  it('restores back/forward filters and expanded limit in the same watcher flush', async () => {
    const { browser, filters } = mountLibrary('https://example.com/guide/library?q=old&module=rag&limit=40')
    await nextTick()
    browser.location = new URL('https://example.com/guide/library?q=new&module=agent&kind=mainline&limit=80')
    browser.dispatchEvent(new Event('popstate'))
    await nextTick()
    expect(filters.query.value).toBe('new')
    expect(filters.module.value).toBe('agent')
    expect(filters.kind.value).toBe('mainline')
    expect(filters.limit.value).toBe(80)
    expect(browser.location.searchParams.get('limit')).toBe('80')
  })

  it('does not write queued filters into an article URL or restore after unmount', async () => {
    const { browser, filters } = mountLibrary('https://example.com/guide/library')
    await nextTick()
    browser.history.replaceState.mockClear()
    filters.query.value = 'pending'
    browser.location = new URL('https://example.com/llms/agent/memory?existing=1#summary')
    await nextTick()
    expect(browser.history.replaceState).not.toHaveBeenCalled()
    expect(browser.location.href).toBe('https://example.com/llms/agent/memory?existing=1#summary')
    unmounts.pop()!()
    browser.location = new URL('https://example.com/guide/library?q=after-unmount&limit=80')
    browser.dispatchEvent(new Event('popstate'))
    await nextTick()
    expect(filters.query.value).toBe('pending')
  })
})
