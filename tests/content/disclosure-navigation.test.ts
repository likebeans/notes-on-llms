import { afterEach, expect, it, vi } from 'vitest'
import { installDisclosureNavigation } from '../../docs/.vitepress/theme/components/article/disclosureNavigation'

class ElementStub {
  open = false
  parentElement: ElementStub | null = null
  attributes: Record<string, string> = {}
  constructor(public tagName: string) {}
  getAttribute(name: string) { return this.attributes[name] ?? null }
  hasAttribute(name: string) { return name in this.attributes }
  closest(selector: string): ElementStub | null {
    if (selector === 'a' && this.tagName === 'A') return this
    return this.parentElement?.closest(selector) ?? null
  }
  getBoundingClientRect() { return { top: this.parentElement?.open ? 480 : 0 } }
}

function environment(paths: { address?: string; rendered?: string } = {}) {
  const frames = new Map<number, FrameRequestCallback>()
  let frameId = 0
  const browser = Object.assign(new EventTarget(), {
    location: new URL(`https://example.test${paths.address ?? '/notes/embedding.html'}#hidden`),
    scrollY: 200,
    innerHeight: 800,
    requestAnimationFrame(callback: FrameRequestCallback) { frames.set(++frameId, callback); return frameId },
    cancelAnimationFrame(id: number) { frames.delete(id) },
    getComputedStyle: () => ({ paddingTop: '24px' }),
    scrollTo: vi.fn(),
  })
  const outer = new ElementStub('DETAILS')
  const inner = new ElementStub('DETAILS'); inner.parentElement = outer
  const target = new ElementStub('H3'); target.parentElement = inner
  const targets = new Map([['hidden', target], ['中文标题', target]])
  const doc = { getElementById: (id: string) => targets.get(id) ?? null }
  let path = paths.rendered ?? '/notes/embedding.html'
  const navigation = installDisclosureNavigation({
    window: browser as unknown as Window,
    document: doc as unknown as Document,
    getRenderedPath: () => path,
    getScrollOffset: () => 64,
  })
  const initialFrames = [...frames.values()]; frames.clear(); initialFrames.forEach(callback => callback(0))
  const initialPosition = browser.scrollTo.mock.calls[0]
  browser.scrollTo.mockClear()
  return { browser, outer, inner, target, navigation, initialPosition,
    setPath(value: string) { path = value },
    flush() { const callbacks = [...frames.values()]; frames.clear(); callbacks.forEach(callback => callback(0)) },
    pending: () => frames.size,
    click(overrides: Record<string, unknown> = {}) {
      const { href, ...eventOverrides } = overrides
      const anchor = new ElementStub('A'); anchor.attributes.href = typeof href === 'string' ? href : browser.location.href
      const event = new Event('click', { cancelable: true })
      Object.assign(event, { button: 0, ctrlKey: false, shiftKey: false, altKey: false, metaKey: false }, eventOverrides)
      Object.defineProperty(event, 'target', { value: anchor })
      // VitePress's earlier capture listener already prevented native navigation.
      event.preventDefault()
      browser.dispatchEvent(event)
    },
  }
}

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

it('reveals every collapsed ancestor for direct, decoded and missing hash targets', () => {
  const e = environment()
  expect(e.initialPosition).toEqual([0, 640])
  expect(e.outer.open).toBe(true)
  expect(e.inner.open).toBe(true)
  e.outer.open = e.inner.open = false
  e.browser.location.hash = '#%E4%B8%AD%E6%96%87%E6%A0%87%E9%A2%98'
  e.navigation.revealCurrent()
  expect(e.outer.open && e.inner.open).toBe(true)
  e.browser.location.hash = '#%invalid'
  expect(() => e.navigation.revealCurrent()).not.toThrow()
  e.navigation.dispose()
})

it('reopens an already-current hash and corrects scroll with the VitePress offset', () => {
  const e = environment()
  e.outer.open = e.inner.open = false
  e.click()
  expect(e.outer.open && e.inner.open).toBe(true)
  e.flush()
  expect(e.browser.scrollTo).toHaveBeenCalledWith(0, 640)
  e.navigation.dispose()
})

it('leaves modified clicks and ordinary visible targets to normal navigation', () => {
  const e = environment()
  e.outer.open = e.inner.open = false
  e.click({ ctrlKey: true })
  e.click({ href: 'https://external.test/notes/embedding.html#hidden' })
  expect(e.outer.open || e.inner.open).toBe(false)
  expect(e.pending()).toBe(0)
  e.outer.open = e.inner.open = true
  e.click()
  expect(e.pending()).toBe(0)
  e.navigation.dispose()
})

it('waits for the destination content before expanding a cross-page deep link', () => {
  const e = environment()
  e.outer.open = e.inner.open = false
  e.browser.location.pathname = '/notes/paradigms.html'
  e.click()
  e.navigation.revealCurrent()
  expect(e.inner.open).toBe(false)
  e.setPath('/notes/paradigms.html')
  e.navigation.revealCurrent()
  expect(e.outer.open && e.inner.open).toBe(true)
  e.navigation.dispose()
})

it('reveals hash/history targets without replacing handlers or overriding restored scroll', () => {
  const e = environment()
  const existing = vi.fn()
  e.browser.addEventListener('popstate', existing)
  for (const type of ['hashchange', 'popstate']) {
    e.outer.open = e.inner.open = false
    e.browser.dispatchEvent(new Event(type))
    expect(e.outer.open && e.inner.open).toBe(true)
  }
  expect(existing).toHaveBeenCalledOnce()
  expect(e.browser.scrollTo).not.toHaveBeenCalled()
  e.navigation.dispose()
})

it('cleans up listeners and pending correction when the article layout unmounts', () => {
  const e = environment()
  e.outer.open = e.inner.open = false
  e.click()
  expect(e.pending()).toBe(1)
  e.navigation.dispose()
  expect(e.pending()).toBe(0)
  e.outer.open = e.inner.open = false
  e.browser.dispatchEvent(new Event('hashchange'))
  expect(e.outer.open || e.inner.open).toBe(false)
})


it('corrects after the earlier VitePress capture listener measured a closed same-hash target', () => {
  const e = environment()
  e.navigation.dispose()
  // Characterize the installed router's order: measure now, scroll on the next frame.
  e.browser.addEventListener('click', () => {
    const staleTop = e.browser.scrollY + e.target.getBoundingClientRect().top - 64 + 24
    e.browser.requestAnimationFrame(() => e.browser.scrollTo(0, staleTop))
  }, { capture: true })
  const navigation = installDisclosureNavigation({
    window: e.browser as unknown as Window,
    document: { getElementById: () => e.target } as unknown as Document,
    getRenderedPath: () => '/notes/embedding.html',
    getScrollOffset: () => 64,
  })
  e.outer.open = e.inner.open = false
  e.click()
  e.flush()
  expect(e.outer.open && e.inner.open).toBe(true)
  expect(e.browser.scrollTo.mock.calls).toEqual([[0, 160], [0, 640]])
  navigation.dispose()
})


it.each([
  ['/notes/embedding', '/notes/embedding.html'],
  ['/notes/embedding.html', '/notes/embedding'],
  ['/notes/llms/rag/', '/notes/llms/rag/index.html'],
  ['/notes/llms/rag/index', '/notes/llms/rag/'],
])('reveals current article aliases: address %s, rendered %s', (address, rendered) => {
  const e = environment({ address, rendered })
  expect(e.outer.open && e.inner.open).toBe(true)
  e.outer.open = e.inner.open = false
  e.click({ href: `https://example.test${rendered}#hidden` })
  e.flush()
  expect(e.outer.open && e.inner.open).toBe(true)
  expect(e.browser.scrollTo).toHaveBeenLastCalledWith(0, 640)
  e.navigation.dispose()
})

it.each([
  ['/notes/other', '/notes/embedding.html'],
  ['/other-base/embedding', '/notes/embedding.html'],
])('does not expand stale content across distinct pages or site bases: %s', (address, rendered) => {
  const e = environment({ address, rendered })
  e.browser.dispatchEvent(new Event('hashchange'))
  e.click()
  expect(e.outer.open || e.inner.open).toBe(false)
  expect(e.browser.scrollTo).not.toHaveBeenCalled()
  e.navigation.dispose()
})
