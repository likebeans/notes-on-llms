import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { afterEach, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { compileScript, parse } from 'vue/compiler-sfc'

// Use the existing tsx dependency's transpiler; no DOM or additional test framework.
const require = createRequire(import.meta.url)
const { transformSync } = createRequire(require.resolve('tsx'))('esbuild')
const filename = new URL('../../docs/.vitepress/theme/components/article/ReadingProgress.vue', import.meta.url)
const { descriptor } = parse(readFileSync(filename, 'utf8'), { filename: filename.pathname })
const compiled = compileScript(descriptor, { id: 'reading-progress-test', inlineTemplate: true })
const output = transformSync(compiled.content, { loader: 'ts', format: 'cjs' }).code
const module = { exports: {} as { default: Vue.Component } }
const contentUpdates = new Set<() => void>()
const vitepress = {
  onContentUpdated(callback: () => void) {
    contentUpdates.add(callback)
    Vue.onUnmounted(() => contentUpdates.delete(callback))
  },
}
new Function('require', 'module', 'exports', output)(
  (name: string) => name === 'vitepress' ? vitepress : Vue, module, module.exports,
)

type HostNode = { children: HostNode[]; props: Record<string, unknown>; text: string; parent?: HostNode }
const node = (text = ''): HostNode => ({ children: [], props: {}, text })
const renderer = Vue.createRenderer<HostNode, HostNode>({
  createElement: () => node(), createText: node, createComment: node,
  insert(child, parent) { child.parent = parent; parent.children.push(child) },
  remove(child) { if (child.parent) child.parent.children = child.parent.children.filter(item => item !== child) },
  parentNode: child => child.parent ?? null,
  nextSibling: () => null,
  setText(child, text) { child.text = text },
  setElementText(child, text) { child.text = text },
  patchProp(child, key, _previous, value) { child.props[key] = value },
})

const cleanups: (() => void)[] = []
afterEach(() => {
  cleanups.splice(0).forEach(cleanup => cleanup())
  contentUpdates.clear()
  vi.unstubAllGlobals(); vi.restoreAllMocks()
})

function addFrames<T extends EventTarget>(browser: T) {
  const frames = new Map<number, FrameRequestCallback>()
  let id = 0
  return Object.assign(browser, {
    requestAnimationFrame(callback: FrameRequestCallback) { frames.set(++id, callback); return id },
    cancelAnimationFrame(frame: number) { frames.delete(frame) },
    flushFrames() { const pending = [...frames.values()]; frames.clear(); pending.forEach(callback => callback(0)) },
    pendingFrames: () => frames.size,
  })
}

it('updates local progress on scroll/resize without changing root attributes and unbinds on unmount', async () => {
  const rootAttributeChanges: string[] = []
  const documentElement = {
    scrollHeight: 2000,
    style: {
      setProperty: (name: string) => rootAttributeChanges.push(`style:${name}`),
      removeProperty: (name: string) => rootAttributeChanges.push(`style:${name}`),
    },
    setAttribute: (name: string) => rootAttributeChanges.push(name),
    removeAttribute: (name: string) => rootAttributeChanges.push(name),
  }
  const browser = addFrames(Object.assign(new EventTarget(), { scrollY: 0, innerHeight: 1000 }))
  const added = vi.spyOn(browser, 'addEventListener')
  const removed = vi.spyOn(browser, 'removeEventListener')
  vi.stubGlobal('document', { documentElement })
  vi.stubGlobal('window', browser)
  const container = node()
  const app = renderer.createApp(module.exports.default)
  app.mount(container)
  const progressbar = container.children[0]
  try {
    browser.scrollY = 500
    browser.dispatchEvent(new Event('scroll'))
    browser.flushFrames()
    await Vue.nextTick()
    expect(progressbar.props['aria-valuenow']).toBe(50)
    expect(progressbar.children[1].text).toBe('50%')
    expect(progressbar.props.style).toMatchObject({ '--nl-reading-progress': '50%' })

    browser.innerHeight = 500
    browser.dispatchEvent(new Event('resize'))
    browser.flushFrames()
    await Vue.nextTick()
    expect(progressbar.props['aria-valuenow']).toBe(33)
    expect(progressbar.children[1].text).toBe('33%')
  } finally {
    app.unmount()
  }
  for (const event of ['scroll', 'resize']) {
    const callback = added.mock.calls.find(([type]) => type === event)?.[1]
    expect(callback).toEqual(expect.any(Function))
    expect(removed).toHaveBeenCalledWith(event, callback)
  }
  expect(rootAttributeChanges).toEqual([])
})


function mountProgress() {
  const documentElement = { scrollHeight: 2000 }
  const content = {}
  const browser = addFrames(Object.assign(new EventTarget(), { scrollY: 500, innerHeight: 1000 }))
  vi.stubGlobal('document', { documentElement, querySelector: () => content })
  vi.stubGlobal('window', browser)
  const container = node()
  const app = renderer.createApp(module.exports.default)
  app.mount(container)
  cleanups.push(() => app.unmount())
  return { documentElement, browser, content, bar: container.children[0], app }
}

it('refreshes after disclosure/chart height changes without a scroll event', async () => {
  let notify: ResizeObserverCallback | undefined
  const disconnect = vi.fn()
  const observed: unknown[] = []
  vi.stubGlobal('ResizeObserver', class {
    constructor(callback: ResizeObserverCallback) { notify = callback }
    observe(target: unknown) { observed.push(target) }
    disconnect = disconnect
  })
  const e = mountProgress()
  e.browser.flushFrames()
  await Vue.nextTick()
  expect(e.bar.props['aria-valuenow']).toBe(50)
  expect(observed).toContain(e.content)
  e.documentElement.scrollHeight = 3000
  notify?.([], {} as ResizeObserver)
  e.browser.flushFrames()
  await Vue.nextTick()
  expect(e.bar.props['aria-valuenow']).toBe(25)
  e.app.unmount(); cleanups.length = 0
  expect(disconnect).toHaveBeenCalled()
})

it('refreshes reused article content without ResizeObserver or a changed scroll position', async () => {
  vi.stubGlobal('ResizeObserver', undefined)
  const e = mountProgress()
  e.browser.flushFrames()
  await Vue.nextTick()
  expect(e.bar.props['aria-valuenow']).toBe(50)
  e.documentElement.scrollHeight = 5000
  contentUpdates.forEach(callback => callback())
  e.browser.flushFrames()
  await Vue.nextTick()
  expect(e.bar.props['aria-valuenow']).toBe(13)
})

it('coalesces events into one frame and cancels pending work on unmount', async () => {
  vi.stubGlobal('ResizeObserver', undefined)
  const e = mountProgress()
  e.browser.flushFrames()
  for (const type of ['scroll', 'resize', 'scroll']) e.browser.dispatchEvent(new Event(type))
  expect(e.browser.pendingFrames()).toBe(1)
  e.app.unmount(); cleanups.length = 0
  expect(e.browser.pendingFrames()).toBe(0)
  e.browser.dispatchEvent(new Event('scroll'))
  expect(e.browser.pendingFrames()).toBe(0)
  expect(contentUpdates.size).toBe(0)
})
