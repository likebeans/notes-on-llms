import { normalizeUrl } from '../../../content/selectors'

interface NavigationEnvironment {
  window: Window
  document: Document
  getRenderedPath: () => string
  getScrollOffset: () => number
}

/** Add disclosure handling without replacing VitePress's router hooks or navigation. */
export function installDisclosureNavigation(environment: NavigationEnvironment) {
  const { window: browser, document: doc, getRenderedPath, getScrollOffset } = environment
  let frame: number | undefined
  // VitePress may retain a clean URL while route.path has .html (or index).
  // Normalize aliases without stripping the deployment base or conflating pages.
  const isRenderedPath = (path: string) => normalizeUrl(path) === normalizeUrl(getRenderedPath())

  const revealCurrent = (): HTMLElement | null => {
    // During an SPA transition the address bar changes before the old DOM is replaced.
    if (!isRenderedPath(browser.location.pathname) || !browser.location.hash) return null
    let target: HTMLElement | null
    try {
      target = doc.getElementById(decodeURIComponent(browser.location.hash.slice(1)))
    } catch {
      return null
    }
    let revealed = false
    for (let parent = target?.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName !== 'DETAILS') continue
      const details = parent as HTMLDetailsElement
      if (!details.open) {
        details.open = true
        revealed = true
      }
    }
    return revealed ? target : null
  }

  const correctPosition = (target: HTMLElement | null) => {
    if (!target) return
    if (frame !== undefined) browser.cancelAnimationFrame(frame)
    const href = browser.location.href
    frame = browser.requestAnimationFrame(() => {
      frame = undefined
      if (browser.location.href !== href || !isRenderedPath(browser.location.pathname)) return
      // Match VitePress's public scroll offset and heading padding. This runs after
      // its pending scroll when an unchanged hash produced no hashchange event.
      const padding = parseInt(browser.getComputedStyle(target).paddingTop, 10) || 0
      browser.scrollTo(0, browser.scrollY + target.getBoundingClientRect().top - getScrollOffset() + padding)
    })
  }

  const onHashOrHistory = () => { revealCurrent() }
  const onClick = (event: MouseEvent) => {
    if (event.button !== 0 || event.ctrlKey || event.shiftKey || event.altKey || event.metaKey) return
    const element = event.target as Element | null
    if (!element || typeof element.closest !== 'function' || element.closest('button')) return
    const link = element.closest('a')
    if (!link || link.closest('.vp-raw') || link.hasAttribute('download') || link.hasAttribute('target')) return
    const href = link.getAttribute('href')
    if (!href) return
    const destination = new URL(href, browser.location.href)
    if (destination.origin !== browser.location.origin
      || !isRenderedPath(destination.pathname)
      || destination.search !== browser.location.search
      || !destination.hash
      || destination.hash !== browser.location.hash) return
    // Do not skip defaultPrevented: VitePress's earlier capture listener sets it.
    // Changed hashes have already been revealed synchronously by hashchange.
    correctPosition(revealCurrent())
  }

  browser.addEventListener('hashchange', onHashOrHistory)
  browser.addEventListener('popstate', onHashOrHistory)
  browser.addEventListener('click', onClick, { capture: true })
  // Initial hydration can follow a VitePress scroll measured against closed SSR details.
  correctPosition(revealCurrent())
  return {
    revealCurrent,
    dispose() {
      browser.removeEventListener('hashchange', onHashOrHistory)
      browser.removeEventListener('popstate', onHashOrHistory)
      browser.removeEventListener('click', onClick, { capture: true })
      if (frame !== undefined) browser.cancelAnimationFrame(frame)
    },
  }
}
