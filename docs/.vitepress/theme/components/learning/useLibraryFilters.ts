import { onMounted, onUnmounted, reactive, toRefs, watch } from 'vue'
import { normalizeUrl } from '../../../content/selectors'
import { libraryStateFromSearch, libraryUrlWithState } from './libraryState'

export function useLibraryFilters() {
  const state = reactive(libraryStateFromSearch(''))
  let activePath = ''
  let restoring = false
  const onLibraryPage = () => activePath !== '' && normalizeUrl(window.location.pathname) === activePath

  const restore = () => {
    if (!onLibraryPage()) return
    restoring = true
    Object.assign(state, libraryStateFromSearch(window.location.search))
    restoring = false
  }

  // A new filter starts at the first batch; restoring history keeps its batch size.
  watch(() => [state.query, state.module, state.level, state.kind, state.saved], () => {
    if (!restoring) state.limit = 20
  }, { flush: 'sync' })

  watch(state, () => {
    if (!onLibraryPage()) return
    const next = libraryUrlWithState(window.location.href, state)
    if (next !== window.location.href) {
      // Preserve VitePress's scroll restoration data and avoid a history entry per keystroke.
      window.history.replaceState(window.history.state, '', next)
    }
  })

  onMounted(() => {
    activePath = normalizeUrl(window.location.pathname)
    restore()
    window.addEventListener('popstate', restore)
  })
  onUnmounted(() => {
    activePath = ''
    window.removeEventListener('popstate', restore)
  })

  const reset = () => Object.assign(state, libraryStateFromSearch(''))
  return { ...toRefs(state), reset }
}
