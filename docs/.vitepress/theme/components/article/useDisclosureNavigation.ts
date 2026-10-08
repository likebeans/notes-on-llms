import { onMounted, onUnmounted } from 'vue'
import { getScrollOffset, onContentUpdated, useRoute } from 'vitepress'
import { installDisclosureNavigation } from './disclosureNavigation'

export function useDisclosureNavigation() {
  const route = useRoute()
  let navigation: ReturnType<typeof installDisclosureNavigation> | undefined

  // Content hooks run after the destination DOM is patched, before the router's
  // nextTick scroll. They also cover cross-page history and asynchronously loaded pages.
  onContentUpdated(() => navigation?.revealCurrent())
  onMounted(() => {
    navigation = installDisclosureNavigation({
      window, document, getRenderedPath: () => route.path, getScrollOffset,
    })
  })
  onUnmounted(() => navigation?.dispose())
}
