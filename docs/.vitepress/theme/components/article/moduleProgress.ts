import { normalizeUrl } from '../../../content/selectors'

export interface ModuleProgressItem {
  link: string
}

export function findModuleStep(items: ModuleProgressItem[], currentUrl: string): number | undefined {
  const currentIndex = items.findIndex(
    item => normalizeUrl(item.link) === normalizeUrl(currentUrl),
  )

  return currentIndex < 0 ? undefined : currentIndex + 1
}
