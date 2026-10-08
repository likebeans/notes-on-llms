import { LEVEL_LABELS, MODULE_LABELS } from '../../../content/discovery'

export interface LibraryState {
  query: string
  module: string
  level: string
  kind: string
  saved: string
  limit: number
}

const choices = {
  module: Object.keys(MODULE_LABELS),
  level: Object.keys(LEVEL_LABELS),
  kind: ['mainline', 'mirror'],
  saved: ['bookmarked', 'completed'],
}
const validLimit = (value: number) => Number.isInteger(value) && value >= 20 && value <= 1000 && value % 20 === 0

export function libraryStateFromSearch(search: string): LibraryState {
  const params = new URLSearchParams(search)
  const pick = (key: keyof typeof choices) => {
    const value = params.get(key) ?? ''
    return choices[key].includes(value) ? value : 'all'
  }
  const limit = Number(params.get('limit'))
  return {
    query: (params.get('q') ?? '').slice(0, 200),
    module: pick('module'), level: pick('level'), kind: pick('kind'), saved: pick('saved'),
    limit: validLimit(limit) ? limit : 20,
  }
}

export function libraryUrlWithState(href: string, state: LibraryState): string {
  const url = new URL(href)
  for (const key of ['q', 'module', 'level', 'kind', 'saved', 'limit']) url.searchParams.delete(key)
  if (state.query) url.searchParams.set('q', state.query.slice(0, 200))
  for (const key of Object.keys(choices) as (keyof typeof choices)[]) {
    if (choices[key].includes(state[key])) url.searchParams.set(key, state[key])
  }
  if (validLimit(state.limit) && state.limit !== 20) url.searchParams.set('limit', String(state.limit))
  return url.href
}
