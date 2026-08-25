import { readFileSync } from 'node:fs'
import { expect, it } from 'vitest'

it('defines the approved research handbook tokens', () => {
  const css = readFileSync('docs/.vitepress/theme/styles/tokens.css', 'utf8')

  for (const token of ['--nl-paper', '--nl-ink', '--nl-blue', '--nl-green', '--nl-orange', '--nl-line']) {
    expect(css).toContain(token)
  }
})
