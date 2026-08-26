import { describe, expect, it } from 'vitest'
import { mermaid } from '../../docs/.vitepress/config/mermaid'

describe('mermaid config', () => {
  it('renders flowcharts with fixed SVG dimensions so the outer document container owns dragging and scrolling', () => {
    expect(mermaid.flowchart).toMatchObject({
      useMaxWidth: false,
    })
  })
})
