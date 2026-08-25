import { describe, expect, it } from 'vitest'
import { MODULE_DEFINITIONS } from '../../docs/.vitepress/config/modules'
import { findModuleStep } from '../../docs/.vitepress/theme/components/article/moduleProgress'

describe('module progress', () => {
  it.each([
    ['rag', '/llms/rag/csdn_articles'],
    ['mcp', '/llms/mcp/practice'],
  ] as const)('hides progress for an unlisted %s article', (module, currentUrl) => {
    expect(findModuleStep(MODULE_DEFINITIONS[module].items, currentUrl)).toBeUndefined()
  })
})
