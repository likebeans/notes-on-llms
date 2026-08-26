import type { ContentStatus } from '../../../content/model'

export type ContentStatusCopy = {
  label: string
  guidance: string
}

const copies: Record<ContentStatus, ContentStatusCopy> = {
  verified: {
    label: '已核验',
    guidance: '核心结论已按当前来源复核，仍建议结合引用资料判断适用边界。',
  },
  'needs-review': {
    label: '待复核',
    guidance: '这页包含可能受时效影响的结论，请结合最新资料和实践环境再采用。',
  },
  opinion: {
    label: '观点内容',
    guidance: '这页包含作者观点或经验判断，不应当作所有场景通用的事实结论。',
  },
  historical: {
    label: '历史资料',
    guidance: '这页主要用于理解技术演进与背景，具体做法应以当前文档为准。',
  },
  draft: {
    label: '草稿',
    guidance: '这页还处在草稿状态，结构、事实和示例都可能继续调整。',
  },
}

export function getContentStatusCopy(status: ContentStatus): ContentStatusCopy {
  return copies[status]
}
