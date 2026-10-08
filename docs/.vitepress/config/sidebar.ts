import { MODULE_DEFINITIONS } from './modules'

const moduleSidebar = Object.fromEntries(Object.values(MODULE_DEFINITIONS).map(definition => [
  definition.path,
  definition.groups.map(group => ({
    text: group.text,
    ...(group.collapsed === undefined ? {} : { collapsed: group.collapsed }),
    items: group.itemIndexes.map(index => definition.items[index]),
  })),
]))

export const sidebar = {
  '/guide/': [
    {
      text: '学习路线',
      items: [
        { text: '学习路径', link: '/guide/' },
        { text: '学习路线图', link: '/guide/roadmap' },
        { text: '前置知识', link: '/guide/prerequisites' },
        { text: '按问题学习', link: '/guide/problems' },
        { text: '文章检索与学习记录', link: '/guide/library' },
      ],
    },
  ],
  ...moduleSidebar,
  '/practice/': [
    {
      text: '实践项目',
      items: [
        { text: '项目总览', link: '/practice/' },
        { text: '可运行知识助手', link: '/practice/knowledge-assistant' },
      ],
    },
  ],
  '/interviews/': [
    {
      text: '面试专区',
      items: [
        { text: '概述', link: '/interviews/' },
        { text: '系统设计', link: '/interviews/system-design' },
        { text: 'RAG 面试题', link: '/interviews/rag-questions' },
        { text: 'Agent 面试题', link: '/interviews/agent-questions' },
        { text: '训练微调面试题', link: '/interviews/training-questions' },
        { text: '代码题', link: '/interviews/coding' },
      ],
    },
  ],
  '/reference/': [
    {
      text: '速查手册',
      items: [
        { text: '概述', link: '/reference/' },
        { text: '术语表', link: '/reference/glossary' },
        { text: 'Checklist', link: '/reference/checklists' },
        { text: '评估指标', link: '/reference/metrics' },
        { text: '模板', link: '/reference/templates' },
      ],
    },
  ],
  '/resources/': [
    {
      text: '资源库',
      items: [
        { text: '概述', link: '/resources/' },
        { text: '技术核验与更新', link: '/resources/updates' },
        { text: '视频', link: '/resources/videos' },
        { text: '论文', link: '/resources/papers' },
        { text: '博客', link: '/resources/blogs' },
        { text: 'CSDN 镜像', link: '/resources/csdn' },
        { text: '开源项目', link: '/resources/repos' },
      ],
    },
  ],
  '/about/': [
    {
      text: '关于',
      items: [
        { text: '关于本站', link: '/about/' },
        { text: '更新日志', link: '/about/changelog' },
      ],
    },
  ],
}
