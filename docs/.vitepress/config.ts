import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { buildPageHead } from './config/head'
import { mermaid } from './config/mermaid'
import { nav } from './config/nav'
import { SITE_BASE, SITE_DESCRIPTION, SITE_TITLE, SITE_URL } from './config/site'
import { sidebar } from './config/sidebar'

export default withMermaid(defineConfig({
  title: SITE_TITLE,
  description: SITE_DESCRIPTION,
  base: SITE_BASE,
  srcExclude: [
    'superpowers/**',
    '开发计划.md',
    ...(process.env.NODE_ENV === 'production' ? ['_drafts/**', '**/*.draft.md'] : []),
  ],
  lang: 'zh-CN',
  lastUpdated: true,
  sitemap: {
    hostname: SITE_URL,
    transformItems: items => items.filter(item => !item.url.includes('/_drafts/') && !item.url.includes('/superpowers/')),
  },
  transformHead({ page, title, description, pageData }) {
    return page === '404.md'
      ? []
      : buildPageHead({ page, title, description, frontmatter: pageData.frontmatter })
  },
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${SITE_BASE}logo.svg` }],
    ['meta', { name: 'theme-color', content: '#174ca6' }],
    ['meta', { property: 'og:locale', content: 'zh_CN' }],
    ['meta', { property: 'og:site_name', content: SITE_TITLE }],
  ],
  themeConfig: {
    logo: '/logo.svg',
    nav,
    sidebar,
    socialLinks: [
      { icon: 'github', link: 'https://github.com/likebeans/notes-on-llms' },
    ],
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档',
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭',
                },
              },
            },
          },
        },
      },
    },
    footer: {
      message: '基于 VitePress 构建',
      copyright: 'Copyright © 2024-present likebeans',
    },
    outline: {
      label: '页面导航',
      level: [2, 3],
    },
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short',
      },
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },
    darkModeSwitchLabel: '主题',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '返回顶部',
  },
  markdown: {
    lineNumbers: true,
    image: {
      lazyLoading: true,
    },
  },
  vite: {
    optimizeDeps: {
      include: ['mermaid', 'dayjs'],
    },
    ssr: {
      noExternal: ['mermaid'],
    },
  },
  mermaid,
}))
