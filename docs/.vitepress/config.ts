import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Notes on LLMs',
  description: '大模型学习笔记 - RAG, Agent, 训练微调',
  
  base: '/notes-on-llms/',
  
  lang: 'zh-CN',
  
  lastUpdated: true,
  
  head: [
    ['link', { rel: 'icon', href: '/notes-on-llms/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#5f67ee' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:locale', content: 'zh_CN' }],
    ['meta', { name: 'og:site_name', content: 'Notes on LLMs' }],
  ],

  themeConfig: {
    logo: '/logo.svg',
    
    nav: [
      { text: '首页', link: '/' },
      { text: '学习路线', link: '/guide/roadmap' },
      { text: 'RAG', link: '/rag/' },
      { text: 'Agent', link: '/agent/' },
      { text: 'Prompt', link: '/prompt/' },
      { text: 'MCP', link: '/mcp/' },
      { text: '训练微调', link: '/training/' },
      { text: '面试', link: '/interviews/' },
      { text: '速查', link: '/reference/glossary' },
      { text: '资源', link: '/resources/papers' },
      { text: '关于', link: '/about/' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: '学习路线',
          items: [
            { text: '学习路线图', link: '/guide/roadmap' },
            { text: '前置知识', link: '/guide/prerequisites' },
          ]
        }
      ],
      '/rag/': [
        {
          text: 'RAG 专区',
          items: [
            { text: '概述', link: '/rag/' },
            { text: '范式演进', link: '/rag/paradigms' },
            { text: '文档切分', link: '/rag/chunking' },
            { text: 'Embedding', link: '/rag/embedding' },
            { text: '向量数据库', link: '/rag/vector-db' },
            { text: '检索策略', link: '/rag/retrieval' },
            { text: '重排序', link: '/rag/rerank' },
            { text: '评估', link: '/rag/evaluation' },
            { text: '生产实践', link: '/rag/production' },
          ]
        }
      ],
      '/agent/': [
        {
          text: 'Agent 专区',
          items: [
            { text: '概述', link: '/agent/' },
            { text: '工具调用', link: '/agent/tool-calling' },
            { text: '规划', link: '/agent/planning' },
            { text: '记忆', link: '/agent/memory' },
            { text: '多智能体', link: '/agent/multi-agent' },
            { text: '评估', link: '/agent/evaluation' },
            { text: '安全', link: '/agent/safety' },
          ]
        }
      ],
      '/prompt/': [
        {
          text: 'Prompt 专区',
          items: [
            { text: '概述', link: '/prompt/' },
            { text: '基础技术', link: '/prompt/basics' },
            { text: '高级技术', link: '/prompt/advanced' },
            { text: '上下文工程', link: '/prompt/context' },
            { text: '安全测试', link: '/prompt/security' },
          ]
        }
      ],
      '/mcp/': [
        {
          text: 'MCP 专区',
          items: [
            { text: '概述', link: '/mcp/' },
            { text: '快速入门', link: '/mcp/quickstart' },
            { text: '核心概念', link: '/mcp/concepts' },
            { text: '高级功能', link: '/mcp/advanced' },
          ]
        }
      ],
      '/training/': [
        {
          text: '训练与微调',
          items: [
            { text: '概述', link: '/training/' },
            { text: '数据处理', link: '/training/data' },
            { text: 'SFT 监督微调', link: '/training/sft' },
            { text: 'DPO', link: '/training/dpo' },
            { text: 'RLHF', link: '/training/rlhf' },
            { text: 'LoRA', link: '/training/lora' },
            { text: '评估', link: '/training/eval' },
            { text: '部署推理', link: '/training/serving' },
          ]
        }
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
          ]
        }
      ],
      '/reference/': [
        {
          text: '速查手册',
          items: [
            { text: '术语表', link: '/reference/glossary' },
            { text: 'Checklist', link: '/reference/checklists' },
            { text: '评估指标', link: '/reference/metrics' },
            { text: '模板', link: '/reference/templates' },
          ]
        }
      ],
      '/resources/': [
        {
          text: '资源库',
          items: [
            { text: '论文', link: '/resources/papers' },
            { text: '博客', link: '/resources/blogs' },
            { text: '开源项目', link: '/resources/repos' },
          ]
        }
      ],
      '/about/': [
        {
          text: '关于',
          items: [
            { text: '关于本站', link: '/about/' },
            { text: '更新日志', link: '/about/changelog' },
          ]
        }
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/likebeans/notes-on-llms' }
    ],

    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭'
                }
              }
            }
          }
        }
      }
    },

    footer: {
      message: '基于 VitePress 构建',
      copyright: 'Copyright © 2024-present likebeans'
    },

    outline: {
      label: '页面导航',
      level: [2, 3]
    },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    darkModeSwitchLabel: '主题',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '返回顶部',
  },

  markdown: {
    lineNumbers: true,
    image: {
      lazyLoading: true
    }
  }
})
