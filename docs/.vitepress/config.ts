import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
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
      {
        text: 'LLMs',
        items: [
          { text: 'RAG', link: '/llms/rag/' },
          { text: 'Agent', link: '/llms/agent/' },
          { text: '训练微调', link: '/llms/training/' },
          { text: '多模态', link: '/llms/multimodal/' },
          { text: 'Prompt', link: '/llms/prompt/' },
          { text: 'MCP', link: '/llms/mcp/' },
        ]
      },
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
      '/llms/rag/': [
        {
          text: 'RAG 专区',
          items: [
            { text: '概述', link: '/llms/rag/' },
          ]
        },
        {
          text: '核心组件',
          collapsed: false,
          items: [
            { text: '范式演进', link: '/llms/rag/paradigms' },
            { text: '文档切分', link: '/llms/rag/chunking' },
            { text: 'Embedding', link: '/llms/rag/embedding' },
            { text: '向量数据库', link: '/llms/rag/vector-db' },
          ]
        },
        {
          text: '检索与优化',
          collapsed: false,
          items: [
            { text: '检索策略', link: '/llms/rag/retrieval' },
            { text: '重排序', link: '/llms/rag/rerank' },
          ]
        },
        {
          text: '生产部署',
          collapsed: false,
          items: [
            { text: '评估', link: '/llms/rag/evaluation' },
            { text: '生产实践', link: '/llms/rag/production' },
          ]
        }
      ],
      '/llms/agent/': [
        {
          text: 'Agent 专区',
          items: [
            { text: '概述', link: '/llms/agent/' },
          ]
        },
        {
          text: '核心设计模式',
          collapsed: false,
          items: [
            { text: '提示链', link: '/llms/agent/prompt-chain' },
            { text: '路由', link: '/llms/agent/routing' },
            { text: '并行化', link: '/llms/agent/parallelization' },
            { text: '反思', link: '/llms/agent/reflection' },
            { text: '工具调用', link: '/llms/agent/tool-calling' },
            { text: '规划', link: '/llms/agent/planning' },
            { text: '多智能体协作', link: '/llms/agent/multi-agent' },
          ]
        },
        {
          text: '高级设计模式',
          collapsed: false,
          items: [
            { text: '记忆管理', link: '/llms/agent/memory' },
            { text: '推理技术', link: '/llms/agent/reasoning' },
          ]
        },
        {
          text: '集成设计模式',
          collapsed: false,
          items: [
            { text: '异常处理与恢复', link: '/llms/agent/exception-handling' },
            { text: '人机协同', link: '/llms/agent/human-in-the-loop' },
          ]
        },
        {
          text: '生产设计模式',
          collapsed: false,
          items: [
            { text: '智能体间通信', link: '/llms/agent/a2a' },
            { text: '资源感知优化', link: '/llms/agent/resource-optimization' },
            { text: '护栏与安全', link: '/llms/agent/safety' },
            { text: '评估与监控', link: '/llms/agent/evaluation-monitoring' },
            { text: '优先级排序', link: '/llms/agent/prioritization' },
            { text: '探索与发现', link: '/llms/agent/exploration' },
          ]
        },
        {
          text: '其他资源',
          collapsed: true,
          items: [
            { text: '评估方法', link: '/llms/agent/evaluation' },
          ]
        }
      ],
      '/llms/prompt/': [
        {
          text: 'Prompt 专区',
          items: [
            { text: '概述', link: '/llms/prompt/' },
          ]
        },
        {
          text: '基础技术',
          collapsed: false,
          items: [
            { text: '提示工程基础', link: '/llms/prompt/basics' },
            { text: '上下文工程', link: '/llms/prompt/context' },
          ]
        },
        {
          text: '高级技术',
          collapsed: false,
          items: [
            { text: '高级提示技术', link: '/llms/prompt/advanced' },
            { text: '安全测试', link: '/llms/prompt/security' },
          ]
        }
      ],
      '/llms/mcp/': [
        {
          text: 'MCP 专区',
          items: [
            { text: '概述', link: '/llms/mcp/' },
          ]
        },
        {
          text: '入门指南',
          collapsed: false,
          items: [
            { text: '快速入门', link: '/llms/mcp/quickstart' },
            { text: '核心概念', link: '/llms/mcp/concepts' },
          ]
        },
        {
          text: '进阶内容',
          collapsed: false,
          items: [
            { text: '高级功能', link: '/llms/mcp/advanced' },
          ]
        }
      ],
      '/llms/training/': [
        {
          text: '训练与微调',
          items: [
            { text: '概述', link: '/llms/training/' },
            { text: '数据处理', link: '/llms/training/data' },
            { text: 'SFT 监督微调', link: '/llms/training/sft' },
            { text: 'DPO', link: '/llms/training/dpo' },
            { text: 'RLHF', link: '/llms/training/rlhf' },
            { text: 'LoRA', link: '/llms/training/lora' },
            { text: '评估', link: '/llms/training/eval' },
            { text: '部署推理', link: '/llms/training/serving' },
          ]
        }
      ],
      '/llms/multimodal/': [
        {
          text: '多模态',
          items: [
            { text: '概述', link: '/llms/multimodal/' },
            { text: '视觉编码器', link: '/llms/multimodal/vision-encoder' },
            { text: '模态连接器', link: '/llms/multimodal/connector' },
            { text: '多模态架构', link: '/llms/multimodal/architecture' },
            { text: '数据工程', link: '/llms/multimodal/data' },
            { text: '扩散模型', link: '/llms/multimodal/diffusion' },
            { text: 'RAG 与智能体', link: '/llms/multimodal/rag-agent' },
            { text: '统一架构', link: '/llms/multimodal/unified' },
            { text: '部署与评测', link: '/llms/multimodal/deployment' },
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
  },
  
  vite: {
    optimizeDeps: {
      include: ['mermaid', 'dayjs'],
    },
    ssr: {
      noExternal: ['mermaid']
    }
  },
  
  mermaid: {
    theme: 'base',
    themeVariables: {
      // 节点颜色
      primaryColor: '#e0e7ff',
      primaryTextColor: '#1e293b',
      primaryBorderColor: '#6366f1',
      // 次要节点
      secondaryColor: '#fef3c7',
      secondaryTextColor: '#1e293b',
      secondaryBorderColor: '#f59e0b',
      // 第三节点
      tertiaryColor: '#dcfce7',
      tertiaryTextColor: '#1e293b',
      tertiaryBorderColor: '#22c55e',
      // 连接线
      lineColor: '#6366f1',
      // 文字
      textColor: '#1e293b',
      fontSize: '14px',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      // subgraph 子图
      clusterBkg: '#f1f5f9',
      clusterBorder: '#cbd5e1',
      titleColor: '#1e293b',
      // 边标签
      edgeLabelBackground: '#ffffff'
    }
  }
}))
