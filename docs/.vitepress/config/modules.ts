import type { ModuleKey } from '../content/model'
import { normalizeUrl } from '../content/selectors'

type ModuleItem = { text: string; link: string }
type ModuleGroup = { text: string; collapsed?: boolean; itemIndexes: number[] }

interface ModuleDefinition {
  title: string
  shortDescription: string
  path: string
  items: ModuleItem[]
  groups: ModuleGroup[]
}

export const MODULE_DEFINITIONS: Record<Exclude<ModuleKey, 'site'>, ModuleDefinition> = {
  rag: {
    title: 'RAG',
    shortDescription: '检索增强生成的核心组件、优化与生产实践。',
    path: '/llms/rag/',
    items: [
      { text: '概述', link: '/llms/rag/' },
      { text: '范式演进', link: '/llms/rag/paradigms' },
      { text: '文档切分', link: '/llms/rag/chunking' },
      { text: 'Embedding', link: '/llms/rag/embedding' },
      { text: '向量数据库', link: '/llms/rag/vector-db' },
      { text: '检索策略', link: '/llms/rag/retrieval' },
      { text: '重排序', link: '/llms/rag/rerank' },
      { text: '评估', link: '/llms/rag/evaluation' },
      { text: '生产实践', link: '/llms/rag/production' },
    ],
    groups: [
      { text: 'RAG 专区', itemIndexes: [0] },
      { text: '核心组件', collapsed: false, itemIndexes: [1, 2, 3, 4] },
      { text: '检索与优化', collapsed: false, itemIndexes: [5, 6] },
      { text: '生产部署', collapsed: false, itemIndexes: [7, 8] },
    ],
  },
  agent: {
    title: 'Agent',
    shortDescription: '智能体的设计模式、协作机制与生产实践。',
    path: '/llms/agent/',
    items: [
      { text: '概述', link: '/llms/agent/' },
      { text: '提示链', link: '/llms/agent/prompt-chain' },
      { text: '路由', link: '/llms/agent/routing' },
      { text: '并行化', link: '/llms/agent/parallelization' },
      { text: '反思', link: '/llms/agent/reflection' },
      { text: '工具调用', link: '/llms/agent/tool-calling' },
      { text: '规划', link: '/llms/agent/planning' },
      { text: '多智能体协作', link: '/llms/agent/multi-agent' },
      { text: '记忆管理', link: '/llms/agent/memory' },
      { text: '推理技术', link: '/llms/agent/reasoning' },
      { text: '异常处理与恢复', link: '/llms/agent/exception-handling' },
      { text: '人机协同', link: '/llms/agent/human-in-the-loop' },
      { text: '智能体间通信', link: '/llms/agent/a2a' },
      { text: '资源感知优化', link: '/llms/agent/resource-optimization' },
      { text: '护栏与安全', link: '/llms/agent/safety' },
      { text: '评估与监控', link: '/llms/agent/evaluation-monitoring' },
      { text: '优先级排序', link: '/llms/agent/prioritization' },
      { text: '探索与发现', link: '/llms/agent/exploration' },
      { text: '评估方法', link: '/llms/agent/evaluation' },
    ],
    groups: [
      { text: 'Agent 专区', itemIndexes: [0] },
      { text: '核心设计模式', collapsed: false, itemIndexes: [1, 2, 3, 4, 5, 6, 7] },
      { text: '高级设计模式', collapsed: false, itemIndexes: [8, 9] },
      { text: '集成设计模式', collapsed: false, itemIndexes: [10, 11] },
      { text: '生产设计模式', collapsed: false, itemIndexes: [12, 13, 14, 15, 16, 17] },
      { text: '其他资源', collapsed: true, itemIndexes: [18] },
    ],
  },
  prompt: {
    title: 'Prompt',
    shortDescription: '提示与上下文工程的基础、进阶和安全实践。',
    path: '/llms/prompt/',
    items: [
      { text: '概述', link: '/llms/prompt/' },
      { text: '提示工程基础', link: '/llms/prompt/basics' },
      { text: '上下文工程', link: '/llms/prompt/context' },
      { text: '高级提示技术', link: '/llms/prompt/advanced' },
      { text: '安全测试', link: '/llms/prompt/security' },
    ],
    groups: [
      { text: 'Prompt 专区', itemIndexes: [0] },
      { text: '基础技术', collapsed: false, itemIndexes: [1, 2] },
      { text: '高级技术', collapsed: false, itemIndexes: [3, 4] },
    ],
  },
  mcp: {
    title: 'MCP',
    shortDescription: '模型上下文协议的入门、核心概念与高级功能。',
    path: '/llms/mcp/',
    items: [
      { text: '概述', link: '/llms/mcp/' },
      { text: '快速入门', link: '/llms/mcp/quickstart' },
      { text: '核心概念', link: '/llms/mcp/concepts' },
      { text: '高级功能', link: '/llms/mcp/advanced' },
    ],
    groups: [
      { text: 'MCP 专区', itemIndexes: [0] },
      { text: '入门指南', collapsed: false, itemIndexes: [1, 2] },
      { text: '进阶内容', collapsed: false, itemIndexes: [3] },
    ],
  },
  training: {
    title: 'Training',
    shortDescription: '训练、对齐、评估与部署推理的系统学习路径。',
    path: '/llms/training/',
    items: [
      { text: '概述', link: '/llms/training/' },
      { text: '数据处理', link: '/llms/training/data' },
      { text: 'SFT 监督微调', link: '/llms/training/sft' },
      { text: 'DPO', link: '/llms/training/dpo' },
      { text: 'RLHF', link: '/llms/training/rlhf' },
      { text: 'LoRA', link: '/llms/training/lora' },
      { text: '评估', link: '/llms/training/eval' },
      { text: '部署推理', link: '/llms/training/serving' },
    ],
    groups: [{ text: '训练与微调', itemIndexes: [0, 1, 2, 3, 4, 5, 6, 7] }],
  },
  multimodal: {
    title: 'Multimodal',
    shortDescription: '多模态模型的编码、连接、训练和部署评测。',
    path: '/llms/multimodal/',
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
    ],
    groups: [{ text: '多模态', itemIndexes: [0, 1, 2, 3, 4, 5, 6, 7, 8] }],
  },
}

export function findModuleOrder(module: ModuleKey, url: string): number | undefined {
  const definition = MODULE_DEFINITIONS[module as keyof typeof MODULE_DEFINITIONS]
  const index = definition?.items.findIndex(item => normalizeUrl(item.link) === normalizeUrl(url)) ?? -1
  return index < 0 ? undefined : index
}
