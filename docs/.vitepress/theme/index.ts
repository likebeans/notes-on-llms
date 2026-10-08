import DefaultTheme from 'vitepress/theme-without-fonts'
import type { Theme } from 'vitepress'
import Layout from './Layout.vue'
import LearningObjectives from './components/article/LearningObjectives.vue'
import SourceList from './components/article/SourceList.vue'
import HomePage from './components/home/HomePage.vue'
import ArticleLibrary from './components/learning/ArticleLibrary.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('LearningObjectives', LearningObjectives)
    app.component('SourceList', SourceList)
    app.component('HomePage', HomePage)
    app.component('ArticleLibrary', ArticleLibrary)
  },
} satisfies Theme
