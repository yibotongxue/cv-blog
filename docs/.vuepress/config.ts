import { viteBundler } from '@vuepress/bundler-vite'
import { defaultTheme } from '@vuepress/theme-default'
import { defineUserConfig } from 'vuepress'

export default defineUserConfig({
  base: '/cv-blog/',
  bundler: viteBundler(),
  theme: defaultTheme(
    {
      navbar: [
        {
          text: "GitHub",
          link: "https://github.com/yibotongxue/cv-blog"
        }
      ],
    }
  ),
  lang: 'zh-CN',
  title: '计算机视觉',
  description: '这是我的计算机视觉笔记托放的地方',
})
