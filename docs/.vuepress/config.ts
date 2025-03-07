import { viteBundler } from '@vuepress/bundler-vite'
import { defaultTheme } from '@vuepress/theme-default'
import { defineUserConfig } from 'vuepress'
import MarkdownItKatex from "markdown-it-katex";

export default defineUserConfig({
  base: '/cv-blog/',
  bundler: viteBundler(),
  extendsMarkdown: (md) => {
    md.use(MarkdownItKatex);
  },
  theme: defaultTheme(
    {
      navbar: [
        {
          text: "GitHub",
          link: "https://github.com/yibotongxue/cv-blog"
        }
      ],
      sidebar: [
        {
          text: "计算机视觉笔记",
          children: [
            "/notes/dof",
            "/notes/camera-coordinate",
            "/notes/3d-vision",
            "/notes/sfm",
            "/notes/iMAP",
            "/notes/gaussian-splatting-slam",
          ]
        },
      ]
    }
  ),
  lang: 'zh-CN',
  title: '计算机视觉',
  description: '这是我的计算机视觉笔记托放的地方',
})
