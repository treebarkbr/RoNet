import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitepress'

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const themeIndex = path.join(root, '.vitepress', 'theme', 'index.js')
let theme = process.env.VITE_THEME
if (!theme) {
  try {
    const m = fs
      .readFileSync(themeIndex, 'utf8')
      .match(/from\s+'\.\.\/themes\/(\w+)\/index\.js'/)
    if (m) theme = m[1]
  } catch {}
}
theme = theme || 'carbon'
const apiDir = path.join(root, 'docs', 'api')

const groups = [
  { name: 'Core', prefix: 'core-' },
  { name: 'NN layers', prefix: 'nn-' },
  { name: 'Models', prefix: 'models-' },
  { name: 'Losses', prefix: 'loss-' },
  { name: 'Optimizers', prefix: 'optim-' },
  { name: 'Training', prefix: 'train-' },
  { name: 'Data', prefix: 'data-' },
]

function titleOf(file) {
  try {
    const t = fs.readFileSync(path.join(apiDir, file), 'utf8')
    const m = t.match(/^# (.+)$/m)
    if (m) return m[1].trim()
  } catch {}
  return path.basename(file, '.md')
}

function sidebar() {
  let files = []
  try {
    files = fs.readdirSync(apiDir).filter((f) => f.endsWith('.md') && f !== 'index.md')
  } catch {}
  return [
    {
      text: 'Guide',
      items: [
        { text: 'Quick start', link: '/guide/quickstart' },
        { text: 'Roblox port', link: '/guide/roblox' },
      ],
    },
    ...groups.map((g) => ({
      text: g.name,
      collapsed: false,
      items: files
        .filter((f) => f.startsWith(g.prefix))
        .sort()
        .map((f) => ({
          text: titleOf(f),
          link: '/api/' + f.replace(/\.md$/, ''),
        })),
    })),
  ]
}

const base = {
  title: 'RoNetV4.1',
  description: 'A modular neural network library for Luau',
  lang: 'en-US',
  head: [['link', { rel: 'icon', href: '/logo.svg', type: 'image/svg+xml' }]],
  srcDir: 'docs',
  cleanUrls: true,
  lastUpdated: true,
  themeConfig: {
    logo: '/logo.svg',
    nav: [
      { text: 'Guide', link: '/guide/quickstart' },
      { text: 'Roblox', link: '/guide/roblox' },
      { text: 'API', link: '/api/' },
    ],
    sidebar: sidebar(),
    search: { provider: 'local' },
    outline: { label: 'On this page', level: [2, 3] },
  },
}

let config = base

if (theme === 'carbon') {
  const { baseConfig } = await import('vitepress-carbon/config')
  config = { ...base, ...baseConfig }
} else if (theme === 'duxweb') {
  const { withDuxTheme } = await import('@duxweb/vitepress-theme/config')
  config = withDuxTheme(base)
}

export default defineConfig(config)