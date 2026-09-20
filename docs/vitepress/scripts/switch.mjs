import fs from 'node:fs'
import path from 'node:path'
import { spawn } from 'node:child_process'

const root = path.resolve(import.meta.dirname, '..')
const THEME_JSON = path.join(root, '.vitepress', '.dev.pid')
const THEME_LOG = path.join(root, '.vitepress', '.dev.log')
const ALLOWED = ['carbon', 'default', 'duxweb']

const theme = process.argv[2] || 'carbon'
const port = Number(process.argv[3] || 5173)

if (!ALLOWED.includes(theme)) {
  console.error(`unknown theme "${theme}" — use one of: ${ALLOWED.join(', ')}`)
  process.exit(1)
}

if (fs.existsSync(THEME_JSON)) {
  try {
    process.kill(-Number(fs.readFileSync(THEME_JSON, 'utf8').trim()), 'SIGTERM')
  } catch {}
  try {
    fs.unlinkSync(THEME_JSON)
  } catch {}
}

fs.writeFileSync(
  path.join(root, '.vitepress', 'theme', 'index.js'),
  `export { default } from '../themes/${theme}/index.js'\n`,
)

const bin = path.join(root, 'node_modules', '.bin', 'vitepress')
const out = fs.openSync(THEME_LOG, 'a')
const child = spawn(bin, ['dev', '--port', String(port), '--host'], {
  env: { ...process.env, VITE_THEME: theme },
  detached: true,
  stdio: ['ignore', out, out],
})
child.unref()
fs.writeFileSync(THEME_JSON, String(child.pid))

console.log(`theme=${theme} port=${port} pid=${child.pid} log=${THEME_LOG}`)