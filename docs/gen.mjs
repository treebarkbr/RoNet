#!/usr/bin/env node
// Generates Markdown API pages for the VitePress docs site from the module
// sources. Reads each module's header comment and the doc-comment run directly
// above every public method, then writes one .md page per module plus an index.
//
// Run from the RoNetV4.1 root:
//   node docs/gen.mjs
//
// The module manifest is static (mirrors the library tree); add new modules
// here when they are added to the library.

import fs from 'node:fs'
import path from 'node:path'

const ROOT = path.resolve(import.meta.dirname, '..')
const OUT_DIR = path.join(ROOT, 'docs', 'vitepress', 'docs', 'api')

const PACKAGES = [
  { name: 'Core', dir: 'core', files: ['Util', 'PRNG', 'Tensor', 'Matrix'] },
  { name: 'NN', dir: 'nn', files: ['Module', 'Inits', 'Linear', 'Activations', 'Norm', 'Dropout', 'Embedding', 'RoPE', 'Attention', 'FFN', 'Sequential'] },
  { name: 'Models', dir: 'models', files: ['MLP', 'TransformerBlock', 'Transformer'] },
  { name: 'Loss', dir: 'loss', files: ['CrossEntropy', 'Losses'] },
  { name: 'Optim', dir: 'optim', files: ['AdamW', 'AdEMAMix', 'Lion', 'NAdamW', 'CautiousAdamW', 'ScheduleFreeAdamW', 'SOAP', 'Muon', 'MuonAdamW'] },
  { name: 'Train', dir: 'train', files: ['Scheduler', 'EMA', 'Serialize', 'Trainer'] },
  { name: 'Data', dir: 'data', files: ['BPE'] },
]

const COMMENT = /^\s*--/
const DIRECTIVE = /^\s*--!/
const FN = /^\s*function\s+([\w_]+)[.:]([\w_]+)/

function stripComment(line) {
  if (DIRECTIVE.test(line)) return null
  if (!COMMENT.test(line)) return null
  const text = line.replace(/^\s*-+\s?/, '').replace(/\s+$/, '')
  return text === '' ? null : text
}

// Angle brackets can look like HTML tags to the markdown engine; escape them.
function escapeAngles(s) {
  return s.replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function parseMethod(line) {
  const m = line.match(FN)
  if (!m) return null
  const cls = m[1]
  const name = m[2]
  const parenStart = line.indexOf('(', m[0].length - 1)
  if (parenStart < 0) return null
  const parenEnd = line.length > parenStart ? line.indexOf(')', parenStart) : -1
  if (parenEnd < 0) return null
  let args = line.slice(parenStart + 1, parenEnd).replace(/^\s*self\s*(?:,\s*)?/, '')
  let sig = name + '(' + args + ')'
  const rest = line.slice(parenEnd + 1).trim()
  if (rest.startsWith(':')) sig += ' -> ' + rest.slice(1).trim()
  return { cls, name, sig }
}

function headerDescription(lines) {
  const out = []
  let started = false
  for (const line of lines) {
    if (COMMENT.test(line)) {
      const t = stripComment(line)
      if (t) {
        out.push(t)
        started = true
      }
    } else if (started) {
      break
    } else if (!/^\s*$/.test(line)) {
      break
    }
  }
  return out
}

// Splits a header description into prose plus the structured `Class:`/`API:`
// meta lines that several modules use to document their factory and surface.
function splitHeaderMeta(desc) {
  const prose = []
  let klass = ''
  let api = ''
  for (const line of desc) {
    if (/^Class:/i.test(line)) klass = line.replace(/^Class:\s*/i, '').trim()
    else if (/^API:/i.test(line)) api = line.replace(/^API:\s*/i, '').trim()
    else prose.push(line)
  }
  return { prose, klass, api }
}

// Collects `local function <name>(...)` signatures so factory-exposed helpers
// can be rendered with their argument lists.
function collectLocalFunctions(lines) {
  const fns = {}
  const re = /^\s*local function\s+(\w+)\s*\(/
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(re)
    if (!m) continue
    let args = ''
    let depth = 0
    let done = false
    for (let j = i; j < lines.length && !done; j++) {
      for (const ch of lines[j]) {
        if (ch === '(') depth++
        else if (ch === ')') {
          depth--
          if (depth === 0) {
            args += ch
            done = true
            break
          }
        }
        if (!done) args += ch
      }
    }
    if (done) {
      const sig = args.slice(args.indexOf('('))
      fns[m[1]] = sig
    }
  }
  return fns
}

// Finds the public helpers a factory returns, e.g. `return { crossEntropy = crossEntropy }`.
function factoryExpose(lines, localFns) {
  const exposed = []
  let start = -1
  let startCol = 0
  for (let i = 0; i < lines.length; i++) {
    const brace = lines[i].indexOf('return {')
    if (brace >= 0) {
      start = i
      startCol = brace + 'return {'.length
    }
  }
  if (start < 0) return exposed
  let chunk = ''
  let depth = 1
  let closed = false
  for (let i = start; i < lines.length && !closed; i++) {
    const line = lines[i]
    for (let c = i === start ? startCol : 0; c < line.length; c++) {
      const ch = line[c]
      if (ch === '{') depth++
      else if (ch === '}' && depth > 0) {
        depth--
        if (depth === 0) {
          closed = true
          break
        }
      }
    }
    chunk += (i === start ? line.slice(startCol) : line) + ' '
  }
  if (!closed) return exposed
  for (const m of chunk.matchAll(/(\w+)\s*=\s*(\w+)/g)) {
    if (localFns[m[2]]) exposed.push({ name: m[1], args: localFns[m[2]] })
  }
  return exposed
}

function renderModule(rel, name) {
  const src = fs.readFileSync(path.join(ROOT, rel), 'utf8')
  const lines = src.split('\n')
  const desc = headerDescription(lines)
  const { prose, klass, api } = splitHeaderMeta(desc)
  const methods = []
  const seenClasses = []
  const classOf = {}
  let pending = []

  for (const line of lines) {
    const t = stripComment(line)
    if (t !== null) {
      pending.push(t)
      continue
    }
    const m = parseMethod(line)
    if (m) {
      if (!classOf[m.cls]) {
        classOf[m.cls] = true
        seenClasses.push(m.cls)
      }
      methods.push({ cls: m.cls, sig: m.sig, doc: pending.join(' ') })
    }
    pending = []
  }

  const localFns = collectLocalFunctions(lines)
  for (const ex of factoryExpose(lines, localFns)) {
    let args = ex.args || ''
    if (args.startsWith('(') && args.endsWith(')')) args = args.slice(1, -1)
    methods.push({ cls: null, sig: `${ex.name}(${args})`, doc: '' })
  }

  const md = []
  md.push(`# ${name}`)
  md.push('')
  md.push(`Source: \`${rel}\``)
  md.push('')
  if (prose.length) md.push(prose.map(escapeAngles).join('  '))
  else md.push('*This module has no description yet.*')
  md.push('')
  if (klass) {
    md.push('## Constructor')
    md.push('')
    md.push('```luau')
    md.push(`Class: ${escapeAngles(klass)}`)
    md.push('```')
    md.push('')
  }
  if (api) {
    md.push('## API')
    md.push('')
    for (const item of api.split(', ')) {
      if (item.trim()) md.push(`- \`${escapeAngles(item.trim())}\``)
    }
    md.push('')
  }
  if (methods.length) {
    md.push('## Methods')
    md.push('')
    let current = '__none__'
    for (const m of methods) {
      if (seenClasses.length > 1 && m.cls && m.cls !== current) {
        md.push(`### Class ${m.cls}`)
        md.push('')
        current = m.cls
      }
      md.push(`#### \`${m.sig}\``)
      md.push('')
      if (m.doc) {
        md.push(escapeAngles(m.doc))
        md.push('')
      }
    }
  }
  return md.join('\n')
}

function renderIndex(pages) {
  const md = []
  md.push('# API Reference')
  md.push('')
  md.push('Every module in the library, generated from the doc comments and')
  md.push('method signatures in the source. Run `node docs/gen.mjs` to')
  md.push('regenerate this page and the module pages.')
  md.push('')
  for (const p of pages) {
    md.push(`## ${p.pkg}`)
    md.push('')
    md.push(`- [${p.title}](./${p.file})`)
    md.push('')
  }
  return md.join('\n')
}

fs.mkdirSync(OUT_DIR, { recursive: true })

const seen = {}
const pages = []
for (const pkg of PACKAGES) {
  for (const name of pkg.files) {
    const rel = `${pkg.dir}/${name}.luau`
    if (seen[rel]) continue
    seen[rel] = true
    const md = renderModule(rel, name)
    const outFile = `${pkg.dir}-${name.toLowerCase()}.md`
    fs.writeFileSync(path.join(OUT_DIR, outFile), md + '\n')
    pages.push({ pkg: pkg.name, file: outFile, title: name })
  }
}

fs.writeFileSync(path.join(OUT_DIR, 'index.md'), renderIndex(pages) + '\n')
console.log(`gen: wrote ${pages.length} module pages + index.md under ${path.relative(ROOT, OUT_DIR)}`)