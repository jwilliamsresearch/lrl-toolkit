# lrl-toolkit — landing page

Marketing landing page for [lrl-toolkit](https://github.com/jwilliamsresearch/lrl-toolkit).
Built with [Astro](https://astro.build). No runtime framework, no external
network calls — a single static page with an inline canvas animation.

## Develop

```bash
cd site
npm install
npm run dev       # http://localhost:4321
```

## Build

```bash
npm run build     # -> site/dist/
npm run preview   # serve the built output
```

## Highlights

- **Hero:** a canvas ASCII globe made of morphing multilingual glyphs (Latin,
  Arabic, Ge'ez, Tibetan, Devanagari, Cyrillic, Khmer). Text left, globe right;
  stacks on mobile. See [`src/components/Hero.astro`](src/components/Hero.astro).
- **Theme:** teal/emerald on ink, light + dark. Follows the system preference,
  remembers a manual toggle in `localStorage`, and applies before first paint
  (no flash). The animation is theme-aware and honors `prefers-reduced-motion`.
- **Content** is sourced from the repo README (pipeline stages, features,
  languages, models, connectors). Edit the `*.astro` components in
  [`src/components/`](src/components/).

## Deploy

The page is fully static (`site/dist/`), so any static host works.

- **Vercel / Netlify:** set the project root to `site/`, build command
  `npm run build`, output `dist`.
- **GitHub Pages (project site):** uncomment `base: '/lrl-toolkit'` in
  [`astro.config.mjs`](astro.config.mjs) and set `site` to
  `https://jwilliamsresearch.github.io`, then publish `site/dist/`.
- **Custom domain / root deploy:** leave `base` unset.
