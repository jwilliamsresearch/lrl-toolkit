// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  // If you deploy to https://jwilliamsresearch.github.io/lrl-toolkit/ (project
  // Pages), uncomment `base` and set `site` accordingly. For a custom domain or
  // Vercel root deploy, leave `base` unset.
  site: 'https://jwilliamsresearch.github.io',
  // base: '/lrl-toolkit',
  build: { inlineStylesheets: 'auto' },
});
