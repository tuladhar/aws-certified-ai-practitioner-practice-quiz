# AWS Certified AI Practitioner (AIF-C01) - Practice Quiz

Visit: https://aif.purutuladhar.com


## Local Development

1. Run development server
```
cd /Users/puru/next.js/quiz-app/quiz-app2
npx next dev
```
2. Browse https://localhost:3000

## Release

1. Build static files
```
cd /Users/puru/next.js/quiz-app/quiz-app2
npx next build
# A custom script to move single HTML file, such as dashboard.html to dashboard/index.html
# Why? API Routes cannot be used with "output: export" - See next.config.ts
./move.sh
```

2. Upload static files to GitHub
3. Once uploaded, Cloudflare pages will auto-trigger the deploy.
4. Browse https://aif.purutuladhar.com
