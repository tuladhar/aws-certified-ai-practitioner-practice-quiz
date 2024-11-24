# **Free Quiz** - AWS Certified AI Practitioner (AIF-C01)

👇 Challenge your AI knowledge:
- https://aif.purutuladhar.com

![Nov-15-2024 3-20-01 PM](https://github.com/user-attachments/assets/ef6d2893-cbac-42ca-b400-a6ddde017848)

## Local Development

1. Run development server
```bash
cd /Users/puru/next.js/quiz-app/quiz-app2
npx next dev
```
2. Browse https://localhost:3000

## Release

1. Build static files
```bash
cd /Users/puru/next.js/quiz-app/quiz-app2
npx next build
# A custom script to move single HTML file, such as dashboard.html to dashboard/index.html
# Why? API Routes cannot be used with "output: export" - See next.config.ts
./move.sh
```

2. Upload static files to GitHub
3. Once uploaded, Cloudflare pages will auto-trigger the deploy.
4. Browse https://aif.purutuladhar.com
