#!/bin/bash

# HakiTrade Pre-Deployment Check Script
# This script ensures your project is ready for a production push.

echo "🚀 Starting HakiTrade Pre-Deployment Checks..."

# 1. Update Backend Requirements
echo "📂 Updating requirements.txt..."
./venv/Scripts/pip freeze > requirements.txt
echo "✅ requirements.txt updated."

# 2. Test Frontend Build
echo "🏗️ Testing Frontend Build..."
cd frontend
npm run build
if [ $? -eq 0 ]; then
    echo "✅ Frontend build successful."
else
    echo "❌ Frontend build failed! Check for errors in App.jsx or App.css."
    exit 1
fi
cd ..

# 3. Git Status Check
echo "🔍 Checking Git status..."
if ! command -v git &> /dev/null
then
    echo "⚠️ Git is not installed. Please install it to push to GitHub."
else
    git status
fi

echo "------------------------------------------------"
echo "✨ HakiTrade is ready for deployment!"
echo "1. Push to GitHub: git add . && git commit -m 'Release v1.0' && git push"
echo "2. Backend will auto-deploy on Render/Railway."
echo "3. Frontend will auto-deploy on Vercel/Netlify."
echo "------------------------------------------------"
